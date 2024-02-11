import asyncio
import json
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.base import BaseLLMOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from transformers import LlamaTokenizerFast
from airflow.providers.postgres.hooks.postgres import PostgresHook
from langchain.chains import AnalyzeDocumentChain
from custom.synthetic_data.prompt_running.article_summary_prompt import article_summary_prompt
from langchain.text_splitter import TokenTextSplitter


def remove_references(entities):
    return [entity for entity in entities if not entity.startswith("@xmath") and not entity.startswith("@xchem") and not entity.startswith("@xref") and not entity.startswith("@xlink") and not entity.startswith("@xlink")]

def test_and_unify_types(parsed_json):
    assert "description" in parsed_json
    assert "list_of_entities" in parsed_json
    assert "relations" in parsed_json

    assert type(parsed_json["description"]) == str
    assert type(parsed_json["list_of_entities"]) == list
    assert type(parsed_json["relations"]) == list

    for entity in parsed_json["list_of_entities"]:
        assert type(entity) == str
    
    forbiden_entities = {"dopamine", "D1", "D2", "D3", "D4", "D5", "dopamine receptors", "adenosine", "A1", "A2A", "A2B", "A3", "caffeine"}

    fixed_relations = {}

    for relation in parsed_json["relations"]:
        # print(relation)

        assert type(relation) == dict

        if "description" not in relation:
            continue        
        
        description = None
        source_entities = set()
        target_entities = set()
        strength = set()
        for key in relation:
            if key.startswith("source_"):
                if type(relation[key]) == str:
                    source_entities.add(relation[key])
                elif type(relation[key]) == list:
                    source_entities.update(relation[key])
            if key.startswith("target_"):
                if type(relation[key]) == str:
                    target_entities.add(relation[key])
                elif type(relation[key]) == list:
                    target_entities.update(relation[key])

            if key == "description":
                description = relation[key]
                assert type(description) == str

            if key == "strength":
                strength = {relation[key]}

            if description not in fixed_relations:
                fixed_relations["description"] = {
                    "description": description,
                    "source_entities": source_entities,
                    "target_entities": target_entities,
                    "strength": strength
                }
            else:
                fixed_relations["description"]["source_entities"].update(source_entities)
                fixed_relations["description"]["target_entities"].update(target_entities)
                fixed_relations["description"]["strength"].update(strength)


            if "dopamine" in description or "adenosine" in description:
                del fixed_relations["description"]
                continue

            parsed_json["list_of_entities"].extend(list(source_entities))
            parsed_json["list_of_entities"].extend(list(target_entities))


    for fr in fixed_relations:
        fixed_relations[fr]["source_entities"] = list(fixed_relations[fr]["source_entities"])
        fixed_relations[fr]["target_entities"] = list(fixed_relations[fr]["target_entities"])
        fixed_relations[fr]["strength"] = list(fixed_relations[fr]["strength"])
            
        if len(fixed_relations[fr]["strength"]) >= 1:
            fixed_relations[fr]["strength"] = fixed_relations[fr]["strength"][0]
        else:
            del fixed_relations[fr]["strength"]
        

    parsed_json["list_of_entities"] = list(set(parsed_json["list_of_entities"]).difference(forbiden_entities))

    parsed_json["list_of_entities"] = remove_references(parsed_json["list_of_entities"])
    
    assert len(parsed_json["list_of_entities"]) > 0

    parsed_json["relations"] = [fixed_relations[fr] for fr in fixed_relations]
    
    assert len(parsed_json["relations"]) > 0

    parsed_json["section_description"] = parsed_json["description"]
    del parsed_json["description"]



def clean_up_rows(rows):
    parsed_rows = []
    tested = 0
    correct = 0
    unfit = 0

    for row in rows[:]:
        tested += 1
        text = row[1]
        text = text.replace('    ', '')
        text = text.replace('"]\n"', '"],\n"')
        text = text.replace('"\n"', '",\n"')
        text = text.replace('\n",\n', '",\n')
        text = text.replace('\n",\n"', '",\n"')
        text = text.replace('\n" ,\n', '",\n')
        text = text.replace('\n\n', '\n')
        text = text.replace('",\n]', '"\n]')
        

        if '"description":' not in text[:200]:
            text = '{\n"description": "'+text
            text = text.split('}')
            text = "}".join(text[:-1])

            if len(text) > 0 and text[-1] != '}':
                text += '\n}'
            # print(text)
        else:
            text = text[text.find("{"):]
            text = text.split('}')
            text = "}".join(text[:-1])
            if len(text) > 0 and text[-1] != '\n}':
                text += '}'
            # print(text)
            pass

        sections = text.split('"list_of_entities"')

        if len(sections) < 2:
            unfit += 1
            # print("Unfit: ", row[1])
            continue

        first_sections = sections[0].split('{\n')

        selected_description = first_sections[-1]
        selected_description = selected_description.replace('\n', ' ')
        selected_description = selected_description.strip()
        selected_description = selected_description.replace('  ', ' ').replace('  ', ' ')
        
        if selected_description.endswith('"'):
            selected_description = selected_description + ','

        if not selected_description.endswith('" ,') and not selected_description.endswith('",'):
            selected_description = selected_description + '",'

        text = '{\n'+selected_description+'\n"list_of_entities"'+sections[1]
        text = text.replace("]}", "]")
        text = text.replace(" \n", " ")

        splitted_lines = text.split("\n")

        rejoined = ""
        for line in splitted_lines:
            
            if line.endswith('",'):
                rejoined += line + "\n"
            else:
                rejoined += line 
                
        text = rejoined

        

        if text.endswith(']'):
            text = text + '}'

        if text.startswith('{",'):
            # print(first_sections[:2])
            
            potencial_start = '{'+first_sections[1]


            potencial_start = potencial_start.replace("/n", "").replace("}", "").strip()
            if potencial_start.endswith('"'):
                potencial_start = potencial_start + ','
                
            text = potencial_start + text[3:]
            # print(text)

        if not text.endswith(']}') and  not text.endswith('}]}') and text.endswith('}'):
            text = text + ']}'

        text = text.replace('"Ekmann-Hilton" ', '\\"Ekmann-Hilton\\" ')
        text = text.replace('  ', ' ').replace('  ', ' ')
        text = text.replace(']"', '],"')
        text = text.replace('},]', '}]')
        text = text.replace('}{', '},{')
        text = text.replace(']]', ']')
        text = text.replace('"] "', '"], "')
        text = text.replace(']{', '],"relations":{')
        text = text.replace('}}', '}')
        text = text.replace('" ', '"')

        try:
            parsed = json.loads(text)
            # print(parsed.keys())
            test_and_unify_types(parsed)
            # print(parsed)
            parsed_rows.append(
                {
                    "part_id": row[0],
                    "part_topics": parsed,
                    "topics_variant": row[2],
                    "topics_id": row[3],
                    "article_summary_id": row[4]
                }
            )
            correct += 1
        except:
            print("---------------------------------------------------------")
            # print(row[1])
            # print(descriptions)
            # for s in first_sections:
            #     print("- ",s)
            # print(selected_description)
            print(text)

        # print("---------------------------------------------------------")
            

    print("Tested: ", tested)
    print("Correct: ", correct)
    print("Unfit: ", unfit)
    print("Achievable: ", correct/(tested-unfit))
    print("All: ", correct/(tested))

    return parsed_rows


def extract_raw_relations(summary_variant, topics_variant):
    hook = PostgresHook(postgres_conn_id='synthetic_data')
    
    query = f"""
select ept.part_id, ept.part_topics, ept.topics_variant, ept.topics_id, ept.article_summary_id 
from extracted_part_topics ept 
where ept.topics_variant = {topics_variant} and 
(ept.part_id not in (select part_id from short_part_summary where summary_variant={summary_variant}) or 
ept.article_summary_id not in (select article_summary_id from short_part_summary where summary_variant={summary_variant}));
"""
    rows = hook.get_records(query)

    cleaned_rows = clean_up_rows(rows)

    hook.insert_rows(
        table="short_part_summary",
        rows=[
            (
                r['part_id'], 
                summary_variant, 
                r['part_topics']['section_description'], 
                r['article_summary_id']
            ) for r in cleaned_rows
        ],
        replace=False,
        commit_every=1000,
        target_fields=["part_id", "summary_variant", "part_summary", "article_summary_id"],
    )

#     query = f"""
# select ept.part_id, ept.part_topics, ept.topics_variant, ept.topics_id, ept.article_summary_id 
# from extracted_part_topics ept 
# where ept.topics_variant = {topics_variant} and ept.part_id not in (select part_id from short_part_summary) and ept.article_summary_id not in (select article_summary_id from short_part_summary);
# """
#     rows = hook.get_records(query)

#     cleaned_rows = clean_up_rows(rows)



    return cleaned_rows[:10]