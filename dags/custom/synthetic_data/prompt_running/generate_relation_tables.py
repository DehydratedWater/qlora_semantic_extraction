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

def generate_relations_tables(relations_variant):
    hook = PostgresHook(postgres_conn_id='synthetic_data')
    
    query = f"""
select err.relation_id, err.raw_relation_text 
from extracted_relations_raw err 
where err.relations_variant = {relations_variant}
;
"""
    rows = hook.get_records(query)

    for relation_id, raw_relation_text in rows:
        json_data = json.loads(raw_relation_text)
        entities = json_data['list_of_entities']
        relations = json_data['relations']

        entities_with_id = [(relation_id, entity) for entity in entities]

        hook.insert_rows(
            table="relation_objects",
            rows=entities_with_id,
            replace=False,
            commit_every=1000,
            target_fields=["source_id", "object_name"],
        )

        for r in relations:
            description = r['description']
            source_entities = r['source_entities']
            target_entities = r['target_entities']

            double_quoted = [e.replace("'", "''") for e in source_entities + target_entities]

            if len(double_quoted) == 0:
                continue

            sql_entities = "(" + ", ".join([f"'{e}'" for e in double_quoted]) + ")"

            query = f"""
select ro.object_id, ro.object_name, ro.source_id 
from relation_objects ro 
where ro.object_name in {sql_entities} and ro.source_id = {relation_id}
;
"""
            
            object_rows = hook.get_records(query)

            name_to_object_id = {r[1]: r[0] for r in object_rows}

            list_of_relations = []
            for source_entity in source_entities:
                if source_entity not in name_to_object_id:
                    continue
                source_id = name_to_object_id[source_entity]
                for target_entity in target_entities:
                    if target_entity not in name_to_object_id:
                        continue
                    target_id = name_to_object_id[target_entity]
                    list_of_relations.append((relation_id, source_id, target_id, description))
                    

            hook.insert_rows(
                table="relations",
                rows=list_of_relations,
                replace=False,
                commit_every=1000,
                target_fields=["source_id", "object_id", "subject_id", "relation_type"],
            )