import asyncio
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from langchain.chat_models import ChatOpenAI
from datetime import datetime, timedelta
from langchain.text_splitter import TokenTextSplitter
from transformers import LlamaTokenizerFast
# os environment variables must be declared before importing datasets
# once datasets is imported, it will use current values and does not update them
import os
os.environ['HF_HOME'] = '/opt/airflow/data/hf/cache/'
os.environ['HF_DATASETS_CACHE'] = '/opt/airflow/data/hf/datasets/'
os.environ['TRANSFORMERS_CACHE'] = '/opt/airflow/data/hf/models/'
from datasets import load_dataset
from langchain.chains import LLMChain






default_args = {
    'owner': 'dehydratedwater',
    'depends_on_past': True,
    'start_date': datetime(2021, 1, 1),
    'retries': 1,
}


def download_dataset():
    
    os.makedirs('/opt/airflow/data/hf/cache', exist_ok=True)
    os.makedirs('/opt/airflow/data/hf/datasets', exist_ok=True)
    os.makedirs('/opt/airflow/data/hf/models', exist_ok=True)
    load_dataset(
        path='scientific_papers', 
        name='arxiv',
        cache_dir='/opt/airflow/data/hf/datasets',
        trust_remote_code=True,
        )
    load_dataset(
        path='scientific_papers', 
        name='pubmed',
        cache_dir='/opt/airflow/data/hf/datasets',
        trust_remote_code=True,
        )


def test_dataset():
    dataset = load_dataset(
        path='scientific_papers', 
        name='arxiv',
        cache_dir='/opt/airflow/data/hf/datasets',
        trust_remote_code=True,
        )
    print(dataset['train'][0])


def insert_articles(inserted_dataset, chunk_size, chunk_overlap):
    dataset = load_dataset(
        path='scientific_papers', 
        name=inserted_dataset,
        cache_dir='/opt/airflow/data/hf/datasets',
        trust_remote_code=True,
        )

    hook = PostgresHook(postgres_conn_id='synthetic_data')
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer", 
                                                   cache_dir='/opt/airflow/data/hf/models',)
    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
    )



    max_article_index = hook.get_records(sql=f"SELECT max(article_id) FROM articles  WHERE dataset = '{inserted_dataset}'")
    max_part_index = hook.get_records(sql=f"""
        SELECT max(part_index) 
        FROM article_part_register 
        WHERE chunk_size = {chunk_size} AND chunk_overlap = {chunk_overlap}""")

    article_index = 0
    if len(max_article_index) > 0 and max_article_index[0][0] is not None:
        article_index = max_article_index[0][0] + 1

    part_index = 0
    if len(max_part_index) > 0 and max_part_index[0][0] is not None:
        part_index = max_part_index[0][0] + 1

    all_articles = int(len(dataset['train']))

    for index, article in enumerate(dataset['train']):
        if index < article_index:
            continue


        is_alredy_inserted = hook.get_records(sql=f"SELECT article_id FROM articles WHERE article_id = {article_index}")
        if len(is_alredy_inserted) > 0:
            article_index += 1
            continue

        articles = []
        article_part_register = []

        article_text = article['article'].replace('\x00', '')
        article_abstract = article['abstract'].replace('\x00', '')
        article_section_names = article['section_names'].replace('\x00', '')
        
        texts = text_splitter.split_text(article_text)

        if len(texts) == 0:
            article_index += 1
            continue


        articles.append([
                article_index,
                article_abstract,
                article_section_names,
                inserted_dataset
            ])

        map_part_indexes_to_texts = {}

        for text in texts:

            article_part_register.append([
                part_index,
                chunk_size,
                chunk_overlap,
                article_index,
            ])
            map_part_indexes_to_texts[part_index] = text
            part_index += 1


        hook.insert_rows(
            table='articles',
            rows=articles,
            target_fields=['article_id', 'abstract', 'section_names', 'dataset'],
            commit_every=1,
            replace=True,
            replace_index=['article_id'], # ON CONFLICT (article_id) DO UPDATE
            replace_target=None, # ON CONFLICT DO NOTHING
        )

        hook.insert_rows(
            table='article_part_register',
            rows=article_part_register,
            target_fields=['part_index', 'chunk_size', 'chunk_overlap', 'article_id'],
            commit_every=100,
            replace=True,
            replace_index=['part_index', 'chunk_size', 'chunk_overlap', 'article_id'], # ON CONFLICT (article_id) DO UPDATE
            replace_target=None, # ON CONFLICT DO NOTHING
        )
        
        min_part_index = min(map_part_indexes_to_texts.keys())
        max_part_index = max(map_part_indexes_to_texts.keys())

        mapping_index_to_id = hook.get_records(sql=f"""
            SELECT part_index, part_id 
            FROM article_part_register
            WHERE article_id = {article_index} AND chunk_size = {chunk_size} AND chunk_overlap = {chunk_overlap} AND part_index >= {min_part_index} AND part_index <= {max_part_index}
        """)


        part_index_to_id = [
            [
                record[1],
                map_part_indexes_to_texts[record[0]],
            ] for record in mapping_index_to_id
        ]

        hook.insert_rows(
            table='article_parts',
            rows=part_index_to_id,
            target_fields=['part_id', 'part_text'],
            commit_every=100,
            replace=True,
            replace_index=['part_id'], # ON CONFLICT (article_id) DO UPDATE
            replace_target=None, # ON CONFLICT DO NOTHING
        )

        article_index += 1

        print(f'{index} / {all_articles} = {index / all_articles * 100:0.2f}%')

    
extracton_prompt = """
[INST] 
Provided below segment of the text, list all specialistic terms, and ideas contained in this text, that list will be later used for mining relations between concepts in order of building graph of semantic relations. 

Example:
Format data as a JSON containing {{"segment_summary": "summary ...", extracted_categories:{{"<<name of category>>": [{{"description": "What is idea number 1...","name": "idea_1"}}, {{"description": "What is term number 2...","name": "term_2"}}, ...], "<<name of category>>": [{{"description": "What is idea number 1...","name": "idea_1"}}, {{"description": "What is term number 2...","name": "term_2"}}, {{"description": "What is idea number 3...","name": "idea_3"}}...]}}. Categories should be general.

Text to extract categories: ```
{text_to_extract}
```

Use text above to extract concepts, ideas, people, ect, find name for the category and format them into flat JSON containing lists. Return JSON within 
json``` {{...}} ```
[/INST]
Before creating JSON with the short segment summary and results, containing categories matched with context independent description of interaction, I will list categories that fits into nodes of semantic graph, contained inside provided text and give them more compact and universal name:
"""
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.base import BaseLLMOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from pydantic import BaseModel


class ExtractedJson(BaseModel):
    segment_summary: str
    extracted_categories: dict[str, list[dict[str, str]]]



class ExtractJsonParser(BaseLLMOutputParser):
    def parse_result(self, result: list[Generation]) -> list[dict]:
        final_results = []

        for r in result:
            extracted_json = None
            

            fail = False
            try:
                raw_text = r.text
                print(raw_text)
                start = raw_text.find("{")
                end = len(raw_text) - raw_text[::-1].find("}")
                raw_text = raw_text[start:end]
                extracted_json = json.loads(raw_text)
                final_results.append(extracted_json)
                fail = False
            except:
                fail = True
            
            if fail:
                try:
                    raw_text = r.text
                    print(raw_text)
                    start = raw_text.find("{")
                    end = len(raw_text) - raw_text[::-1].find("]")
                    raw_text = raw_text[start:end]+"}"
                    print(raw_text)
                    extracted_json = json.loads(raw_text)
                    final_results.append(extracted_json)
                    fail = False
                except:
                    fail = True

            if fail:
                final_results.append(None)
            
                    
        return final_results
    
async def run_chain(sr: list, chain: LLMChain):
    for i, r in enumerate(sr):
        result = await chain.arun(r[1])
        print(result)
        print(f"Finished batch {i} / {len(sr)}")
        # print(len(sr))

async def generate_general_categories_base():

    model = "models/llama-2-13b-chat.Q4_K_M.gguf"

    number_of_llms = 4 

    llms = [
        ChatOpenAI(temperature=0.7,
                    model=model, 
                    openai_api_base=f"http://llm-server-{i}:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=4098,
                    request_timeout=5000,
                    max_retries=0,
                    model_kwargs={
                        "logit_bias": {},
                    },
                    streaming=False,
                    )
        for i in range(1, 1+number_of_llms)
    ]

    print("Loading LLMs")

    prompt = PromptTemplate(
        template=extracton_prompt, 
        input_variables=["text_to_extract"],
        
    )

    print("Creating chains")

    chain_array = [
        LLMChain(
            llm=llm, 
            prompt=prompt,
            output_parser=ExtractJsonParser(),
            verbose=True,
        ) for llm in llms
    ]

    print("Loading tokenizer")

    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer", 
                                                   cache_dir='/opt/airflow/data/hf/models',)
    hook = PostgresHook(postgres_conn_id='synthetic_data')
    results = hook.get_records(sql=f"SELECT * FROM article_parts LIMIT 20")

    print("Splitting results")

    num_of_batches = number_of_llms
    size = len(results)
    batch_size = int(size / num_of_batches)

    splitted_results = [results[r*batch_size: (r+1)*batch_size] for r in range(num_of_batches)]

    print("Number of arrays",len(splitted_results))
    list_of_tasks = []
    for sr, chain in zip(splitted_results, chain_array):
        print("Creating task")
        list_of_tasks.append(run_chain(sr, chain))
    
    
    await asyncio.gather(*list_of_tasks)
    # segment = result[0][1]

    # len_of_input_prompt = len(tokenizer.encode(prompt.template.format(text_to_extract=segment)))
    # print(f"len_of_input_prompt: {len_of_input_prompt}")

    # generated_json = chain1.run(segment)
    # print(generated_json)

def generate_general_categories_async():
   loop = asyncio.get_event_loop()
   result = loop.run_until_complete(generate_general_categories_base())
   return result


with DAG(
    'prepare_synthetic_data_v4',
    default_args=default_args,
    description='Prepare synthetic data',
    schedule_interval='@once',
    catchup=False,
    template_searchpath='/opt/airflow/sql'
) as dag:
    
    download_dataset = PythonOperator(
        task_id='download_dataset',
        python_callable=download_dataset,
    )

    initialize_database = PostgresOperator(
        task_id='initialize_database',
        postgres_conn_id='synthetic_data',
        sql='synthetic_data/initialize_database.sql',
    )

    insert_articles = PythonOperator(
        task_id='insert_articles',
        python_callable=insert_articles,
        op_kwargs={
            'inserted_dataset': 'arxiv',
            'chunk_size': 1024, 
            'chunk_overlap': 256,
            },
    )



    test_dataset = PythonOperator(
        task_id='test_dataset',
        python_callable=test_dataset,
    )

    generate_general_categories = PythonOperator(
        task_id='generate_general_categories',
        python_callable=generate_general_categories_async,
    )

    download_dataset >> test_dataset
    download_dataset >> insert_articles
    initialize_database >> insert_articles
    insert_articles >> generate_general_categories
