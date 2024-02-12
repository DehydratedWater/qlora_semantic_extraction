import asyncio
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator


from datetime import datetime
from custom.synthetic_data.prompt_running.extract_topics import generate_general_categories_async
from custom.synthetic_data.prompt_running.extract_article_summary import generate_article_summaries_async
from custom.synthetic_data.inserting_articles import insert_articles
# os environment variables must be declared before importing datasets
# once datasets is imported, it will use current values and does not update them

import os

from custom.synthetic_data.prompt_running.extract_topics_base_on_summaries import generate_topics_from_parts_and_summaries_async
from custom.synthetic_data.prompt_running.extract_raw_relations import extract_raw_relations
from custom.synthetic_data.prompt_running.generate_relation_tables import generate_relations_tables


try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []

print(user_paths)

import sys
print(sys.path)

import os
os.environ['HF_HOME'] = '/opt/airflow/data/hf/cache/'
os.environ['HF_DATASETS_CACHE'] = '/opt/airflow/data/hf/datasets/'
os.environ['TRANSFORMERS_CACHE'] = '/opt/airflow/data/hf/models/'
from datasets import load_dataset







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



 


with DAG(
    'prepare_synthetic_data_v5',
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

    generate_article_summaries = PythonOperator(
        task_id='generate_article_summaries',
        python_callable=generate_article_summaries_async,
        op_kwargs={
            'num_of_llms': 4,
            'max_tokens': 4096,
            'amount_to_process': 10000, 
            'summary_variant': 0, 
            'overwrite_variant': False
        }
    )


    generate_general_categories = PythonOperator(
        task_id='generate_general_categories',
        python_callable=generate_topics_from_parts_and_summaries_async,
        op_kwargs={
            'num_of_llms': 4,
            'max_tokens': 4096,
            'amount_to_process': 10850, 
            'summary_variant': 0, 
            'overwrite_variant': False
        }
    )

    extract_raw_relations_op = PythonOperator(
        task_id='extract_raw_relations',
        python_callable=extract_raw_relations,
        op_kwargs={
            'summary_variant': 0,
            'topics_variant': 0,
            'relations_variant': 0,
        }
    )

    generate_relations_tables_op = PythonOperator(
        task_id='generate_relations_tables',
        python_callable=generate_relations_tables,
        op_kwargs={
            'relations_variant': 0,
        }
    )



    download_dataset >> test_dataset
    download_dataset >> insert_articles
    initialize_database >> insert_articles
    insert_articles >> generate_article_summaries
    generate_article_summaries >> generate_general_categories
    generate_general_categories >> extract_raw_relations_op
    extract_raw_relations_op >> generate_relations_tables_op
