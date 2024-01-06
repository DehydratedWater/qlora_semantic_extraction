from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta

# os environment variables must be declared before importing datasets
# once datasets is imported, it will use current values and does not update them
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

with DAG(
    'prepare_synthetic_data',
    default_args=default_args,
    description='Prepare synthetic data',
    schedule_interval='@once',
    catchup=False,
) as dag:
    
    download_dataset = PythonOperator(
        task_id='download_dataset',
        python_callable=download_dataset,
    )

    download_dataset
