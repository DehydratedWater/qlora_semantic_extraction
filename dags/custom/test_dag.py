from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta

default_args = {
    'owner': 'dehydratedwater',
    'depends_on_past': False,
    'start_date': datetime(2020, 1, 1),
}


def print_hello():
    return 'Hello world!'

with DAG(
    dag_id='test_dag',
    default_args=default_args,
    description='Test DAG',
    catchup=False,
    schedule_interval='@once',
) as dag:
    hello_operator = PythonOperator(
        task_id='hello_task',
        python_callable=print_hello,
    )

    hello_operator
