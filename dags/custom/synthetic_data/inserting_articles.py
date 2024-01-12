import os
os.environ['HF_HOME'] = '/opt/airflow/data/hf/cache/'
os.environ['HF_DATASETS_CACHE'] = '/opt/airflow/data/hf/datasets/'
os.environ['TRANSFORMERS_CACHE'] = '/opt/airflow/data/hf/models/'

from datasets import load_dataset
from langchain.chains import LLMChain
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from transformers import LlamaTokenizerFast
from langchain.text_splitter import TokenTextSplitter



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
