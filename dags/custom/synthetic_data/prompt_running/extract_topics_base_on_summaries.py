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
from custom.synthetic_data.prompt_running.topic_extraction_prompt import extracton_prompt
from langchain.text_splitter import TokenTextSplitter


async def run_chains(sr: list, chain: LLMChain, register: set[int], tokenizer: LlamaTokenizerFast, summary_variant: int):

    for i, r in enumerate(sr):
        if i in register:
            continue
        else:
            register.add(i)

            input_data = {"text_to_extract": str(r[2]), "summary": str(r[3])}
            # print(f"Przycinanie 1: {input_data}")
            constructed_prompt = chain.prep_prompts([input_data])[0][0].text
            # print("Created prompt 1")
            # print(constructed_prompt)
            prompt_length = len(tokenizer.encode(text=constructed_prompt))
            # print(f"Promp lenght 1: {prompt_length}")
            if prompt_length > 2048:
                text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=int(1024*1.5), 
                    chunk_overlap=256,
                )

                text_to_extract = text_splitter.split_text(str(r[2]))[0]
                summary = text_splitter.split_text(str(r[3]))[0]

                input_data = {"text_to_extract": text_to_extract, "summary": summary}
                # print(f"Przycinanie 2: {input_data}")
                constructed_prompt = chain.prep_prompts([input_data])[0][0].text
                # print("Created prompt 2")
                # print(constructed_prompt)
                prompt_length = len(tokenizer.encode(text=constructed_prompt))
                # print(f"Promp lenght 2: {prompt_length}")

            try:
                result = await chain.arun(input_data)
                # result = await chain.llm.abatch([input_data])
                print("PROMPT ****************")
                print(constructed_prompt)
                print("RESULT ****************")
                print(result)

                print(f"Finished batch {i} / {len(sr)}")

                hook = PostgresHook(postgres_conn_id='synthetic_data')
                hook.insert_rows(
                    table="extracted_part_topics",
                    rows=[(r[1], 0, result, r[0])],
                    replace=False,
                    commit_every=1,
                    target_fields=["part_id", "topics_variant", "part_topics", "article_summary_id"],
                )
            except Exception as e:
                print(f"Error while processing {i} / {len(sr)}")
                print(e)
                register.remove(i)
                
                continue
            



async def generate_topics_from_parts_and_summaries_base(
        num_of_llms: int, 
        max_tokens: int, 
        amount_to_process: int, 
        summary_variant: int, 
        overwrite_variant: bool
        ):

    model = "models/llama-2-13b-chat.Q4_K_M.gguf"

    number_of_llms = num_of_llms

    llms = [
        ChatOpenAI(temperature=0.7,
                    model=model, 
                    openai_api_base=f"http://llm-server-{i}:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=max_tokens,
                    request_timeout=1800,
                    # verbose=True,
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
        input_variables=["abstract", "list_of_sections"],
        
    )

    

    print("Creating chains")

    chain_array = [
        LLMChain(
            llm=llm, 
            prompt=prompt,
            # verbose=True,
            # output_parser=ExtractArticleSummaryParser(),
            # verbose=True,
        ) for llm in llms
    ]

    print("Loading tokenizer")

    hook = PostgresHook(postgres_conn_id='synthetic_data')
    # results = hook.get_records(sql=f"SELECT abstract, section_names FROM articles LIMIT {number_of_llms*5};")

    # results = hook.get_records(sql=f"SELECT abstract, section_names FROM articles where article_id not in  LIMIT {number_of_llms*5};")

    if overwrite_variant:
        print("Deleting old summaries")
        hook.run(sql=f"DELETE FROM short_article_summary WHERE summary_variant = {summary_variant};")    

    query = """
select sas.article_id, ap.part_id, ap.part_text, sas.article_summary 
from article_part_register apr
join article_parts ap on ap.part_id  = apr.part_id  
join short_article_summary sas on sas.article_id = apr.article_id 
;
    """

#     results = hook.get_records(sql=f"""
# SELECT article_id, abstract, section_names 
# FROM articles 
# where article_id not in (
# 	select sas.article_id 
# 	FROM short_article_summary sas 
# 	where sas.summary_variant = {summary_variant}
# ) and article_id < {amount_to_process}
# order by article_id 
# LIMIT {amount_to_process};
# """)


    results = hook.get_records(sql=query)

    print("Splitting results")
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer", 
                                                   cache_dir='/opt/airflow/data/hf/models',)
    list_of_tasks = []
    common_register = set()
    for chain in chain_array:
        print("Creating task")
        list_of_tasks.append(run_chains(results, chain, common_register, tokenizer, summary_variant))

    await asyncio.gather(*list_of_tasks)


def generate_topics_from_parts_and_summaries_async(num_of_llms: int, max_tokens: int, amount_to_process: int, summary_variant: int, overwrite_variant: bool = False):
   loop = asyncio.get_event_loop()
   result = loop.run_until_complete(generate_topics_from_parts_and_summaries_base(num_of_llms, max_tokens, amount_to_process, summary_variant, overwrite_variant))
   return result