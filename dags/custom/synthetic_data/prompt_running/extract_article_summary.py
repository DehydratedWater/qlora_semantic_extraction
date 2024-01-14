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


class ExtractArticleSummaryParser(BaseLLMOutputParser):
    def parse_result(self, result: list[Generation]) -> str:
        final_results = []

        for r in result:
            extracted_json = None
            
            fail = False
            try:
                raw_text = r.text
                print(raw_text)
                start = raw_text.find("'''segment_summary")
                end = len(raw_text) - raw_text[::-1].find("'''")
                raw_text = raw_text[start:end]
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


async def run_chains(sr: list, chain: LLMChain, register: set[int], tokenizer: LlamaTokenizerFast):

    for i, r in enumerate(sr):
        if i in register:
            continue
        else:
            register.add(i)

            input_data = {"abstract": str(r[0]), "list_of_sections": str(r[1])}
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

                new_abstract = text_splitter.split_text(str(r[0]))[0]
                list_of_sections = text_splitter.split_text(str(r[1]))[0]

                input_data = {"abstract": new_abstract, "list_of_sections": list_of_sections}
                # print(f"Przycinanie 2: {input_data}")
                constructed_prompt = chain.prep_prompts([input_data])[0][0].text
                # print("Created prompt 2")
                # print(constructed_prompt)
                prompt_length = len(tokenizer.encode(text=constructed_prompt))
                # print(f"Promp lenght 2: {prompt_length}")

            result = await chain.arun(input_data)
            # result = await chain.llm.abatch([input_data])
            print("PROMPT ****************")
            print(constructed_prompt)
            print("RESULT ****************")
            print(result)

            print(f"Finished batch {i} / {len(sr)}")



async def generate_article_summaries_base(num_of_llms: int, max_tokens: int):

    model = "models/llama-2-13b-chat.Q4_K_M.gguf"

    number_of_llms = num_of_llms

    llms = [
        ChatOpenAI(temperature=0.7,
                    model=model, 
                    openai_api_base=f"http://llm-server-{i}:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=max_tokens,
                    request_timeout=5000,
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
        template=article_summary_prompt, 
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
    results = hook.get_records(sql=f"SELECT abstract, section_names FROM articles LIMIT {number_of_llms*5};")

    print("Splitting results")
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer", 
                                                   cache_dir='/opt/airflow/data/hf/models',)
    list_of_tasks = []
    common_register = set()
    for chain in chain_array:
        print("Creating task")
        list_of_tasks.append(run_chains(results, chain, common_register, tokenizer))

    await asyncio.gather(*list_of_tasks)


def generate_article_summaries_async(num_of_llms: int, max_tokens: int):
   loop = asyncio.get_event_loop()
   result = loop.run_until_complete(generate_article_summaries_base(num_of_llms, max_tokens))
   return result