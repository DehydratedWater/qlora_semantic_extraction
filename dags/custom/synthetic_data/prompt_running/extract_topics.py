import asyncio
import json
from custom.synthetic_data.prompt_running.topic_extraction_prompt import extracton_prompt
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.base import BaseLLMOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from transformers import LlamaTokenizerFast
from airflow.providers.postgres.hooks.postgres import PostgresHook

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


async def run_chains(sr: list, chain: LLMChain, register: set[int]):

    for i, r in enumerate(sr):
        if i in register:
            continue
        else:
            register.add(i)
        constructed_prompt = chain.prep_prompts([{"text_to_extract": r}])[0][0].text
        result = await chain.arun(r[1])
        print("PROMPT ****************")
        print(constructed_prompt)
        print("RESULT ****************")
        print(result)

        print(f"Finished batch {i} / {len(sr)}")



async def generate_general_categories_base(num_of_llms: int):

    model = "models/llama-2-13b-chat.Q4_K_M.gguf"

    number_of_llms = num_of_llms

    llms = [
        ChatOpenAI(temperature=0.7,
                    model=model, 
                    openai_api_base=f"http://llm-server-{i}:5556/v1", 
                    openai_api_key="sx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    max_tokens=4096,
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
            # verbose=True,
        ) for llm in llms
    ]

    print("Loading tokenizer")

    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer", 
                                                   cache_dir='/opt/airflow/data/hf/models',)
    hook = PostgresHook(postgres_conn_id='synthetic_data')
    results = hook.get_records(sql=f"SELECT * FROM article_parts LIMIT {number_of_llms*5};")

    print("Splitting results")

    list_of_tasks = []
    common_register = set()
    for chain in chain_array:
        print("Creating task")
        list_of_tasks.append(run_chains(results, chain, common_register))

    await asyncio.gather(*list_of_tasks)


def generate_general_categories_async(num_of_llms: int):
   loop = asyncio.get_event_loop()
   result = loop.run_until_complete(generate_general_categories_base(num_of_llms))
   return result