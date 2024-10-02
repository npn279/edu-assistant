import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dotenv
dotenv.load_dotenv()

import pandas as pd
import tiktoken
import asyncio
import re
import subprocess
import logging
logging.basicConfig(level=logging.INFO)

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch


async def search(query: str):
    api_key = os.getenv("GRAPHRAG_API_KEY")
    llm_model = "gpt-4o-mini"

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("o200k_base")

    # parquet files generated from indexing pipeline
    OUTPUT_DIR = "./graph_rag/output/results/artifacts"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"

    # community level in the Leiden community hierarchy from which we will load the community reports
    # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
    COMMUNITY_LEVEL = 2

    entity_df = pd.read_parquet(f"{OUTPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{OUTPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{OUTPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    logging.info(f"Total report count: {len(report_df)}")
    logging.info(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )
    report_df.head()

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )

    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set thisx to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    return await search_engine.asearch(query)

async def aget_response_stream(query: str):
    api_key = os.getenv("GRAPHRAG_API_KEY")
    llm_model = "gpt-4o-mini"

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("o200k_base")

    # parquet files generated from indexing pipeline
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"

    # community level in the Leiden community hierarchy from which we will load the community reports
    # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
    COMMUNITY_LEVEL = 2

    entity_df = pd.read_parquet(f"{OUTPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{OUTPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{OUTPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    logging.info(f"Total report count: {len(report_df)}")
    logging.info(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )
    report_df.head()

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )

    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set thisx to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    return search_engine.astream_search(query)

def extract_data_references(text):
    pattern = r'\[Data: Reports \((.*)\)\]'
    matches = re.findall(pattern, text)

    # Extract all numbers from the matches
    all_numbers = set()
    for match in matches:
        numbers = [int(num) for num in match.split(',')]
        all_numbers.update(numbers)

    # Convert to a sorted list
    return sorted(list(all_numbers))

def get_content_from_ids(df: pd.DataFrame, ids: list) -> dict:
    # Filter the DataFrame for the given ids
    # print(df)
    #print type of id col
    # print(df['id'].dtype)
    #change type of id col to int
    df['id'] = df['id'].astype(int)
    filtered_df = df[df['id'].isin(ids)]
    # print(filtered_df)
    # Return a dictionary with 'id' as keys and 'content' as values
    return dict(zip(filtered_df['id'], filtered_df['content']))

def get_result_graphrag(query):
    result = asyncio.run(search(query=query))
    ids = extract_data_references(result.response)
    logging.info(f"GET RESULT GRAPH RAG: {ids}")
    context = get_content_from_ids(result.context_data['reports'], ids)
    return result.response, context

def indexing_graph(command="python -m graphrag.index --root ./graph_rag"):
    try:
        # Run the command and capture the output
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)

        logging.info(f"INDEXING GRAPH: COMMAND OUTPUT = {result.stdout}")

        # Print the standard error, if any
        if result.stderr:
            logging.error(f"INDEXING GRAPH: ERROR OUTPUT = {result.stderr}")

        return result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return e.returncode


async def main():
    query = "So sánh ứng dụng của Linear regression và Logistic regression trong Machine Learning"
    stream = await aget_response_stream(query)

    report = await stream.__anext__()
    print('Report:', report)


    response = []
    async for result in stream:
        response.append(result)
        print(response[-1], end='', flush=True)

    




if __name__=='__main__':
    # print(indexing_graph())
    # response, context = get_result_graphrag("So sánh ứng dụng của Linear regression và Logistic regression trong Machine Learning")
    # print(response)

    # print(context)
    asyncio.run(main())