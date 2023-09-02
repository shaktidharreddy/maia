# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes. 
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.  
# import nest_asyncio
# nest_asyncio.apply()

# PydanticMultipleSelector
# Use the OpenAI Function API to generate/parse pydantic objects under the hood for the router selector.
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.selectors.pydantic_selectors import PydanticMultiSelector, PydanticSingleSelector
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.query_engine import ToolRetrieverRouterQueryEngine
from langchain.chat_models import ChatOpenAI

import openai
import os
import json
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    ListIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    SimpleKeywordTableIndex,
    LLMPredictor,
)

def llama_listvectorkeyword_index(uploaded_file, query):
    # load documents
    documents = SimpleDirectoryReader(input_files = [uploaded_file]).load_data()
    
    # initialize service context (set chunk size)
    #service_context = ServiceContext.from_defaults(chunk_size=1024)
    llm_predictor_gpt4 = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4-32k"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt4, chunk_size=1024)
    nodes = service_context.node_parser.get_nodes_from_documents(documents)

    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    #Define List Index and Vector Index over Same Data
    list_index = ListIndex(nodes, storage_context=storage_context)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

    #Define Query Engines and Set Metadata
    list_query_engine = list_index.as_query_engine(
        response_mode='tree_summarize',
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    list_tool = QueryEngineTool.from_defaults(
        query_engine=list_query_engine,
        description='Useful for summarization questions',
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description='Useful for retrieving specific context from a particular section or paragraph.',

    )

    #Define Keyword Query Engine
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)
    keyword_query_engine = keyword_index.as_query_engine()
    keyword_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description='Useful for retrieving specific context using keywords',
    )

    query_engine = RouterQueryEngine(
        selector=PydanticMultiSelector.from_defaults(),
        query_engine_tools=[
            list_tool,
            vector_tool,
            keyword_tool,
        ]
    )

    #Define Retrieval-Augmented Router Query Engine
    tool_mapping = SimpleToolNodeMapping.from_objects([list_tool, vector_tool, keyword_tool])
    obj_index = ObjectIndex.from_objects(
        [list_tool, vector_tool, keyword_tool], 
        tool_mapping,
        VectorStoreIndex,
    )

    query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())

    return query_engine.query(query)

def prompt(file):
    with open(file) as f:
        return f.read()
    
