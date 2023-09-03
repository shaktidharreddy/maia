#current working version for streaming
# NOTE: This is ONLY necessary in jupyter notebook.
# Details: Jupyter runs an event-loop behind the scenes. 
#          This results in nested event-loops when we start an event-loop to make async queries.
#          This is normally not allowed, we use nest_asyncio to allow it for convenience.  
# import nest_asyncio
# nest_asyncio.apply()

# PydanticMultipleSelector
# Use the OpenAI Function API to generate/parse pydantic objects under the hood for the router selector.

from langchain.chat_models import ChatOpenAI
from llama_index.llms import OpenAI
import streamlit as st
import time
import openai
import os
import json
os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key #st.secrets["openai_api_key"]
openai.api_key = st.session_state.openai_api_key #st.secrets["openai_api_key"]

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

def llama_vector_index(uploaded_file, query):
    # load documents
    documents = SimpleDirectoryReader(input_files = [uploaded_file]).load_data()
    
    service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-4-32k", temperature=0, streaming=True, ), chunk_size=4096, chunk_overlap = 512, 
    )
    
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    index_query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
    streaming_response = index_query_engine.query(query)


    placeholder = st.empty()
    full_text = ""

    for text in streaming_response.response_gen:
        full_text += text
        time.sleep(0.1)
        placeholder.write(full_text)
    

    return full_text
    
    

def prompt(file):
    with open(file,encoding="utf8") as f:
        return f.read()
    
