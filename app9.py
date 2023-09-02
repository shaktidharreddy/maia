import openai
import streamlit as st
from streamlit_chat import message
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import download_loader, StorageContext, load_index_from_storage

# Setting page title and header
st.set_page_config(page_title="Chatbot", page_icon=":sparkles:")
st.markdown("<h1 style='text-align: center;'>Shakti Chatbot 🌞</h1>", unsafe_allow_html=True)


def create_vector():
    documents = SimpleDirectoryReader(saved_path).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    storage_context = StorageContext.from_defaults()
    index.storage_context.persist("./vectordatabase")
    #print ("Done")

def generate_response(prompt):
    storage_context = StorageContext.from_defaults(persist_dir="./vectordatabase")
    index = load_index_from_storage(storage_context)
    query_engin = index.as_query_engine() 
    question = prompt
    response = query_engin.query(question)
    return str(response)
    #print ("\n", response)
    
import os
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
import glob

saved_path = "/home/cdsw/experimentation_project1/PLS_project/bot_data"

# Ensure the directory exists
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
  
# def remove_all_files(directory):
#    files = glob.glob(directory + '/*')
#    for f in files:
#        os.remove(f)


# Upload the file
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

if uploaded_file is not None:
    # To read file as bytes:
    for file in uploaded_file:
        bytes_data = file.getvalue()
    
        # Save the uploaded file to the 'data' directory
        with open(os.path.join(saved_path, file.name), 'wb') as out_file:
            out_file.write(bytes_data)

    st.success('PDF file saved in data directory')
    create_vector()
    #remove_all_files(saved_path)
    st.success('Vector created')



# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=50)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        #st.session_state['model_name'].append(model_name)
 


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user1')
            message(st.session_state["generated"][i], key=str(i))
