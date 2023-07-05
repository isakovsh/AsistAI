import openai
import streamlit as st
from streamlit_chat import message
import os, dotenv
from google.cloud import translate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = 'Api key'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "translate.json"
translation_client = translate.TranslationServiceClient()

def translate_text(text, target_language=None):
    project_id = "focal-road-362517"
    location = "global"

    parent = f"projects/{project_id}/locations/{location}"
    contents = [text]
    target_language_code = target_language

    response = translation_client.translate_text(
        parent=parent,
        contents=contents,
        target_language_code=target_language_code,
    )

    translation = response.translations[0].translated_text
    return translation


chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0, model_name="gpt-3.5-turbo", max_tokens=500
    ),
    chain_type="stuff",
    retriever=FAISS.from_documents(PyPDFLoader("data.pdf").load_and_split(), OpenAIEmbeddings())
        .as_retriever(search_type="similarity", search_kwargs={"k":1})
)

template = """
respond as succinctly as possible. {query}?
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

# Setting page title and header
st.set_page_config(page_title="Asia", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>AsistentAI - help for you ! </h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "Your name is AsistAI, you are the assistant bot developed by Shohrux. You are here to help with inquiries."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# Generate a response
def generate_response(prompt):
    # Translate user input from Uzbek to English
    translated_input = translate_text(prompt, target_language='en')

    # Generate model response in English
    response = chatbot.run(translated_input)

    # Translate model response from English to Uzbek
    translated_response = translate_text(response, target_language='uz')

    # Append the translated response to the chat history
    st.session_state['messages'].append({"role": "assistant", "content": translated_response})

    return translated_response

# Container for chat history
response_container = st.container()
# Container for text box
container = st.container()

temp = "I want you to act as an AI assistant bot developed by BIKAL, whose name is ASIA."
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        if len(st.session_state['messages']) == 0:
            output = generate_response(temp + user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        else:
            output = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
