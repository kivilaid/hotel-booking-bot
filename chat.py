import os
import streamlit as st
from typing import Literal
from dataclasses import dataclass

from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain, ConversationChain
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationSummaryMemory

from langchain.vectorstores import Qdrant
from langchain.document_loaders import WebBaseLoader
from qdrant_client import models, QdrantClient



os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# "qRMUuF5mmpHClOpnpk5BXsxUvzcnMFmQTUc6Gwh5"
st.set_page_config(page_title="Co:Chat - An LLM-powered chat bot")
st.title("Sheraton-Bot")

#Creating a Class for message
@dataclass
class Message :
    """Class for keepiong track of chat Message."""
    origin : Literal["Customer","elsa"]
    Message : "str"


#Funcion to load css
def load_css():
    with open("static/styles.css", "r")  as f:
        css = f"<style>{f.read()} </style>"
        # st.write(css)
        st.markdown(css, unsafe_allow_html = True)


# save the embeddings in a DB that is persistent
# def manage_pdf() :
# loader = TextLoader("./facts.txt")

#Weblinks for Knowledge Base
# web_links = ["https://in.hotels.com/ho1068250336/four-points-by-sheraton-kochi-infopark-kakkanad-india",
#              "https://www.expedia.co.in/Kochi-Hotels-Four-Points-By-Sheraton-Kochi-Infopark.h33351573.Hotel-Information",
#              ] 


embeddings = CohereEmbeddings(model = "embed-english-v2.0")
print(" embedding docs !")

# creating a client to connect to qdrant
qdrant_client = QdrantClient(
    url = QDRANT_HOST,
    api_key= QDRANT_API_KEY,
)

# creating vector store
collection_name = "hotelDataCollection"
vector_store = Qdrant(
    client=qdrant_client,
    collection_name = collection_name,
    embeddings=embeddings
)
print("connection established !")

# vector_store = Qdrant.from_documents(texts, embeddings, location=":memory:",collection_name="summaries-po", distance_func="Dot")

#initializing Session State
def initialize_session_state() :

    # Initialize a session state to track whether the initial message has been sent
    if "initial_message_sent" not in st.session_state:
        st.session_state.initial_message_sent = False

    # Initialize a session state to store the input field value
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""


    if "history" not in st.session_state:
        st.session_state.history = []

    if "chain" not in st.session_state :  

        #create custom prompt for your use case
        prompt_template = """
        You are a Hotel Receptionist at "Four Points by Sheraton" hotel.

        You will be given a context of the conversation made so far followed by a customer's question, 
        give the answer to the question using the context. 
        The answer should be short, straight and to the point.
        If you don't know the answer, reply that the answer is not available.
        Never Hallucinate.
        
        Context: {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = { "prompt" : PROMPT }
        llm = Cohere(model = "command", temperature=0.5)

        #build your chain for RAG+C
        st.session_state.chain = ConversationalRetrievalChain.from_llm(     
            llm = llm,
            chain_type = "stuff",
            memory = ConversationSummaryMemory(llm = llm, memory_key='chat_history', input_key='question', output_key= 'answer', return_messages=True),
            retriever = vector_store.as_retriever(),
           return_source_documents=False,
           combine_docs_chain_kwargs=chain_type_kwargs,
           
        )

#Callblack function which when activated calls all the other
#functions 
def on_click_callback():

    load_css()
    # print(st.session_state.customer_prompt)
    customer_prompt = st.session_state.customer_prompt

    if customer_prompt:
        
        st.session_state.input_value = ""
        st.session_state.initial_message_sent = True

        with st.spinner('Generating response...'):

            llm_response = st.session_state.chain(
                {"question": customer_prompt,"summaries": st.session_state.chain.memory.buffer}, return_only_outputs=True)
            
            # print(st.session_state.chain.memory.buffer)
            # answer = llm_response["answer"]

    st.session_state.history.append(
        Message("customer", customer_prompt)
    )
    st.session_state.history.append(
        Message("AI", llm_response)
    )



initialize_session_state()
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
creditcard_placeholder = st.empty()

# query = st.chat_input()
# if query!= None :
#     st.session_state.customer_prompt = query
#     # st.write(st.session_state.customer_prompt)
#     result = st.session_state.chain(query)
#     st.write(print_result(result))
#     # st.write(result)
#     # display(Markdown(print_result(result)))

with chat_placeholder:
    for chat in st.session_state.history:
        if type(chat.Message) is dict:
            msg = chat.Message['answer']
        else:
            msg = chat.Message 
        div = f"""
        <div class = "chatRow 
        {'' if chat.origin == 'AI' else 'rowReverse'}">
            <img class="chatIcon" src = "app/static/{'elsa.png' if chat.origin == 'AI' else 'admin.png'}" width=32 height=32>
            <div class = "chatBubble {'adminBubble' if chat.origin == 'AI' else 'humanBubble'}">&#8203; {msg}</div>
        </div>"""
        st.markdown(div, unsafe_allow_html=True)


# Streamlit UI Input field
with st.form(key="chat_form"):
    cols = st.columns((6, 1))
    
    # Display the initial message if it hasn't been sent yet
    if not st.session_state.initial_message_sent:
        cols[0].text_input(
            "Chat",
            placeholder="Hello, how can I assist you?",
            label_visibility="collapsed",
            key="customer_prompt",
        )  
    else:
        cols[0].text_input(
            "Chat",
            value=st.session_state.input_value,
            label_visibility="collapsed",
            key="customer_prompt",
        )

    cols[1].form_submit_button(
        "Ask",
        type="secondary",
        on_click=on_click_callback,
    )

# Update the session state variable when the input field changes
st.session_state.input_value = cols[0].text_input