import os
import streamlit as st
from typing import Literal
from dataclasses import dataclass

import qdrant_client
from langchain.embeddings.cohere import CohereEmbeddings

from langchain.vectorstores import Qdrant

from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory

os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# Streamlit header
st.set_page_config(page_title="Co:Chat - An LLM-powered chat bot")
st.title("Sheraton-Bot")
st.write("This is a chatbot for Sheraton Hotel")
st.write("(Knowleddge base is limited to Sheraton Hotel only and currently using trial api key of cohere.)")

# Defining message class
@dataclass
class Message :
    """Class for keepiong track of chat Message."""
    origin : Literal["Customer","elsa"]
    Message : "str"


# laodinf styles.css
def load_css():
    with open("static/styles.css", "r")  as f:
        css = f"<style>{f.read()} </style>"
        # st.write(css)
        st.markdown(css, unsafe_allow_html = True)

# We are creating the embeddings in dbCheck file and we donot need to create it again
# we only need to load it

# loading Qdrant cloud
def load_db():

    client = qdrant_client.QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )
    embeddings = CohereEmbeddings(model="embed-english-v2.0")
    vector_store = Qdrant(
        client = client,
        collection_name = "hotelDataCollection",
        embeddings = embeddings
    )
    print("connection established !")
    return vector_store


def initialize_session_state() :
    vector_store = load_db()
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
        give the answer to the question using the context. If the context is not provided then answer the question based on the knowledge base.
        The answer should be short, straight and to the point. If you don't know the answer, reply that the answer is not available.
        Never Hallucinate
        
        Context: {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = { "prompt" : PROMPT }
        llm = Cohere(model = "command", temperature=0.5)

        #build your chain for RAG+C
        
        print("vector store loaded !")
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
    customer_prompt = st.session_state.customer_prompt

    if customer_prompt:
        
        st.session_state.input_value = ""
        st.session_state.initial_message_sent = True

        with st.spinner('Generating response...'):

            llm_response = st.session_state.chain(
                {"context": st.session_state.chain.memory.buffer, "question": customer_prompt}, return_only_outputs=True)

    st.session_state.history.append(
        Message("customer", customer_prompt)
    )
    st.session_state.history.append(
        Message("AI", llm_response)
    )



def main():

    initialize_session_state()
    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")

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


if __name__ == "__main__":
    main()