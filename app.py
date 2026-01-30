import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports (LangChain 1.x compatible)
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables
load_dotenv()

# 2. Streamlit config
st.set_page_config(page_title="Website Chatbot", page_icon="🤖")
st.title("🤖 Chat with any Website")

# -----------------------------
# Helper functions
# -----------------------------

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
    )
    splits = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def get_answer_from_rag(vectorstore, user_query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    system_prompt = """
    You are an assistant for question-answering tasks.
    Use ONLY the following context to answer the question.
    If the answer is not in the context, say exactly:
    "The answer is not available on the provided website."

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": user_query
    })

    return response.content


# -----------------------------
# UI
# -----------------------------

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter Website URL")
    process_button = st.button("Process URL")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if process_button and website_url:
    with st.spinner("Processing website..."):
        try:
            st.session_state.vectorstore = get_vectorstore_from_url(website_url)
            st.success("Website processed successfully!")
        except Exception as e:
            st.error(str(e))

user_query = st.chat_input("Ask a question about the website...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.warning("Please process a website URL first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer_from_rag(
                    st.session_state.vectorstore,
                    user_query
                )
                st.write(answer)
