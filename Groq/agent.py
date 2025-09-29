import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA


load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
web_url = os.environ["website_url"]

st.set_page_config(page_title="GroqFusion", page_icon="🧠")
st.title("⚡ GroqFusion - The Multi-Source AI Agent")

# ------------------ Load LLM ------------------ #
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# ------------------ Load Documents ------------------ #
if "vector" not in st.session_state:
    st.write("🔄 Loading Docs...")

    loader = WebBaseLoader(web_url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs[:50])

    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    vector_store = Chroma.from_documents(final_docs, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    st.session_state["retriever"] = retriever

st.success("✅ Docs Loaded")

# ------------------ External Sources ------------------ #
wiki = WikipediaAPIWrapper()
arxiv = ArxivAPIWrapper()

# ------------------ Tools ------------------ #
tools = [
    Tool(
        name="LangChainDocs",
        func=RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state["retriever"],
            chain_type="stuff"
        ).run,
        description="Use this to answer questions from LangChain documentation"
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Use this to answer questions using Wikipedia"
    ),
    Tool(
        name="Arxiv",
        func=arxiv.run,
        description="Use this to answer research or academic questions from Arxiv"
    ),
]

# ------------------ Initialize Agent ------------------ #
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

# ------------------ Streamlit UI ------------------ #
prompt = st.text_input("💬 Ask your question here:")

if prompt:
    st.info("🧠 Thinking... Selecting best source...")
    start = time.process_time()
    response = agent.invoke({"input": prompt})
    end = time.process_time()

    st.write("### ✅ Response")
    st.write(response["output"])
    st.write(f"⏱️ Response Time: {round(end - start, 2)} seconds")
