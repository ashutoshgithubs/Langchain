import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

st.set_page_config(page_title="US Census RAG", page_icon="📊", layout="centered")

st.title("📊 US Census Data RAG Search")
st.markdown("Ask questions from the US Census documents (2022) using AI-powered search.")

# access_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
# login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# Load the data
@st.cache_resource(show_spinner=True)
def load_data():
    loader = PyPDFDirectoryLoader("./us_census")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(documents)
    return final_docs

# Embeddings and Vector Store
@st.cache_resource(show_spinner=True)
def create_vectorstore(final_docs):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma.from_documents(final_docs, embeddings)
    return vectorstore

# Making Retrieval chain
@st.cache_resource(show_spinner=True)
def create_retriever_chain(_vectorstore):
    retriever = _vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = """
    Use the following context to answer the question.
    Answer **only** based on the context provided.

    Context:
    {context}

    Question: {question}

    Helpful Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = HuggingFacePipeline.from_model_id(
        model_id="mistralai/Mistral-7B-v0.1",
        task="text-generation",
        pipeline_kwargs={"temperature": 0.1, "max_new_tokens": 300}
    )

    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return retrieval_chain



with st.spinner("📂 Loading and preparing data..."):
    docs = load_data()
    vectorstore = create_vectorstore(docs)
    retrievalQA = create_retriever_chain(vectorstore)

st.success("✅ Data Loaded Successfully!")

# UI Part
query = st.text_input("🔍 Enter your question:", placeholder="e.g. What is health insurance coverage?")
if st.button("Ask"):
    if query.strip():
        with st.spinner("🤖 Generating Answer..."):
            result = retrievalQA.invoke({"query": query})
            st.subheader("🧠 Answer:")
            st.write(result["result"])

            # Optional: Show source documents
            # with st.expander("📚 View Source Context"):
            #     for i, doc in enumerate(result["source_documents"], 1):
            #         st.markdown(f"**Source {i}:**")
            #         st.write(doc.page_content)
    else:
        st.warning("⚠️ Please enter a question before submitting.")

