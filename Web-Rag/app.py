import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores.cassandra import Cassandra
import cassio
from langchain_community.vectorstores import Cassandra
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']
astra_db_application_token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
astra_db_id=os.environ['ASTRA_DB_ID']
web_url = os.environ["WEB_URL"]

# Building the db connection
cassio.init(token=astra_db_application_token, database_id=astra_db_id)
# print("Cassandra connection established")

loader=WebBaseLoader(web_paths=(web_url,),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")

                     )))

text_documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=text_splitter.split_documents(text_documents)

embeddings = OllamaEmbeddings(model="llama3.2:1b")
astra_vector_store=Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None

)

astra_vector_store.add_documents(docs)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)


prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $10 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

retriever = astra_vector_store.as_retriever()
document_chain = create_stuff_documents_chain(llm,prompt)
retrieval_chain = create_retrieval_chain(retriever,document_chain)

response=retrieval_chain.invoke({"input":"Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique"})
response['answer']