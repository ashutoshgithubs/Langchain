import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.cassandra import Cassandra
import cassio
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']
astra_db_application_token=os.environ['ASTRA_DB_APPLICATION_TOKEN']
astra_db_id=os.environ['ASTRA_DB_ID']


