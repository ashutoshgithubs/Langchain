# ===========================================
# 🧠 ADVANCED RAG AGENT WITH MULTIPLE SOURCES
# Ollama Llama3.2:1b (Local, Free)
# Sources: Wikipedia, Arxiv, LangSmith Docs
# ===========================================

from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import initialize_agent, AgentType

import os
os.environ["USER_AGENT"] = "AdvancedRAGAgent/1.0"

# -------------------------------------------
# 1️⃣ Wikipedia Tool (General Knowledge)
# -------------------------------------------
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

# -------------------------------------------
# 2️⃣ Arxiv Tool (Research Papers)
# -------------------------------------------
arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api)

# -------------------------------------------
# 3️⃣ LangSmith Retriever Tool (Website Content)
# -------------------------------------------
# Load content from website
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Use Ollama embeddings
embeddings = OllamaEmbeddings(model="llama3.2:1b")

# Store vectors in Chroma DB
db = Chroma.from_documents(documents, embeddings)

# Convert vector store to retriever
retriever = db.as_retriever()

# Wrap retriever as a tool for the agent
retriever_tool = create_retriever_tool(
    retriever,
    name="langsmith_search",
    description="Search for information about LangSmith. For any LangSmith-related questions, use this tool!"
)

# -------------------------------------------
# 4️⃣ Combine All Tools
# -------------------------------------------
tools = [wiki_tool, arxiv_tool, retriever_tool]

# -------------------------------------------
# 5️⃣ Initialize Ollama LLM
# -------------------------------------------
llm = OllamaLLM(model="llama3.2:1b", temperature=0)

# -------------------------------------------
# 6️⃣ Create the Agent
# -------------------------------------------
agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------------------------------------------
# 7️⃣ Console Chat Loop
# -------------------------------------------
# Corrected Console Chat Loop
print("\n🧠 Advanced RAG Agent (Ollama + Multi-Source)")
print("Type 'exit' to quit.\n")

while True:
    query = input("💬 Enter your question: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting... 👋")
        break
    try:
        # Use invoke with input dictionary
        response = agent.invoke({"input": query})
        print("\n✅ Answer:\n", response["output"])
    except Exception as e:
        print("❌ Error:", e)
