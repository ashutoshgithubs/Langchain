from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn # a lightweight ASGI server to run FastAPI apps
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv


load_dotenv()

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)


gemma_model=Ollama(model="gemma3:1b")
llama_model=Ollama(model="llama3.2:1b")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} in 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child in 100 words")

add_routes(
    app,
    prompt1|gemma_model,
    path="/essay"
)

add_routes(
    app,
    prompt2|llama_model,
    path="/poem"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000) 
