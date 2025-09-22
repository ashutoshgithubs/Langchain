import requests
import streamlit as st

def get_gemma_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']['content']

def get_llama_response(input_text):
    response=requests.post(
    "http://localhost:8000/poem/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']

    ## streamlit framework

st.title('Chatbot with Llama and Gemma')
input_text=st.text_input("Hi, I'm Gemma. What do you want to know about?")
input_text1=st.text_input("Hi, I'm Llama. How can I assist you?")

if input_text:
    st.write(get_gemma_response(input_text))

if input_text1:
    st.write(get_llama_response(input_text1))