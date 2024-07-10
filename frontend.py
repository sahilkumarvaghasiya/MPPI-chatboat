import streamlit as st
import requests



def huggingfacemodel_output(input_text):
    response=requests.post(
    "http://localhost:8000/home/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']

st.title('Hello my friend')

input_text=st.text_input("Ask me question")

if input_text:
    result = huggingfacemodel_output(input_text)
    st.write(result)

