import streamlit as st
import requests
import time

st.title("Polling Example")

API_URL = "https://jsonplaceholder.typicode.com/posts/1"

placeholder = st.empty()

while True:
    data = requests.get(API_URL).json()
    placeholder.write(data)
    time.sleep(3)
