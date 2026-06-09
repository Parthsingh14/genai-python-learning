# This Streamlit app loads environment variables, initializes the Google Gemini LLM,
# and sets up a simple chat-based Q&A interface with session state.

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st



llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")

st.title("ASKBUDDY - AI QnA Bot")
st.markdown("My QnA Bot with langchain and google gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything ?")
if query:
    st.session_state.messages.append({"role":"user", "content": query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.session_state.messages.append({"role":"ai", "content": res.text})
    st.chat_message("ai").markdown(res.text)

