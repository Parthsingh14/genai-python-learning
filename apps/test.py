from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

st.write("Testing")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
search = TavilySearch()

agent = create_agent(
    model=model,
    tools=[search],
    system_prompt="""
Always search the web before answering.
"""
)
print(agent)
print(agent.get_graph())
print(search)
print(search.name)
print(search.description)
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is today's date?"
        }
    ]
})

st.write(response)