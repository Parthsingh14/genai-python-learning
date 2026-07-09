from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq


# LLM + Tool
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
llm = ChatGroq(model="qwen/qwen3-32b")

search = TavilySearch()
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=model,
    tools=[search],
    checkpointer=st.session_state.memory,
    system_prompt="""
You are a helpful AI assistant.

For any question involving:
- latest news
- current events
- sports
- weather
- prices
- recent information
- today's information
- this year

ALWAYS use the Tavily search tool before answering.
Never rely only on your internal knowledge when freshness matters.
"""
)

st.set_page_config(
    page_title="AskBuddy Search Agent",
    page_icon="🤖"
)

st.title("🤖 AskBuddy Search Agent")
st.markdown("AI chatbot with Gemini + Tavily Search")

print(st.session_state.memory)
# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask Anything...")

if query:

    # Show user message
    st.session_state.history.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                response = agent.invoke(
                    {"messages": [{"role":"user","content": query}]},
                    {"configurable": {"thread_id": "1"}}
                )

                ai_response = response["messages"][-1].content[0]["text"]
                print(response)
                print(ai_response)
                st.markdown(ai_response)

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": ai_response
            }
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")