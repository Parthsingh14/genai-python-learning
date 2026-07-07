from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

# LLM + Tool
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

search = TavilySearch()
memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=[search],
    checkpointer=memory,
    system_prompt="You are a helpful AI assistant with web search capabilities."
)

st.set_page_config(
    page_title="AskBuddy Search Agent",
    page_icon="🤖"
)

st.title("🤖 AskBuddy Search Agent")
st.markdown("AI chatbot with Gemini + Tavily Search")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask Anything...")

if query:

    # Show user message
    st.session_state.messages.append(
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
                    {"messages": st.session_state.messages},
                    {"configurable": {"thread_id": "1"}}
                )

                ai_response = response["messages"][-1].content[0]["text"]

                st.markdown(ai_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": ai_response
            }
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")