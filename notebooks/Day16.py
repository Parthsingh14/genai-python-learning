#Simple chatbot example using LangGraph and LangChain Groq
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Annotated

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph , START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

class ChatState(BaseModel):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="openai/gpt-oss-20b")

def chatBotNode(state: ChatState) -> ChatState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

memory_saver = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chatBotNode", chatBotNode)

graph.add_edge(START, "chatBotNode")
graph.add_edge("chatBotNode", END)

graph = graph.compile(checkpointer = memory_saver)

config = {"configurable":{"thread_id":"test_thread"}}
response = graph.invoke(
    {
        "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
        ]
    }, config)

print(response)