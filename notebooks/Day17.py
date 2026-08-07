#Multi AI Agent

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph , START, END

from pydantic import BaseModel, Field
from typing import Literal


class FlowState(BaseModel):
    question: str = Field(description="The question to be answered by the AI agents.")
    category: Literal['coding','google_search','weather'] = Field(description="The category of the question, which determines which AI agent to use.", default="google_search")
    answer: str = Field(description="The answer provided by the AI agent.", default="")

class QuestionCategory(BaseModel):
    category: Literal['coding','google_search','weather'] = Field(description="The category of the question, which determines which AI agent to use.", default="google_search")


llm = ChatGroq(model="llama-3.3-70b-versatile")

#Node 1
def check_question_category(state:FlowState) -> FlowState:
    st_llm = llm.with_structured_output(QuestionCategory)
    res = st_llm.invoke(f"I want to know the category of the question : {state.question}. And if you are not sure, just give google_search as the category.")
    print(res)
    state.category = res.category
    return state

#Node 2
def route(state: FlowState) -> Literal['coding','google_search','weather']:
    return state.category

#Node 3
def coding_node(state: FlowState) -> FlowState:
    res = llm.invoke(f"Write code to answer the following question: {state.question}")
    state.answer = res.content
    return state

#Node 4
def weather_node(state: FlowState) -> FlowState:
    print("Weather node invoked")
    return state

#Node 5
def google_search_node(state: FlowState) -> FlowState:
    print("Google search node invoked")
    return state

graph = StateGraph(FlowState)

graph.add_node("check_question_category", check_question_category)
graph.add_node("coding", coding_node)
graph.add_node("weather", weather_node)
graph.add_node("google_search", google_search_node)


graph.add_edge(START, "check_question_category")
graph.add_conditional_edges("check_question_category", route)
graph.add_edge("coding", END)
graph.add_edge("weather", END)
graph.add_edge("google_search", END)

graph = graph.compile()

response = graph.invoke({"question": "What is the weather in INDIA?"})