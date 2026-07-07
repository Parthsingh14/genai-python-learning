#Tavily Search AI Agent
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch


search = TavilySearch()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

agent = create_agent(
    model=model,
    tools=[search],
    system_prompt="You are a agent and can search for any question on google"
)

question = "What is happnening in the war iran and america latest news, also mention todays date."

response = agent.invoke({"messages": [{
    "role": "user",
    "content": question
}]})

print(response["messages"][-1].content[0]["text"])