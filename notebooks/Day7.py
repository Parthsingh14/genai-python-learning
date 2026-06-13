# SImple tool calling agent
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@tool
def add_numbers(a:int, b:int)->int:
    """
    it will return the sum of number

    Args:
        a: Number One
        b: Number Two
    """
    return a+b

@tool
def multiply_numbers(a:int, b:int)->int:
    """
    it will return the product of number

    Args:
        a: Number One
        b: Number Two
    """
    return a*b

agent = create_agent(
    model=model,
    tools=[add_numbers,multiply_numbers],
    system_prompt="You are a math teacher and always use tools for calculation."
)

response = agent.invoke(
    {
        "messages":[
            {
                "role":"user" , "content": "what is 2+5?"
            }
        ]
    }
)

# for res in response["messages"]:
#     print(res)
#     print("\n")

print(response["messages"][-1].content)