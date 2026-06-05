# This script demonstrates a simple Google Gemini chat interaction using LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from dotenv import load_dotenv

load_dotenv()


# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash"
)

messages = [
    ("system", "You are a senior expert AI Engineer"),
    ("human", "What is the future of your domain"),
]

res = model.invoke(messages)
print(res.text)