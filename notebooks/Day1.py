from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from dotenv import load_dotenv

load_dotenv("../.env")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash"
)

res = model.invoke("Who is the PM of india")
print(res.content[0]["text"])