#Streaming becomes very simple with langchain because it gives a very simple and centralized template

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", streaming=True)
grok_model = ChatGroq(model = "qwen/qwen3-32b", streaming = True)
question = "What is GENAI"

res = grok_model.stream(question)
for chunk in res:
    print(chunk.content, end="")
