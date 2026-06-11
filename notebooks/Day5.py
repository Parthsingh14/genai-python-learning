#Grok is an AI Company known for its ultra fast-AI Interface
#this is much faster then other platforms because it has
#its custom designed (LPU) Language Processing Unit
#Optimized for running LLM with low latency and high throughput.

#WHat Grok not is - provide llm model, no its a hardware company that design chips
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(model="qwen/qwen3-32b")


res = llm.invoke("Hi who are you?")
print(res.text)