# Practiced using LangChain with Google Gemini, prompt templates, chaining, and output parsing

from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash"
)

out = StrOutputParser()

def transform_case(content: str):
    return content.upper()

prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "You are a {domain}"
    },
    {
        "role": "user",
        "content": "{query}"
    }
])

#Simple way
# final_prompt = prompt.format_messages(domain = "Senior JS Dev", query = "What is the future of your domain")
# res = model.invoke(final_prompt) 


#Chaining - little better
# res = model.invoke(prompt.invoke({"domain": "Senior JS Dev", "query": "WHo are you?"}))
# print(res.text)

#Correct way of chaining - prompt goes to model as input, then model goes as input to out, then out result goes to function as input.
chains = prompt | model | out | transform_case
res = chains.invoke({"domain": "Senior JS Dev", "query": "WHo are you?"})
print(res)
