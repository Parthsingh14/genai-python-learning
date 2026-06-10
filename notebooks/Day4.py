# Practising the Structured Output by using PYDANTIC

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# class ResponseStructure(BaseModel):
#     name: str = Field(description="Complete Name")
#     email: str = Field(description="Email Address")
#     age: int = Field(description="Age")

# structured_llm = llm.with_structured_output(ResponseStructure)

# res = structured_llm.invoke("Please give me only name, email and age from the text: My name is parth singh, i am a ibm employee and my mail is Parth.Singh@ibm.com and i am currently 23 years old")
# dict_format = res.model_dump()
# print(dict_format["name"])
# print(dict_format["email"])
# print(dict_format["age"])




class Movies(BaseModel):
    movie_name: str = Field(description="Name of Movie")
    year: int = Field(description="Release year")


class Multiple_Movies(BaseModel):
    movies: List[Movies]


structured_model = llm.with_structured_output(Multiple_Movies)
res = structured_model.invoke("Give me list of top 5 animes")
print(res.movies[0])

