#Agentic RAG Implementation

from dotenv import load_dotenv
load_dotenv()

from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import create_agent

#creating the objects
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Loading doc and splliting it
FILE_PATH = "../data/resume.pdf"

loader = DoclingLoader(file_path=FILE_PATH)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
splitted_data = text_splitter.split_documents(doc)

#removing the metadata becuase of some error
for document in splitted_data:
    document.metadata = {}

# generating the embeddings and storing it also.
vector_store = InMemoryVectorStore.from_documents(
    documents=splitted_data,
    embedding=embeddings
)

# Agent , tool , prompt
@tool
def retriever_tool(query:str):
    """
        This tool can help you to retrieve the relevant data of the PDF Documents, and these pdf documents have details about the resume.
    """
    print("Tool Called :" , query)
    docs = vector_store.similarity_search(query=query,k=2)

    context = ""
    for doc in docs:
        context = context + doc.page_content + "\n\n"
    
    return context

System_prompt = """
You are an intelligent retrieval agent.

Before answering:

1. Analyze the user's query.
2. If it contains multiple independent questions, break it into smaller queries.
3. Call the retriever_tool separately for each query if needed.
4. Combine all retrieved contexts.
5. Produce one final answer.

Never answer without retrieving information.
"""

agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt=System_prompt
)

query = """Who is Parth?
What companies has he worked for?
What are his technical skills?
Which AI projects has he built?"""

response = agent.invoke({"messages": [{"role":"user" , "content":query}]})
result = response["messages"][-1].content
print(result)