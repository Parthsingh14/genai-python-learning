from dotenv import load_dotenv
load_dotenv()

from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)


FILE_PATH = "../data/resume.pdf"

loader = DoclingLoader(file_path=FILE_PATH)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
splitted_data = text_splitter.split_documents(doc)

for document in splitted_data:
    document.metadata = {}

vector_store = Chroma.from_documents(
    documents=splitted_data,
    embedding=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

def get_context(query:str):
    data = vector_store.similarity_search(query=query)

    context = ""
    for doc in data:
        context = context + doc.page_content + "\n"
    
    return {
        "context":context,
        "question":query
    }


prompt = PromptTemplate.from_template("""
You are a helpful assistant and provide answers based on the context for user question.
If you dont know the answer, then you can say 'I dont know'.
Context : {context}
Question:{question}
""")

rag_chain = get_context | prompt | llm

res = rag_chain.invoke("Tell me about the work experience")
print(res.content)
