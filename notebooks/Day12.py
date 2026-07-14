from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

texts = [
    "My name is Parth Singh.",
    "I work at IBM.",
    "My favorite programming language is Python.",
    "I live in Bangalore."
]

vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",
)

queries = [
    "Where do I work?",
    "What is my name?",
    "Which programming language do I like?",
    "Where do I live?"
]

for q in queries:
    result = vector_store.similarity_search(q, k=1)
    print(f"\nQuery: {q}")
    print(result[0].page_content)