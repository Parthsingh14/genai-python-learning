from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# -------------------------------
# Streamlit Session State
# -------------------------------

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Models
# -------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

memory = InMemorySaver()


# -------------------------------
# Process Documents
# -------------------------------

def process_documents(uploaded_files):

    os.makedirs("./doc_files", exist_ok=True)

    all_docs = []

    # Load every uploaded PDF
    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            "./doc_files",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = DoclingLoader(file_path=file_path)

        docs = loader.load()

        all_docs.extend(docs)

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
    )

    split_docs = splitter.split_documents(all_docs)

    # Remove metadata (optional)
    for doc in split_docs:
        doc.metadata = {}

    # Vector Store
    vector_store = InMemoryVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
    )

    # -----------------------
    # Retriever Tool
    # -----------------------

    @tool
    def retriever_tool(query: str):
        """
        Retrieve relevant information from uploaded PDF documents.
        """

        print(f"\nTool Called : {query}\n")

        docs = vector_store.similarity_search(
            query=query,
            k=3,
        )

        context = ""

        for doc in docs:
            context += doc.page_content
            context += "\n\n"

        return context

    system_prompt = """
        You are an intelligent RAG assistant.

        Rules:

        1. ALWAYS use retriever_tool before answering.
        2. If the user asks multiple questions, decompose them.
        3. Call retriever_tool multiple times whenever necessary.
        4. Combine all retrieved context.
        5. If information is unavailable, clearly say so.
        6. Never hallucinate.
        """

    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=system_prompt,
        checkpointer=memory,
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# -------------------------------
# Upload UI
# -------------------------------

st.title("Agentic RAG using Docling")

if not st.session_state.document_uploaded:

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        with st.spinner("Processing PDFs..."):

            process_documents(uploaded_files)

        st.rerun()


# -------------------------------
# Chat UI
# -------------------------------

if st.session_state.document_uploaded:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input(
        "Ask anything about the uploaded PDFs..."
    )

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = st.session_state.agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query,
                            }
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": "pdf-chat"
                        }
                    },
                )

                answer = response["messages"][-1].content[0]["text"]

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )