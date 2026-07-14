#Here we gonna load the document and split it also
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_PATH = "../data/resume.pdf"

loader = DoclingLoader(file_path=FILE_PATH)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=70)

doc = loader.load()

doc_chunks = text_splitter.split_documents(doc)
for i, chunk in enumerate(doc_chunks):
    print("=" * 80)
    print(f"Chunk {i+1}")
    print("-" * 80)
    print(chunk.page_content)