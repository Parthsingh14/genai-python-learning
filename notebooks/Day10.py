#This is a simple text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

document = "Hey my name is Parth Singh, and I am currently working at IBM Consulting at Bangalore office, the address is BCIT Thanisandra Main Road, Karnataka. Here They have assigned me SAP MM domain, which I dont want at all, but due to not having any offer of other company I have to do this."

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = text_splitter.split_text(document)
print(chunks)