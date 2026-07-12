#data loading from a pdf

from langchain_docling.loader import DoclingLoader

FILE_PATH = "../data/resume.pdf"

loader = DoclingLoader(file_path=FILE_PATH)

doc = loader.load()

for i in doc:
    print(i.page_content)