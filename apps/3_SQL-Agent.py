from dotenv import  load_dotenv
load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq

db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:postgres123@localhost:5432/my_tasks")
print("DB created Successfully")