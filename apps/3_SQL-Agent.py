from dotenv import  load_dotenv
load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from pprint import pprint
import streamlit as st

st.set_page_config(
    page_title="TaskBot",
    page_icon="✅",
    layout="centered"
)

db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:postgres123@localhost:5432/my_tasks")
db.run("""
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

model = ChatGroq(model="openai/gpt-oss-20b")
toolkit = SQLDatabaseToolkit(db=db,llm=model)
tools = toolkit.get_tools()



system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

IMPORTANT RULES:
1. Every request about tasks MUST be answered only by querying the SQL database.
2. Never fabricate, assume, or invent any task, row, or database record.
3. If a SQL query returns zero rows or an empty result, reply that no matching records were found.
4. If the SQL tool returns an empty result, do not create example data.
5. Use only the information returned by SQL tools.

TASK RULES:
1. Limit SELECT queries to 10 results using ORDER BY created_at DESC.
2. After every CREATE, UPDATE, or DELETE, verify the operation with a SELECT query.
3. Present task lists in a structured markdown table.

CREATE RULES:
- Title is required.
- Description is optional.
- If the user does not provide a description, store NULL.
- Do not ask for a description unless the user explicitly requests to add one.

CRUD OPERATIONS:
CREATE: INSERT INTO tasks(title, description, status)
READ: SELECT * FROM tasks ...
UPDATE: UPDATE tasks SET ...
DELETE: DELETE FROM tasks ...

Table schema:
id, title, description, status (pending, in_progress, completed), created_at
"""

@st.cache_resource
def get_agent():
    agent = create_agent(
    model = model,
    tools=tools,
    checkpointer=InMemorySaver(),
    system_prompt=system_prompt
    )
    return agent

agent = get_agent()

st.title("✅ TaskBot")
st.caption("Manage your PostgreSQL tasks using natural language.")

with st.sidebar:
    st.header("💡 Example Commands")

    st.markdown("""
- Show all tasks
- Create a task to learn LangGraph
- Create a task to revise SQL
- Mark task 2 as completed
- Delete task 3
- Show pending tasks
""")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

prompt = st.chat_input("Ask me to manage your task ?")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = agent.invoke(
            {"messages": [{"role":"user", "content": prompt}]},
            {"configurable": {"thread_id":"1"}}
            )

            result = response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role":"assistant","content":result})
    

st.divider()
st.caption("Built with LangChain • LangGraph • PostgreSQL • Groq • Streamlit")