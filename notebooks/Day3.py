# Its a simple memory based bot
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash"
)
history = []
while True:
    query = input("User: ")
    if query.lower() in ["exit","quit","bye"]:
        break
    history.append({"role": "user", "content": query})

    res = model.invoke(history)
    history.append({"role": "ai", "content": res.text})
    print("Bot: "+res.text)