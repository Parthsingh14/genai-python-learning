#Basics of Langgraph
from pydantic import BaseModel
from langgraph.graph import StateGraph, START , END
from IPython.display import Image, display

#Defining The state(The Shared Memory of the Graph)
class GreetMessage(BaseModel):
    message: str = ""

#Creating the Graph - Here we are telling that this graph will use GreetMessage as its shared state
graph = StateGraph(GreetMessage)

#Nodes- Python Functions
def Greet(state: GreetMessage):
    state.message = f"{state.message}............. World!"
    return state

def upperCase(state: GreetMessage):
    state.message = state.message.upper()
    return state

graph.add_node("greet" , Greet)
graph.add_node("upperCase", upperCase)

graph.add_edge(START, "greet")
graph.add_edge("greet", "upperCase")
graph.add_edge("upperCase", END)

finalGraph = graph.compile()
res = finalGraph.invoke({"message": "Hello"})
print(res)
png = finalGraph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png)

print("Graph saved as graph.png")
