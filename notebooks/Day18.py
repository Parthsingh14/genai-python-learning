#Human in the Loop

from dotenv import load_dotenv
load_dotenv()

from langgraph import graph
from langgraph.graph import StateGraph , START, END
from langgraph.types import interrupt

from langchain_groq import ChatGroq

from pydantic import BaseModel

llm = ChatGroq(model="op")


class MailState(BaseModel):
    query: str = ""
    draft: str = ""
    human_feedback: str = ""
    final_response: str = ""


### Defining the nodes

def DraftEmail(state: MailState) -> MailState:
    draft = llm.invoke(state.query).content
    state.draft = draft
    return state


def HumanFeedback(state: MailState) -> MailState:
    feedback = interrupt({
        "draft_email": state.draft,
        "question": "Do you want to continue or re-write the mail ?"
    })

    fb = (feedback or "").strip().lower()
    if fb in ("approved", "ok", "continue"):
        state.human_feedback = ""
        return state
    else:
        state.human_feedback = feedback
        return state



def finalize_node(state: MailState) -> MailState:
    if state.human_feedback:
        state.final_response = state.human_feedback
        return state
    else:
        print("Email sent successfully!")
        state.final_response = "Email sent successfully!"
        return state


graphBuilder = StateGraph(MailState)

#nodes
graphBuilder.add_node("DraftEmail", DraftEmail)
graphBuilder.add_node("HumanFeedback", HumanFeedback)
graphBuilder.add_node("Finalize", finalize_node)

#edges
graphBuilder.add_edge(START, "DraftEmail")
graphBuilder.add_edge("DraftEmail", "HumanFeedback")
graphBuilder.add_edge("HumanFeedback", "Finalize")
graphBuilder.add_edge("Finalize", END)

graph = graphBuilder.compile()