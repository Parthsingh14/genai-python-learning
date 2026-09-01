from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from typing import Literal

from state import BlogState
from agents import get_llm, researcher_agent, writer_agent, editor_agent

MAX_REVISION = 3

###Defining the nodes with there particular HITL(HUman in the Loop) nodes

def researcher_node(state: BlogState):
    """Researcher Agent generates or revises the research outline"""
    llm = get_llm()

    research_data = researcher_agent(
        llm=llm,
        topic=state.topic,
        audience=state.audience,
        feedback= state.research_feedback
    )

    state.research = research_data
    state.research_feedback = ""
    return state

def human_review_research_node(state: BlogState):
    """Pause and ask the human to approve the research or send the feedback"""
    decision = interrupt({
        "stage":"researcher_review",
        "research": state.research,
        "instructions": (
            "Reply with 'Approve' to continue to writing.",
            "or describe what to change to send it back to the researcher."
        )
    })

    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "ok","yes","process",""] else "revise"
        feedback = "" if action == "approve" else text

    state.research_feedback = feedback
    return state



def writer_node(state: BlogState):
    """Writer Agent produces the full draft of the blog or revises it"""
    llm = get_llm()
    
    draft_data = writer_agent(
            llm=llm,
            topic=state.topic,
            audience=state.audience,
            feedback= state.draft_feedback,
            research= state.research
        )
    
    state.draft = draft_data
    state.draft_feedback = ""
    return state

def human_review_writer_node(state: BlogState):
    """Pause and ask the human to approve the draft or send a feedback"""

    decision = interrupt({
        "stage": "draft_review",
        "draft": state.draft,
        "instructions": (
            "Reply with 'Approve' to continue to editor.",
            "or describe what to change to send it back to the writer."
        )
    })
    if isinstance(decision, dict):
            action = decision.get("action", "approve")
            feedback = decision.get("feedback", "")
    else:
            text = str(decision)
            action = "approve" if text.lower() in ["approve", "ok","yes","process",""] else "revise"
            feedback = "" if action == "approve" else text

    if feedback:
         state.review_count += 1

    state.draft_feedback = feedback
    return state



def editor_node(state: BlogState):
    """Writer Agent produces the full draft of the blog or revises it"""
    llm = get_llm()
        
    final = editor_agent(
                llm=llm,
                topic=state.topic,
                draft= state.draft
            )
        
    state.final_draft = final
    return state






### Conditonal Edges
def route_after_research_review(state: BlogState) -> Literal["research", "writer"]:
     if state.research_feedback:
          return "research"
     else:
          return "writer"

def route_after_draft_review(state: BlogState) -> Literal["writer", "editor"]:
     if state.draft_feedback and state.review_count < MAX_REVISION:
          return "writer"
     else:
          return "editor"


###Build and Compile the graph

def build_blog_graph():
     builder = StateGraph(BlogState)

     ### add nodes
     builder.add_node("research", researcher_node)
     builder.add_node("review_research", human_review_research_node)
     builder.add_node("writer", writer_node)
     builder.add_node("review_draft", human_review_writer_node)
     builder.add_node("editor", editor_node)

     ###add edges
     builder.add_edge(START,"research")
     builder.add_edge("research","review_research")
     builder.add_conditional_edges("review_research" , route_after_research_review,{"research": "research", "writer": "writer"})
     builder.add_edge("writer","review_draft")
     builder.add_conditional_edges("review_draft", route_after_draft_review, {"writer":"writer", "editor": "editor"})
     builder.add_edge("editor", END)

     GRAPH = builder.compile(checkpointer=InMemorySaver())
     return GRAPH