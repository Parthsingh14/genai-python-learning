import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


# ============================================================
# 1. STATE
# ============================================================

class MailState(BaseModel):
    query: str = ""
    draft: str = ""
    human_feedback: str = ""
    final_response: str = ""


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# ============================================================
# 3. NODES
# ============================================================

def draft_email(state: MailState) -> MailState:

    if state.human_feedback:
        prompt = f"""
You previously generated this email:

{state.draft}

The human gave this feedback:

{state.human_feedback}

Rewrite the email according to the feedback.
Return only the revised email.
"""
    else:
        prompt = f"""
Write a professional email based on this request:

{state.query}

Return only the email body.
"""

    response = llm.invoke(prompt)
    state.draft = response.content

    return state


def human_feedback(state: MailState) -> MailState:

    feedback = interrupt(
        {
            "draft_email": state.draft,
            "question": "Approve this email or provide feedback for a rewrite."
        }
    )

    feedback = (feedback or "").strip()

    if feedback.lower() in {
        "approved",
        "approve",
        "ok",
        "continue",
        "yes"
    }:
        # Empty feedback means: approved
        state.human_feedback = ""
    else:
        # Anything else is treated as rewrite feedback
        state.human_feedback = feedback

    return state


def finalize_node(state: MailState) -> MailState:

    state.final_response = state.draft

    return state


# ============================================================
# 4. ROUTER
# ============================================================

def conditional_routing(state: MailState):

    if state.human_feedback:
        # Human requested changes
        return "DraftEmail"

    # Human approved
    return "Finalize"


# ============================================================
# 5. BUILD GRAPH
# ============================================================

def build_graph():

    builder = StateGraph(MailState)

    builder.add_node("DraftEmail", draft_email)
    builder.add_node("HumanFeedback", human_feedback)
    builder.add_node("Finalize", finalize_node)

    builder.add_edge(START, "DraftEmail")
    builder.add_edge("DraftEmail", "HumanFeedback")

    builder.add_conditional_edges(
        "HumanFeedback",
        conditional_routing,
        {
            "DraftEmail": "DraftEmail",
            "Finalize": "Finalize",
        },
    )

    builder.add_edge("Finalize", END)

    checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)


# IMPORTANT:
# Streamlit reruns this Python file every time a button is clicked.
# If we create InMemorySaver() normally, the memory would be recreated
# and the interrupted graph would be lost.
#
# cache_resource keeps the graph/checkpointer alive across Streamlit reruns.
@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()


# ============================================================
# 6. STREAMLIT SETUP
# ============================================================

st.set_page_config(
    page_title="Human-in-the-Loop Email",
    page_icon="✉️"
)

st.title("✉️ Human-in-the-Loop Email Writer")
st.caption("AI drafts → you review → approve or request a rewrite.")


# A unique thread for this browser session.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "email-session-1"

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ============================================================
# 7. SESSION STATE
# ============================================================

if "started" not in st.session_state:
    st.session_state.started = False

if "waiting_for_feedback" not in st.session_state:
    st.session_state.waiting_for_feedback = False

if "finished" not in st.session_state:
    st.session_state.finished = False


# ============================================================
# 8. HELPER: READ CURRENT LANGGRAPH STATE
# ============================================================

def get_current_state():

    snapshot = graph.get_state(config)

    # snapshot.values is the latest checkpointed state.
    return snapshot.values


# ============================================================
# 9. INPUT
# ============================================================

query = st.text_area(
    "What email do you want to write?",
    value=(
        "Write a mail to my manager asking for a leave of absence "
        "for 2 weeks starting from next Monday."
    ),
    height=120,
)


# ============================================================
# 10. FIRST RUN — GENERATE DRAFT
# ============================================================

if not st.session_state.started:

    if st.button("Generate Draft", type="primary"):

        if not query.strip():
            st.warning("Please enter an email request.")
            st.stop()

        # Start the LangGraph execution.
        result = graph.invoke(
            {"query": query},
            config=config,
        )

        # The graph reaches interrupt() and pauses.
        # At this point the draft has already been generated
        # and checkpointed.
        if result.get("__interrupt__"):

            st.session_state.started = True
            st.session_state.waiting_for_feedback = True

            st.rerun()


# ============================================================
# 11. AFTER INTERRUPT — SHOW DRAFT
# ============================================================

else:

    state = get_current_state()

    draft = state.get("draft", "")
    final_response = state.get("final_response", "")

    # --------------------------------------------------------
    # Always show the latest draft while working.
    # --------------------------------------------------------

    if draft:

        st.subheader("Current Draft")

        st.text_area(
            "Email",
            value=draft,
            height=300,
            disabled=True,
            key="current_draft",
        )


    # ========================================================
    # 12. HUMAN DECISION
    # ========================================================

    if st.session_state.waiting_for_feedback:

        st.subheader("Review the draft")

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # APPROVE
        # ----------------------------------------------------

        with col1:

            if st.button(
                "Approve & Finish",
                type="primary"
            ):

                # Resume the paused interrupt.
                result = graph.invoke(
                    Command(resume="approved"),
                    config=config,
                )

                st.session_state.waiting_for_feedback = False
                st.session_state.finished = True

                st.rerun()


        # ----------------------------------------------------
        # REWRITE
        # ----------------------------------------------------

        with col2:

            feedback = st.text_input(
                "Rewrite feedback",
                placeholder="e.g. Make it more formal and concise.",
                key="rewrite_feedback",
            )

            if st.button("Rewrite"):

                if not feedback.strip():
                    st.warning("Please enter feedback.")
                    st.stop()

                # Resume the paused interrupt with human feedback.
                #
                # The router will then send the graph:
                #
                # HumanFeedback
                #       ↓
                #   DraftEmail
                #       ↓
                # HumanFeedback
                #
                result = graph.invoke(
                    Command(resume=feedback),
                    config=config,
                )

                # The graph has now generated a NEW draft and
                # paused again at interrupt().
                st.session_state.waiting_for_feedback = True

                st.rerun()


    # ========================================================
    # 13. FINAL RESULT
    # ========================================================

    if st.session_state.finished:

        state = get_current_state()

        final_response = state.get("final_response", "")

        st.success("Email approved successfully.")

        st.subheader("Final Email")

        st.text_area(
            "Final response",
            value=final_response,
            height=300,
            disabled=True,
            key="final_email",
        )


        # ----------------------------------------------------
        # NEW EMAIL
        # ----------------------------------------------------

        if st.button("Start New Email"):

            # Give the new execution a different thread.
            import uuid

            st.session_state.thread_id = (
                f"email-session-{uuid.uuid4()}"
            )

            st.session_state.started = False
            st.session_state.waiting_for_feedback = False
            st.session_state.finished = False

            st.rerun()