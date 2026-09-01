from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command

from graph import build_blog_graph

graph = build_blog_graph()

config = { 
    "configurable": {
        "thread_id": "1"
        }
    }

topic = input("\nEnter Your Topic: ").strip()
audience = input("\nEnter Your Audience: ").strip()

if not audience:
    audience = "general readers"


state = graph.invoke({
    "topic": topic,
    "audience": audience
},
config=config) ### is moment pe mera graph pehla intruppt pe pause hoga.


# Handle Human-in-the-loop interruptions
while True:
    snap = graph.get_state(config)

    if not snap.interrupts:
        break

    interrupt_payload = snap.interrupts[0].value

    stage = interrupt_payload.get("stage")

    if stage == "researcher_review":
        print("\n"+ "=" * 60)
        print("RESEARCH OUTLINE FOR YOUR REVIEW")
        print("=" * 60)

        print(interrupt_payload.get("research"))

    elif stage == "draft_review":
        print("\n"+ "=" * 60)
        print("BLOG DRAFT FOR YOUR REVIEW")
        print("=" * 60)
        
        print(interrupt_payload.get("draft"))

    else:
        print("\nUnknown review stage:", stage)

    decision = input("\nYour Decision (approve/revise): ").strip().lower()

    if decision == "approve":
        resume_value = {
            "action": "approve",
            "feedback": ""
        }
    else:
        feedback = input("WHat would you like to change? ").strip()
        resume_value = {
            "action": "revise",
            "feedback": feedback
        }


    graph.invoke(
        Command(resume=resume_value),
        config=config
    )


#Graph has reached END

final_state = graph.get_state(config)
print("\n" + "=" * 60)
print("FINAL BLOG")
print("=" * 60)

print(final_state.values.get("final_draft", ""))