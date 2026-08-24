import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


### GET LLM
def get_llm():
    """
    Get the LLM instance.
    """
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return llm




###Agents

RESERSCHER_PROMPT = ChatPromptTemplate.from_messages([
    {"role": "system" , "content": """
    You are a Research Agent. Given a blog topic and target audience, produce a clear, structured research outline. 
    Include:
        1. 5-7 key points the blog should cover
        2. Important facts, stats, or examples for each point.
        3. Suggest angle or hook.
    Be concise. USe Bullet points. DO NOT write the full blog yet.
    """},
    {"role": "user", "content": "Topic: {topic} , Audience: {audience}, {revision_hints}, Write the research outline now. " }
    ])


def researcher_agent(llm: ChatGroq, topic:str, audience:str, feedback:str =""):
    revision_hints = f"The human provided this feedback on your previous reasearch - please address it: {feedback}."
    if not feedback:
        revision_hints = "This is your first attempt."


    chain = RESERSCHER_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "revision_hints": revision_hints
    })

    return result.content