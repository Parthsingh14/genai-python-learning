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

###Researcher prompt and agent
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

def researcher_agent(llm: ChatGroq, topic:str, audience:str, feedback:str ="") -> str:
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


###Writer prompt and agent
WRITER_PROMPT = ChatPromptTemplate.from_messages([
    {"role": "system" , "content": """
    You are a Research Blog Writer. Using the reaearch notes provided, write a complete,engaging blog post.
    Rules:
        1. Length 200-300 words.
        2. Structure: catchy title,intro hook, 3-5 sections with H2 headings, conclusion.
        3. Tone: clear, friendly suited to the target audience.
        4. Use Markdown formating
        5. Do not add a "word count" line at the end.
    """},
    {"role": "user", "content": """
        Topic: {topic},
        Audience: {audience},
        Research Notes: {research},

        {revision_hints}

        Write the full blog post now.

    """ }
    ])

def writer_agent(llm: ChatGroq, topic:str, audience:str, feedback:str ="" , research:str ="") -> str:
    revision_hints = f"The human provided this feedback on your previous draft and asked for this changes: {feedback}. Please apply these changes during writting the blog."
    if not feedback:
        revision_hints = "This is your first attempt."


    chain = WRITER_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "revision_hints": revision_hints,
        "research": research
    })

    return result.content


### Editor/Final Blog writer agent

EDITOR_PROMPT = ChatPromptTemplate.from_messages([
    {"role": "system" , "content": """
    You are a Editor Agent - the final quality gate before publishing.
    Take the draft and produce the FINAL polished version. 
    Specifically:
        1. Fix grammer, spelling and awkward phrases.
        2. Tighten wordy sentences.
        3. Improve flow and transitions between sections.
        4. Make the title and intro more compelling if needed.
        5. Keep the same structure and markdown formatting.
        6. Output only the final polished blog post - no commentary.
    """},
    {"role": "user", "content": """
        Topic: {topic},
        Draft: {draft},

        Return the published blog post.

    """ }
    ])

def editor_agent(llm: ChatGroq, topic:str, draft:str) -> str:

    chain = EDITOR_PROMPT | llm

    result = chain.invoke({
        "topic": topic,
        "draft": draft
    })

    return result.content