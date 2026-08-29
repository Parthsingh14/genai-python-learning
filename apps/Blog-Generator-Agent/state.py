from pydantic import BaseModel

class BlogState(BaseModel):
    ### User input
    topic: str = ""
    audience: str = "general readers"

    ### Researcher Output
    research: str = ""
    research_feedback: str = ""

    ### Writer Output
    draft: str = ""
    draft_feedback: str = ""

    ### Editor Output
    final_draft: str = ""

    ### Metadata
    review_count: int = 0

