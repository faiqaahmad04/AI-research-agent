from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    queries: list[str] = Field(
        description="A list of focused web search queries needed to research the topic."
    )


class ResearchEvaluation(BaseModel):
    sufficient: bool = Field(
        description="Whether the collected research is sufficient to answer the question."
    )
    reasoning: str = Field(
        description="Explain why the research is or is not sufficient."
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Important information that is still missing."
    )


class SourceEvaluation(BaseModel):
    source_type: str = Field(
        description=(
            "Classify the source as one of: Academic, Government, "
            "Industry Organization, Company/Vendor, News/Media, "
            "Technical Publication, or Other."
        )
    )
    credibility: str = Field(
        description=(
            "Assess credibility as High, Moderate, or Low based on "
            "the source type, reputation, and available evidence."
        )
    )
    relevant: bool = Field(
        description="Whether the source directly helps answer the research question."
    )
    potential_bias: str = Field(
        description=(
            "Briefly describe any obvious commercial, organizational, "
            "or other potential bias. Use 'None obvious' when appropriate."
        )
    )
    reason: str = Field(
        description="Brief explanation supporting the evaluation."
    )