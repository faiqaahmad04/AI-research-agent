from typing import TypedDict


class ResearchState(TypedDict, total=False):
    question: str
    research_plan: list[str]
    current_query: str

    search_results: list[dict]
    evaluated_sources: list[dict]

    sources: list[dict]

    research_summary: str
    critique: str
    research_sufficient: bool

    report: str
    pdf_path: str

    search_count: int