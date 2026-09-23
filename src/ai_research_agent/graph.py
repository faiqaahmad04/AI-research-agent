import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

from ai_research_agent.state import ResearchState
from ai_research_agent.models import (
    ResearchPlan,
    ResearchEvaluation,
    SourceEvaluation,
)
from ai_research_agent.prompts import (
    PLANNER_PROMPT,
    ANALYST_PROMPT,
    CRITIC_PROMPT,
    SOURCE_EVALUATION_PROMPT,
    WRITER_PROMPT,
)
from ai_research_agent.tools import search_tool


load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_SEARCHES = 3
MAX_RETRIES = 3
RETRY_DELAY = 3

PDF_DIRECTORY = Path("reports")
PDF_DIRECTORY.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Initialize Gemini
# ---------------------------------------------------------

model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    temperature=0,
)


# Structured output models

planner_model = model.with_structured_output(ResearchPlan)

critic_model = model.with_structured_output(ResearchEvaluation)

source_evaluation_model = model.with_structured_output(
    SourceEvaluation
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def content_to_text(content) -> str:
    """Convert Gemini response content into plain text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)

            if text:
                text_parts.append(str(text))

        return "\n".join(text_parts)

    return str(content)


def is_retryable_error(error: Exception) -> bool:
    """
    Determine whether an API error is likely temporary.
    """

    error_text = str(error).upper()

    retryable_messages = [
        "503",
        "UNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "TOO MANY REQUESTS",
        "INTERNAL SERVER ERROR",
        "TEMPORARILY UNAVAILABLE",
    ]

    return any(
        message in error_text
        for message in retryable_messages
    )


def invoke_with_retry(
    runnable,
    input_data,
    retries: int = MAX_RETRIES,
):
    """
    Invoke an LLM or tool with basic retry handling.
    """

    last_error = None

    for attempt in range(retries):
        try:
            return runnable.invoke(input_data)

        except Exception as error:
            last_error = error

            if not is_retryable_error(error):
                raise

            if attempt == retries - 1:
                raise

            wait_time = RETRY_DELAY * (attempt + 1)

            print(
                f"Temporary API error. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise last_error


def remove_duplicate_results(results: list[dict]) -> list[dict]:
    """
    Remove duplicate search results using their URL.
    """

    unique_results = []
    seen_urls = set()

    for result in results:
        url = result.get("url", "").strip()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)
        unique_results.append(result)

    return unique_results


# ---------------------------------------------------------
# Research planning
# ---------------------------------------------------------

def plan_research(state: ResearchState):
    """Create focused search queries for the research question."""

    prompt = PLANNER_PROMPT.format(
        question=state["question"]
    )

    result = invoke_with_retry(planner_model, prompt)

    queries = result.queries

    if not queries:
        raise ValueError(
            "The research planner did not generate any queries."
        )

    # Limit the number of queries.
    queries = queries[:MAX_SEARCHES]

    return {
        "research_plan": queries,
        "current_query": queries[0],
        "search_count": 0,
    }


# ---------------------------------------------------------
# Web search
# ---------------------------------------------------------

def search_web(state: ResearchState):
    """Search the web using the current research query."""

    query = state["current_query"]

    result = invoke_with_retry(
        search_tool,
        {
            "query": query
        },
    )

    results = result.get("results", [])

    existing_results = state.get(
        "search_results",
        []
    )

    formatted_results = []

    for item in results:
        formatted_results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
        )

    # Combine old and new results.
    combined_results = (
        existing_results + formatted_results
    )

    # Remove duplicate URLs.
    unique_results = remove_duplicate_results(
        combined_results
    )

    return {
        "search_results": unique_results,
        "search_count": state.get(
            "search_count",
            0
        ) + 1,
    }


# ---------------------------------------------------------
# Source evaluation
# ---------------------------------------------------------

def evaluate_sources(state: ResearchState):
    evaluated_sources = []

    for index, source in enumerate(state.get("search_results", []), start=1):
        try:
            prompt = SOURCE_EVALUATION_PROMPT.format(
                question=state["question"],
                title=source.get("title", ""),
                url=source.get("url", ""),
                content=source.get("content", ""),
            )

            evaluation = source_evaluation_model.invoke(prompt)

            evaluated_sources.append({
                "id": index,
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "content": source.get("content", ""),
                "source_type": evaluation.source_type,
                "credibility": evaluation.credibility,
                "relevant": evaluation.relevant,
                "potential_bias": evaluation.potential_bias,
                "reason": evaluation.reason,
            })

        except Exception as e:
            evaluated_sources.append({
                "id": index,
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "content": source.get("content", ""),
                "source_type": "Other",
                "credibility": "Unknown",
                "relevant": True,
                "potential_bias": "Unknown",
                "reason": f"Source evaluation failed: {str(e)}",
            })

    return {
        "evaluated_sources": evaluated_sources
    }

# ---------------------------------------------------------
# Research analysis
# ---------------------------------------------------------

def analyze_research(state: ResearchState):
    sources = state.get("evaluated_sources", [])

    research_data = []

    for source in sources:
        research_data.append({
            "source_id": source.get("id"),
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "source_type": source.get("source_type", ""),
            "credibility": source.get("credibility", ""),
            "relevant": source.get("relevant", True),
            "potential_bias": source.get("potential_bias", ""),
            "content": source.get("content", ""),
        })

    prompt = ANALYST_PROMPT.format(
        question=state["question"],
        search_results=json.dumps(research_data, indent=2),
    )

    result = model.invoke(prompt)

    summary = content_to_text(result.content)

    return {
        "research_summary": summary
    }


# ---------------------------------------------------------
# Research critique
# ---------------------------------------------------------

def critique_research(state: ResearchState):
    """Check whether the research is sufficient."""

    prompt = CRITIC_PROMPT.format(
        question=state["question"],
        research_summary=state[
            "research_summary"
        ],
    )

    result = invoke_with_retry(
        critic_model,
        prompt,
    )

    return {
        "critique": result.reasoning,
        "research_sufficient": result.sufficient,
    }


# ---------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------

def choose_next_step(state: ResearchState):
    """
    Decide whether to search again or write the report.
    """

    # If research is sufficient, write the report.
    if state.get("research_sufficient"):
        return "write_report"

    # Don't exceed maximum number of searches.
    if state.get("search_count", 0) >= MAX_SEARCHES:
        return "write_report"

    plan = state.get(
        "research_plan",
        []
    )

    current_query = state.get(
        "current_query",
        ""
    )

    remaining_queries = [
        query
        for query in plan
        if query != current_query
    ]

    if not remaining_queries:
        return "write_report"

    return "search_again"


# ---------------------------------------------------------
# Prepare next search
# ---------------------------------------------------------

def prepare_next_search(state: ResearchState):
    """Select another research query."""

    plan = state.get(
        "research_plan",
        []
    )

    current_query = state.get(
        "current_query",
        ""
    )

    remaining = [
        query
        for query in plan
        if query != current_query
    ]

    if remaining:
        return {
            "current_query": remaining[0]
        }

    return {}


# ---------------------------------------------------------
# PDF generation
# ---------------------------------------------------------

def create_pdf(report: str):
    PDF_DIRECTORY.mkdir(exist_ok=True)

    timestamp = int(time.time())
    pdf_path = PDF_DIRECTORY / f"research_report_{timestamp}.pdf"

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading1"]
    body_style = styles["BodyText"]

    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    elements = []

    lines = report.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            elements.append(Spacer(1, 0.12 * inch))
            continue

        if line.startswith("# "):
            text = line[2:].strip()
            elements.append(Paragraph(text, heading_style))

        elif line.startswith("## "):
            text = line[3:].strip()
            elements.append(Paragraph(text, heading_style))

        elif line.startswith("* "):
            text = line[2:].strip()
            elements.append(Paragraph(f"• {text}", body_style))

        else:
            elements.append(
                Paragraph(line, body_style)
            )

    document.build(elements)

    return str(pdf_path)

# ---------------------------------------------------------
# Write final report
# ---------------------------------------------------------

def write_report(state: ResearchState):
    sources = []

    for source in state.get("evaluated_sources", []):
        url = source.get("url")

        if not url:
            continue

        sources.append({
            "id": source.get("id"),
            "title": source.get("title", ""),
            "url": url,
            "source_type": source.get("source_type", ""),
            "credibility": source.get("credibility", ""),
            "relevant": source.get("relevant", True),
            "potential_bias": source.get("potential_bias", ""),
        })

    prompt = WRITER_PROMPT.format(
        question=state["question"],
        research_summary=state.get("research_summary", ""),
        sources=json.dumps(sources, indent=2),
    )

    result = model.invoke(prompt)
    report = content_to_text(result.content)
    pdf_path = create_pdf(report)

    return {
        "report": report,
        "sources": sources,
        "pdf_path": pdf_path,
    }

# ---------------------------------------------------------
# Build LangGraph workflow
# ---------------------------------------------------------

def build_graph():
    """Build the LangGraph research workflow."""

    graph = StateGraph(
        ResearchState
    )

    # Nodes
    graph.add_node(
        "plan_research",
        plan_research,
    )

    graph.add_node(
        "search_web",
        search_web,
    )

    graph.add_node(
        "evaluate_sources",
        evaluate_sources,
    )

    graph.add_node(
        "analyze_research",
        analyze_research,
    )

    graph.add_node(
        "critique_research",
        critique_research,
    )

    graph.add_node(
        "prepare_next_search",
        prepare_next_search,
    )

    graph.add_node(
        "write_report",
        write_report,
    )

    # START → Planner
    graph.add_edge(
        START,
        "plan_research",
    )

    # Planner → Search
    graph.add_edge(
        "plan_research",
        "search_web",
    )

    # Search → Source Evaluation
    graph.add_edge(
        "search_web",
        "evaluate_sources",
    )

    # Source Evaluation → Analysis
    graph.add_edge(
        "evaluate_sources",
        "analyze_research",
    )

    # Analysis → Critic
    graph.add_edge(
        "analyze_research",
        "critique_research",
    )

    # Critic → Conditional routing
    graph.add_conditional_edges(
        "critique_research",
        choose_next_step,
        {
            "search_again": "prepare_next_search",
            "write_report": "write_report",
        },
    )

    # Next query → Search
    graph.add_edge(
        "prepare_next_search",
        "search_web",
    )

    # Writer → END
    graph.add_edge(
        "write_report",
        END,
    )

    return graph.compile()


# Compile graph
research_graph = build_graph()