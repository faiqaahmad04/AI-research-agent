from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ai_research_agent.graph import research_graph


app = FastAPI(
    title="AI Research Agent",
    description="Agentic AI system for autonomous web research.",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    question: str = Field(
        min_length=10,
        max_length=1000,
    )


class ResearchResponse(BaseModel):
    question: str
    report: str
    sources: list[dict]
    pdf_file: str


@app.get("/")
def root():
    return {
        "message": "AI Research Agent is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/research",
    response_model=ResearchResponse,
)
def research(request: ResearchRequest):

    try:

        result = research_graph.invoke(
            {
                "question": request.question
            }
        )

        pdf_path = result.get(
            "pdf_path",
            "",
        )
        pdf_filename = Path(pdf_path).name if pdf_path else ""
        pdf_url = (
            f"/download-pdf?filename={pdf_filename}"
            if pdf_filename
            else ""
        )

        return ResearchResponse(
            question=request.question,
            report=result["report"],
            sources=result.get(
                "sources",
                [],
            ),
            pdf_file=pdf_url,
        )

    except Exception as e:

        error_text = str(e)

        if (
            "RESOURCE_EXHAUSTED"
            in error_text.upper()
            or "429"
            in error_text
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini free-tier quota is exhausted. "
                    "Wait for the quota reset or use another API key."
                ),
            )

        if (
            "503"
            in error_text
            or "UNAVAILABLE"
            in error_text.upper()
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "The AI model is temporarily unavailable. "
                    "Please try again later."
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=error_text,
        )


@app.get("/download-pdf")
def download_pdf(filename: str):

    pdf_path = Path("reports") / filename

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF report not found.",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
    )