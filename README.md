# AI Research Agent

An AI research agent that researches a question using web search, Gemini, and LangGraph. It plans searches, evaluates sources, analyzes research, and generates a cited research report as a PDF.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).

2. Create a `.env` file from `.env.example` and add:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

3. Install dependencies:

```powershell
uv sync
```

## Run

```powershell
uv run uvicorn ai_research_agent.main:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

## Endpoints

- `GET /` - Check that the API is running.
- `GET /health` - Check API health.
- `POST /research` - Submit a research question and receive a research report, sources, and PDF file path.
- `GET /download-pdf` - Download a generated PDF report.

## Project Folders

- `src/ai_research_agent/` - Application code, research workflow, prompts, tools, and API.
- `reports/` - Generated PDF research reports.
- `Output_SwaggerUI/` - Screenshot of the API in Swagger UI.