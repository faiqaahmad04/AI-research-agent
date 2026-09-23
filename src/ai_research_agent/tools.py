from dotenv import load_dotenv
from langchain_tavily import TavilySearch


load_dotenv()


search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="basic",
)