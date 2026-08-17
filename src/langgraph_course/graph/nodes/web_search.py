from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from ..state import GraphState
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Performs a web search using the TavilySearch tool and updates the graph state with the retrieved documents.

    Args:
        state (GraphState): The current graph state containing the question.

    Returns:
        Dict[str, Any]: Updated graph state with retrieved documents and the original question.
    """
    print("--- WEB SEARCH ---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})['results']
    joined_tavily_results = "\n".join([result["content"] for result in tavily_results])

    web_results_document = Document(
        page_content=joined_tavily_results,
    )

    if documents is not None:
        documents.append(web_results_document)
    else:
        documents = [web_results_document]
    return {"documents": documents, "question": question}
