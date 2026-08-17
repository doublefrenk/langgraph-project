from typing import List, TypedDict

class GraphState(TypedDict):
  """
  Represnts the State of our graph.

  Attributes:
    question: question
    generation: LLM generation
    web_search: whether web search
    documents: list of documents
  """

  question: str
  generation: str
  web_search: bool
  documents: List[str]