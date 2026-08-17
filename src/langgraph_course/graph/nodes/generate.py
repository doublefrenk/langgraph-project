from typing import Any, Dict

from ..chains.generation import generation_chain
from ..state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
  print("--- GENERATE ---")
  question = state["question"]
  documents = state["documents"]

  generation_result = generation_chain.invoke({"question": question, "context": documents})
  return {"documents": documents, "question": question, "generation": generation_result}