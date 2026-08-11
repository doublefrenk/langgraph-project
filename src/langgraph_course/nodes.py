from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage

from .chains import generate_chain, reflect_chain

def generation_node(state: MessagesState):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessagesState):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}
