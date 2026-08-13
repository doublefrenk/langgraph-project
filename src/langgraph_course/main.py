from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from .chains import revisor, first_responder
from .tool_executor import execute_tool

MAX_ITERATION = 2

def draft_node(state: MessagesState) -> MessagesState:
    """ Draft the initial response"""
    response = first_responder.invoke({'messages' : state['messages']})
    return {'messages': response['messages']}

def revise_node(state: MessagesState) -> MessagesState:
    """ Revise the response"""
    response = revisor.invoke({'messages' : state['messages']})
    return {'messages': response['messages']}

def event_loop(state: MessagesState) -> Literal['execute_tool', 'END']:
    """ Determine whether to continue or end based on iteration count."""
    count_tool_visit = sum(isinstance(msg, ToolMessage) for msg in state['messages'])

    if count_tool_visit > MAX_ITERATION:
        return END
    return 'execute_tool'

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tool)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()

print(graph.get_graph().draw_mermaid())

res = graph.invoke({
    'messages': [
        {
            'role': 'user',
            "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
        }
    ]
})

last_message = res['messages'][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(res)