from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from .nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT="act"
LAST=-1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

# StateGraph rappresenta proprio la nostra macchina a stati e per funzionare ha bisogno di uno schema per ragionare, ovvero gli dobbiamo dire come è fatto lo stato su cui ragionano i nodi. MessagesState è perfetto perchè oltre a essere un dizinario tipizzato descrive anche come si fondono più messaggi se più nodi scrivono sulla stessa chiave.
flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
# E' equivalente a creare un arco START -> AGENT_REASON, ma START è un nodo speciale che non ha bisogno di essere definito. In questo caso il nodo AGENT_REASON è il primo nodo che viene eseguito quando si avvia il flusso.
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {ACT: ACT, END: END})
flow.add_edge(ACT, AGENT_REASON)


app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)