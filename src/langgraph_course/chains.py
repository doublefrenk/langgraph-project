import datetime
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from .schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model_name="gpt-4o-mini")
parser = JsonOutputToolsParser(return_id = true)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages([
  (
    "system",
            """You are expert researcher.
    Current time: {time}

    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvement.
    3. Recommend search queries to research information and improve your answer.""",
  ),
  MessagesPlaceholder(variable_name="messages")
  ("system", "Answer the user's question above using the required format."),
]).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

"""Questa riga è molto interessante: il punto di partenza è capire che llm
NON esegue nessun tool, ma solo genera un output che può essere interpretato come una chiamata a un tool
attraverso un json. Ciò che esegue la funzione può essere solamente un agente (create_agent(...)) oppure il grafo

tools=[AnswerQuestion]: Dice all'LLM che ha a disposizione lo schema della classe/funzione AnswerQuestion.

tool_choice="AnswerQuestion": Obbliga (forza) l'LLM a rispondere sempre generando una struttura di argomenti compatibile con il tool AnswerQuestion.

Quando fai first_responder.invoke(...), l'LLM restituisce un oggetto AIMessage.

Questo oggetto non contiene il risultato dell'esecuzione della funzione Python, ma contiene la struttura dati con i parametri generati dall'LLM, che trovi all'interno della proprietà tool_calls:

AIMessage(
    content="",
    tool_calls=[
        {
            "name": "AnswerQuestion",
            "args": {
                "answer": "La risposta formattata...",
                "reflection": "La riflessione/critica..."
            },
            "id": "call_12345"
        }
    ]
)"""
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

"""In questo caso il main serve solo per testare e fare debug sulle chains senza dover richiamare il grafo"""

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
