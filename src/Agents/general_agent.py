import os
import sys
import asyncio
from langgraph.graph import StateGraph, START, END

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.utils import get_llm
from src.components.prompts import GENERAL_SYSTEM_PROMPT
from src.components.states import GeneralAgentInput

# Respond to general query
async def respond_to_general_query(state: GeneralAgentInput):
    """Generate a response to a general query."""
    llm = get_llm()
    system_prompt = GENERAL_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["query"]}
    ]
    response = await llm.ainvoke(messages)
    return {"response": response}


# Create the graph
def create_general_agent():
    workflow = StateGraph(GeneralAgentInput)

    workflow.add_node("respond_to_general_query", respond_to_general_query)
    workflow.add_edge(START, "respond_to_general_query")
    workflow.add_edge("respond_to_general_query", END)
    
    general_agent = workflow.compile()
    general_agent.name = "GeneralAgent"
    return general_agent


# Example usage:
#graph = create_general_agent()
#state = GeneralAgentInput(query="What is the capital of France?")
#result = asyncio.run(graph.ainvoke(state))
#print(result["response"].content)
