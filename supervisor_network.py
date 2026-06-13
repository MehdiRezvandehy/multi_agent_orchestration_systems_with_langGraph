from typing import Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from rich.markdown import Markdown

# Modular Imports
from agents import agents_registry, supervisor_model
from tools import format_workflow_activity, console, MemorySaver

supervisor_system_prompt = """
You are a supervisor coordinating a team of specialized agents.
Based on the user's request, decide which agent should handle the next step.

1. Determine whether the task is already completed -> choose END
2. Otherwise choose the next best agent. Prefer END over unnecessary extra work.

Available agents:
- symbolic_reasoning_agent: Handles logic and problem decomposition.
- google_search_agent: Retrieves real-time information from the web.
- python_agent: Executes Python code.
"""

def supervisor(state: MessagesState) -> Command[Literal["symbolic_reasoning_agent", "google_search_agent", "python_agent", END]]:
    class SupervisorResponse(BaseModel):
        next_agent_reasoning: str
        next_agent: Literal["symbolic_reasoning_agent", "google_search_agent", "python_agent", "END"]
        final_response: str | None = None

    structured_supervisor = supervisor_model.with_structured_output(SupervisorResponse)
    response = structured_supervisor.invoke([SystemMessage(content=supervisor_system_prompt)] + state["messages"])

    if response.next_agent == "END":
        return Command(goto=END, update={"messages": [AIMessage(content=response.final_response or "Task completed.")]})
    
    return Command(goto=response.next_agent, update={"messages": AIMessage(content=response.next_agent_reasoning)})

def symbolic_reasoning_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(agents_registry['symbolic_reasoning_agent'][0], agents_registry['symbolic_reasoning_agent'][1], prompt=agents_registry['symbolic_reasoning_agent'][3])
    return Command(goto="supervisor", update={"messages": agent.invoke(state)["messages"]})

def google_search_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(agents_registry['google_search_agent'][0], agents_registry['google_search_agent'][1], prompt=agents_registry['google_search_agent'][3])
    return Command(goto="supervisor", update={"messages": agent.invoke(state)["messages"]})

def python_agent(state: MessagesState) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(agents_registry['python_agent'][0], agents_registry['python_agent'][1], prompt=agents_registry['python_agent'][3])
    return Command(goto="supervisor", update={"messages": agent.invoke(state)["messages"]})

builder = StateGraph(MessagesState)
builder.add_node(supervisor)
builder.add_node(symbolic_reasoning_agent)
builder.add_node(google_search_agent)
builder.add_node(python_agent)
builder.add_edge(START, "supervisor")

# Enable memory
memory = MemorySaver()
supervisor_network = builder.compile(checkpointer=memory)

# ============================================================
# INTERACTIVE TERMINAL LOOP
# ============================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "supervisor_session"}}
    console.print("[bold green]🤖 Supervisor Network Initialized. Type 'exit' to stop.[/bold green]\n")
    
    while True:
        try:
            user_input = input("User 👤> ")
            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("[bold red]Exiting App. Goodbye![/bold red]")
                break
            if not user_input.strip():
                continue
                
            for event in supervisor_network.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
                markdown_text = format_workflow_activity(event)
                console.print(Markdown(markdown_text))
                
            print("\n" + "="*50 + "\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Exiting App. Goodbye![/bold red]")
            break
