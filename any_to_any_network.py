# ============================================================
# DECENTRALIZED ROUTING PROMPT CONFIGURATION
# ============================================================
ROUTING_PROMPT_TEMPLATE = """
You are {agent_name} operating in a fully decentralized multi-agent system.
You have full autonomy and access to the complete conversation history shared across all agents.
Your task is to decide the next step in the workflow.

## Decision Process

Choose exactly ONE action:

### Continue solving
Select the best next agent:
- symbolic_reasoning_agent
- google_search_agent
- python_agent

### Finish the task
If the task is complete:
next_agent = "END"

and provide:
final_response = complete answer to the user

## Rules
- Never output null for next_agent.
- Only select ONE next agent.
- No supervisor exists.
- Any agent may delegate to any other agent.
- Avoid repeating completed work.
"""

from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from rich.markdown import Markdown

# Modular Imports
from agents import agents_registry
from tools import format_workflow_activity, console, MemorySaver

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int

class AgentRoutingDecision(BaseModel):
    reasoning: str
    next_agent: Literal["symbolic_reasoning_agent", "google_search_agent", "python_agent", "END"]
    final_response: str | None = None

MAX_STEPS = 5  # Increased slightly for dynamic multi-turn interactions

def autonomous_agent_node(state: AgentState, agent_name: str):
    step_count = state.get("step_count", 0)
    if step_count >= MAX_STEPS:
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content="Stopped: maximum step limit reached.")],
                "step_count": step_count,
            },
        )

    model_instance = agents_registry[agent_name][0]
    tools = agents_registry[agent_name][1]
    system_prompt = agents_registry[agent_name][3]

    react_agent = create_react_agent(model=model_instance, tools=tools, prompt=system_prompt)
    response = react_agent.invoke({"messages": state["messages"]})
    agent_messages = response["messages"]
    last_message = agent_messages[-1]

    routing_model = model_instance.with_structured_output(AgentRoutingDecision)
    formatted_routing_prompt = ROUTING_PROMPT_TEMPLATE.format(agent_name=agent_name)

    routing_decision = routing_model.invoke([
        SystemMessage(content=formatted_routing_prompt),
        *state["messages"],
        *agent_messages,
    ])

    next_agent = routing_decision.next_agent
    if next_agent not in {"symbolic_reasoning_agent", "google_search_agent", "python_agent", "END"}:
        next_agent = "symbolic_reasoning_agent"

    if next_agent == "END":
        final_response = routing_decision.final_response or last_message.content or "Task completed."
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=final_response)],
                "step_count": step_count + 1,
            },
        )

    return Command(
        goto=next_agent,
        update={
            "messages": [AIMessage(content=f"[{agent_name} → {next_agent}]\nReason: {routing_decision.reasoning}")],
            "step_count": step_count + 1,
        },
    )

def symbolic_reasoning_agent_node(state: AgentState):
    return autonomous_agent_node(state, "symbolic_reasoning_agent")

def google_search_agent_node(state: AgentState):
    return autonomous_agent_node(state, "google_search_agent")

def python_agent_node(state: AgentState):
    return autonomous_agent_node(state, "python_agent")

builder = StateGraph(AgentState)
builder.add_node("symbolic_reasoning_agent", symbolic_reasoning_agent_node)
builder.add_node("google_search_agent", google_search_agent_node)
builder.add_node("python_agent", python_agent_node)
builder.add_edge(START, "symbolic_reasoning_agent")

# Enable memory
memory = MemorySaver()
any_to_any_network = builder.compile(checkpointer=memory)

# ============================================================
# INTERACTIVE TERMINAL LOOP
# ============================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "any_to_any_session"}}
    console.print("[bold cyan]🤖 Any-to-Any Network Initialized. Type 'exit' to stop.[/bold cyan]\n")
    
    while True:
        try:
            user_input = input("User 👤> ")
            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("[bold red]Exiting App. Goodbye![/bold red]")
                break
            if not user_input.strip():
                continue
                
            for event in any_to_any_network.stream({"messages": [HumanMessage(content=user_input)], "step_count": 0}, config=config):
                markdown_text = format_workflow_activity(event)
                console.print(Markdown(markdown_text))
                
            print("\n" + "="*50 + "\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Exiting App. Goodbye![/bold red]")
            break
