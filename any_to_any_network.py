# ============================================================
# DECENTRALIZED ROUTING PROMPT CONFIGURATION
# ============================================================
ROUTING_PROMPT_TEMPLATE = """
You are {agent_name} operating in a fully decentralized multi-agent network.

You have access to the complete shared conversation history and can independently decide 
the next step required to solve the user's request.

Your objective is to either:
1. Complete the task if you have sufficient information and capability.
2. Delegate the task to the most appropriate agent if additional work is required.

## Available Agents

- symbolic_reasoning_agent:
  Planning, decomposition, logical reasoning.

- google_search_agent:
  Search the web and retrieve current information.

- python_agent:
  Execute Python code, perform calculations,
  data analysis, and visualization.

## Decision Process

Choose exactly ONE action:

### Option 1: Continue Solving

If more work is required, select the single best agent to perform the next step:

next_agent = "<agent_name>"

Provide a brief explanation of why this agent is best suited for the next action.

### Option 2: Finish the Task

If the user's request has been fully resolved:

next_agent = "END"

Provide:

final_response = "<complete answer for the user>"

## Delegation Guidelines

- Delegate only if another agent can make better progress.
- Choose the agent best suited for the remaining work.
- Avoid unnecessary handoffs and repeated work.
- Complete the task yourself whenever possible.

## Rules

- Always provide a valid `next_agent`.
- Select exactly one agent or `END`.
- Never use `null`.
- No supervisor exists.
- Any agent may delegate to any other agent.    

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

    # ------------------------------------------------
    # Safety fallback
    # ------------------------------------------------
    valid_agents = {
        "symbolic_reasoning_agent",
        "google_search_agent",
        "python_agent",
        "END",
    }

    if next_agent not in valid_agents:
        next_agent = "symbolic_reasoning_agent"

    # ------------------------------------------------
    # Finish workflow
    # ------------------------------------------------
    if next_agent == "END":

        final_response = (
            routing_decision.final_response
            or last_message.content
            or "Task completed."
        )

        return Command(
            goto=END,
            update={
                "messages": [
                    AIMessage(content=final_response)
                ],
                "step_count": step_count + 1,
            },
        )

    # ------------------------------------------------
    # Delegate to next agent
    # ------------------------------------------------
    return Command(
        goto=next_agent,
        update={
            "messages": [
                AIMessage(
                    content=(
                        f"[{agent_name} → {next_agent}]\n"
                        f"Reason: {routing_decision.reasoning}"
                    )
                )
            ],
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
