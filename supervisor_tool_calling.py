supervisor_with_tools_prompt = """
You are a supervisor responsible for managing a team of specialized agents.
Based on the user's request, invoke the required agent tools sequentially to solve the problem.

Available tools:
1. symbolic_reasoning_tool: Logical reasoning and decomposition.
2. google_search_tool: Search information from Google.
3. python_tool: Executes raw Python code.
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from rich.markdown import Markdown

# Modular Imports
from agents import agents_registry, model
from tools import format_workflow_activity, console, MemorySaver

@tool
def symbolic_reasoning_tool(task: str) -> str:
    """Handle logical reasoning, algebraic manipulation, and problem decomposition."""
    agent = create_react_agent(agents_registry['symbolic_reasoning_agent'][0], agents_registry['symbolic_reasoning_agent'][1], prompt=agents_registry['symbolic_reasoning_agent'][3])
    return agent.invoke({'messages': [HumanMessage(content=task)]})['messages'][-1].content

@tool
def google_search_tool(task: str) -> str:
    """Retrieves and answers questions using up-to-date information from Google search."""
    agent = create_react_agent(agents_registry['google_search_agent'][0], agents_registry['google_search_agent'][1], prompt=agents_registry['google_search_agent'][3])
    return agent.invoke({'messages': [HumanMessage(content=task)]})['messages'][-1].content

@tool
def python_tool_exec(task: str) -> str:
    """Executes Python code and returns the result."""
    agent = create_react_agent(agents_registry['python_agent'][0], agents_registry['python_agent'][1], prompt=agents_registry['python_agent'][3])
    return agent.invoke({'messages': [HumanMessage(content=task)]})['messages'][-1].content

tools = [symbolic_reasoning_tool, google_search_tool, python_tool_exec]

# Enable memory
memory = MemorySaver()

supervisor_with_tools = create_react_agent(
    model, 
    tools, 
    prompt=supervisor_with_tools_prompt, 
    name="tool_calling_supervisor",
    checkpointer=memory
)

# ============================================================
# INTERACTIVE TERMINAL LOOP
# ============================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "tool_calling_session"}}
    console.print("[bold magenta]🤖 Tool-Calling Supervisor Initialized. Type 'exit' to stop.[/bold magenta]\n")
    
    while True:
        try:
            user_input = input("User 👤> ")
            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("[bold red]Exiting App. Goodbye![/bold red]")
                break
            if not user_input.strip():
                continue
                
            for event in supervisor_with_tools.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
                markdown_text = format_workflow_activity(event)
                console.print(Markdown(markdown_text))
                
            print("\n" + "="*50 + "\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Exiting App. Goodbye![/bold red]")
            break
