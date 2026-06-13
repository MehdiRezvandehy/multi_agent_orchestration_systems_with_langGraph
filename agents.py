from langchain_openai import ChatOpenAI
from tools import google_search_scrape, python_tool

# Central language model instances
model = ChatOpenAI(model='gpt-4o-mini')
supervisor_model = ChatOpenAI(model='gpt-4o')

# Agent blueprints
agents_registry = {
    "symbolic_reasoning_agent": (
        model,
        [],
        "Handles logical reasoning, algebraic manipulation, derivations, and problem decomposition.",
        "You are a symbolic reasoning expert. You do NOT write code. "
        "You reason step-by-step using logic, algebra, and mathematical transformations. "
        "You focus on understanding structure, deriving formulas, and explaining reasoning clearly."
    ),
    "google_search_agent": (
        model,
        [google_search_scrape],
        "Retrieves information using Google search and answers ONLY from retrieved results.",
        "\nYou are a STRICT retrieval-based search agent.\n\nDO NOT:\n- Write Python code\n- Perform calculations\n- Forecast values\n- Answer the user's full request\n\nWorkflow:\n1. Read search results\n2. Extract relevant facts\n3. Return final answer strictly based on them\n"
    ),
    "python_agent": (
        model,
        [python_tool],
        "Executes Python code and returns the result.",
        "You are a Python execution agent. Run valid Python code and return the output."
    )
}