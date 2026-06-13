import os
import warnings
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.checkpoint.memory import MemorySaver
from serpapi import GoogleSearch
from langchain.tools import BaseTool
from rich.console import Console

# Silence non-critical logs and read local environment parameters
warnings.filterwarnings('ignore')
load_dotenv(override=True)

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
os.environ["SERP_API_KEY"] = os.getenv("SERP_API_KEY", "")

console = Console()

def format_workflow_activity(activity_log):
    """Sifts state logs to construct clean human-scannable runtime activity trees."""
    output = []
    for worker_id, worker_data in activity_log.items():
        output.append(f"\n# 🤖 Agent: {worker_id}\n")
        if "messages" not in worker_data:
            continue
        conversation_items = worker_data["messages"]
        if not isinstance(conversation_items, list):
            conversation_items = [conversation_items]
        for item in conversation_items:
            if isinstance(item, AIMessage):
                if item.content:
                    output.append(f"💬 **Response:** {item.content}\n")
                if hasattr(item, "tool_calls") and item.tool_calls:
                    for action in item.tool_calls:
                        output.append(f"🔧 **Tool:** `{action['name']}` | Args: `{action['args']}`\n")
            elif isinstance(item, ToolMessage):
                if item.content != "None":
                    output.append(f"⚙️ **Tool Result:** {item.content}\n")
    return "\n".join(output)

class GoogleSearchTool(BaseTool):
    name: str = "Google Search"
    description: str = "Searches the internet for a given topic and returns relevant results."

    def _run(self, query: str, top_k: int = 3) -> str:
        params = {
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "q": query,
            "api_key": os.environ["SERP_API_KEY"],
        }
        search = GoogleSearch(params)
        response = search.get_dict()
        if 'organic_results' not in response:
            return "Sorry, I couldn't find anything on that topic."
        results = response['organic_results']
        formatted_results = []
        for result in results[:top_k]:
            try:
                formatted_results.append('\n'.join([
                    f"Title: {result['title']}", 
                    f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}", 
                    "\n-----------------"
                ]))
            except KeyError:
                continue
        return '\n'.join(formatted_results)

python_tool = PythonREPLTool(
    name='Python_REPL',
    description='A Python shell. Use this to execute python commands. '
                'Input should be a valid python command. If you want to see the '
                'output of a value, you should print it out with `print(...)`.',
    return_direct=False,
    verbose=True
)

@tool
def google_search_scrape(input_str: str) -> str:
    """Given a user's full query, apply google search and answer using the search results."""
    return GoogleSearchTool().run(input_str)
