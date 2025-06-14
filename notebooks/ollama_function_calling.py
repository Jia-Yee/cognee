from langchain.tools import tool

@tool
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b

from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, AgentType

ollama = ChatOllama(model="qwen2.5", base_url="http://192.168.1.4:11434")
tools = [add_two_numbers]

agent = initialize_agent(
    tools,
    ollama,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

result = agent.run("Add 5 and 7 using the add_two_numbers function.")
print(result)