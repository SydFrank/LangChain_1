from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
# Initialize the Tavily client
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """Search the web for information
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching the web for {query}")
    # return "Sydney weather is sunny"
    # Using the Tavily client to search the web
    return tavily.search(query=query)

#llm = ChatOpenAI()
llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-1!")
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in Sydney?")})
    print(result)
    


if __name__ == "__main__":
    main()
 