import os
from tavily import TavilyClient


def search_query_with_tavily(query: str):
    """
    Perform a web search using Tavily and return the answer and sources.

    Parameters:
        query (str): The search query string.

    Returns:
        dict: A dictionary with 'answer' and 'sources' keys.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY environment variable not set.")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, include_answer=True,
                             search_depth="advanced",  # or "deep"
                             include_raw_content=True,  # optional: fetch full-page content
                             llm_options={"max_tokens": 800})  # try increasing answer size)

    answer = response.get("answer", "No answer found.")
    sources = [
        {"title": res["title"], "url": res["url"], "content": res["content"]}
        for res in response.get("results", [])
    ]

    return {"answer": answer, "sources": sources}


# Example usage
if __name__ == "__main__":
    query = input("Enter your query: ")
    result = search_query_with_tavily(query)

    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for i, src in enumerate(result["sources"], start=1):
        print(f"{i}. {src['title']} – {src['url']}")
        print(f"   {src['content']}\n")
