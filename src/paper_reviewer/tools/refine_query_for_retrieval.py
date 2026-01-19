from crewai.tools import tool


@tool("refine_query_for_retrieval")
def refine_query_for_retrieval(user_question: str) -> str:
    """
    Reformulate the user's natural language question into a concise, keyword-rich query
    optimized for scientific literature search (both vector DB and arXiv).

    The output should:
    - Preserve core intent
    - Include technical terms if ambiguous
    - Be suitable for semantic search and API queries
    - Avoid questions or fluff (e.g., no "What is..." or "Can you explain...")

    Example:
      Input: "How do GNNs predict seizures from EEG?"
      Output: "graph neural networks seizure prediction EEG"
    """
    
    pass  