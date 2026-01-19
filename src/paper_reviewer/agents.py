from crewai import Agent
import os
from paper_reviewer.tools.vector_search import vector_search
from paper_reviewer.tools.ingest_arxiv import ingest_from_arxiv
from paper_reviewer.tools.refine_query_for_retrieval import refine_query_for_retrieval

from crewai import LLM


llm = LLM(
    base_url="http://127.0.0.1:1234/v1",  # LM Studio API
    api_key="lm-studio",  # LM Studio ignores this
    model="local-model",  # just a placeholder; LM Studio uses loaded model
    temperature=0.7,
)

# ... existing code ...

research_agent = Agent(
    role="Scientific Research Specialist",
    goal=(
        "Retrieve and structure relevant scientific information from arXiv papers "
        "to directly answer the user's question with evidence."
    ),
    backstory=(
        "You are an expert AI research assistant capable of exploring scientific literature "
        "across any domain. You always start by reformulating the user's question into a "
        "precise, keyword-rich search query optimized for academic retrieval. "
        "Then you search the vector database. If results are insufficient, you fetch new papers from arXiv."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[
        refine_query_for_retrieval,
        vector_search,
        ingest_from_arxiv,
    ],  # ← add the new tool
)

critic_agent = Agent(
    role="Scientific Review & Synthesis Expert",
    goal=(
        "Critically evaluate retrieved paper excerpts, resolve contradictions, "
        "fill gaps, and produce a concise, well-attributed final answer."
    ),
    backstory=(
        "You are a senior interdisciplinary reviewer with deep expertise in evaluating "
        "scientific rigor, reproducibility, and relevance. You ensure every claim in the "
        "final response is grounded in cited evidence from specific arXiv papers."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)
