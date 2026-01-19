import inspect
from crewai import Crew, Process, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput

from paper_reviewer.agents import research_agent, critic_agent
from paper_reviewer.tools.vector_search import vector_search
from paper_reviewer.tools.ingest_arxiv import ingest_from_arxiv
from paper_reviewer.tools.refine_query_for_retrieval import refine_query_for_retrieval
from typing import Dict, List, Any

# =========================
# Retrieval Quality Gates
# =========================
MIN_UNIQUE_PAPERS = 2  # Need at least 2 different papers
MIN_TOP_SCORE = 0.6  # Best result must score above this
MIN_AVG_SCORE = 0.35  # Average of top results must be above this
MIN_CHUNK_LENGTH = 100  # Chunks should have substantial text (words)

import json

from pydantic import BaseModel, Field
from typing import List


class RetrievalResult(BaseModel):
    arxiv_id: str
    paper_title: str
    text: str
    score: float


class RetrievalOutput(BaseModel):
    results: List[RetrievalResult]
    metrics: dict  # or define a Metrics model


from paper_reviewer.tools.vector_search import get_last_vector_search_result


from paper_reviewer.tools.vector_search import get_last_vector_search_result


def should_ingest(output: TaskOutput) -> bool:
    data = get_last_vector_search_result()
    # print(data)
    metrics = data["metrics"]
    results = data["results"]

    unique_papers = metrics["unique_papers"]
    top_score = metrics["top_score"]
    avg_score = metrics["avg_score"]

    substantial_chunks = sum(
        1 for r in results if len(r.get("text", "").split()) >= MIN_CHUNK_LENGTH
    )

    if (
        unique_papers < MIN_UNIQUE_PAPERS
        or top_score < MIN_TOP_SCORE
        or avg_score < MIN_AVG_SCORE
        or substantial_chunks < 2
    ):
        print("🔄 Ingestion triggered due to low retrieval quality.")
        return True

    print("✅ Retrieval quality sufficient — skipping ingestion.")
    return False


# =========================
def run_research_crew(user_question: str):

    # -------------------------
    # Task 1 — Retrieve
    # -------------------------
    retrieve_task = Task(
        description=(
            f"Step 1: Use 'refine_query_for_retrieval' to optimize the search query.\n"
            f"Step 2: Use 'vector_search' to retrieve relevant paper excerpts.\n"
            f"Step 3: Summarize what you found.\n\n"
            f"User question: '{user_question}'\n\n"
            "The vector_search tool will automatically provide quality metrics. "
            "Review the results and report the key findings from the papers."
        ),
        expected_output=(
            "A retrieval report including:\n"
            "- ArXiv IDs and titles of papers found\n"
            "- Key excerpts and findings\n"
            "- Brief assessment of relevance to the question"
        ),
        agent=research_agent,
        tools=[refine_query_for_retrieval, vector_search],
    )
    # retrieve_task = Task(
    #     description=(
    #         f"Use 'refine_query_for_retrieval' to optimize the query for: '{user_question}'.\n"
    #         f"Then use 'vector_search' to retrieve relevant paper excerpts.\n"
    #         f"Do not summarize or explain — just execute the tools."
    #     ),
    #     expected_output="Raw output from the vector_search tool.",
    #     agent=research_agent,
    #     tools=[refine_query_for_retrieval, vector_search],
    #     # ❌ Remove output_pydantic
    # )

    # -------------------------
    # Task 2 — Conditional Ingest
    # -------------------------
    ingest_task = ConditionalTask(
        description=(
            f"The retrieval quality was insufficient based on the metrics.\n\n"
            f"Your task:\n"
            f"1. Use 'ingest_from_arxiv' to fetch NEW papers about: '{user_question}'\n"
            f"2. The tool will parse and add them to the vector database\n"
            f"3. Report how many papers were ingested and their arXiv IDs\n"
            f"4. DO NOT perform any vector search in this task\n\n"
            f"This will expand the knowledge base for better results in the next step."
        ),
        expected_output=(
            "A confirmation message including:\n"
            "- Number of papers successfully ingested\n"
            "- Their arXiv IDs\n"
            "- Confirmation they were added to the database"
        ),
        condition=should_ingest,
        agent=research_agent,
        # tools=[ingest_from_arxiv],
        tools=[ingest_from_arxiv, vector_search, refine_query_for_retrieval],
    )

    # -------------------------
    # Task 3 — Research Answer
    # -------------------------
    research_answer_task = Task(
        description=(
            f"Now provide a comprehensive answer to: '{user_question}'\n\n"
            "Steps:\n"
            "1. Use 'vector_search' to find the best evidence (database may be updated now)\n"
            "2. Extract specific methods, algorithms, results, and metrics\n"
            "3. Write a detailed technical answer that:\n"
            "   - Directly addresses the question\n"
            "   - Cites specific papers with arXiv IDs (e.g., 'According to Smith et al. (arXiv:2301.12345)...')\n"
            "   - Includes quantitative results and technical details\n"
            "   - Explains methodologies where relevant\n"
            "   - Notes limitations or disagreements between papers\n\n"
            "Be thorough and evidence-based. Every claim should cite a specific paper."
        ),
        expected_output=(
            "A comprehensive research answer that:\n"
            "- Fully addresses the user's question\n"
            "- Cites at least 2-3 relevant papers with arXiv IDs\n"
            "- Includes specific methods, metrics, and results\n"
            "- Is well-organized and clearly written\n"
            "- Properly attributes all information to sources"
        ),
        agent=research_agent,
        tools=[vector_search],
        context=[retrieve_task, ingest_task],
    )

    # Task 4 — Evidence Validation & Final Synthesis
    # -------------------------
    critique_task = Task(
        description=(
            f"As the final scientific authority, you must perform two critical phases:\n\n"
            f"PHASE 1: RIGOROUS VALIDATION\n"
            f"- Re-examine the original question: '{user_question}'\n"
            f"- Cross-verify EVERY claim in the research answer against the ACTUAL retrieved evidence\n"
            f"- Verify that all cited arXiv IDs appeared in vector_search results\n"
            f"- Confirm all technical details match the source excerpts precisely\n"
            f"- Identify and remove any unsupported assertions, speculative content, or inaccuracies\n\n"
            f"PHASE 2: COMPREHENSIVE SYNTHESIS\n"
            f"- Synthesize a definitive, self-contained answer that directly addresses the user's question\n"
            f"- Structure your response with clear sections: key findings, methodological approaches, quantitative results, limitations\n"
            f"- Cite ONLY papers that appeared in the retrieval results with CORRECT arXiv IDs\n"
            f"- Include SPECIFIC technical details (architectures, metrics, parameters) ONLY when directly supported by evidence\n"
            f"- Explicitly acknowledge gaps in the evidence where appropriate\n\n"
            f"Your final output must be authoritative, precise, and fully grounded in the available evidence."
        ),
        expected_output=(
            "A definitive scientific report that:\n"
            "- Begins with a concise summary answering the core question\n"
            "- Contains properly structured sections with clear headings\n"
            "- Cites ONLY verified papers from the retrieval results using correct arXiv IDs\n"
            "- Includes specific quantitative results and technical details with exact attribution\n"
            "- Acknowledges limitations and evidence gaps transparently\n"
            "- Maintains professional scientific tone throughout\n"
            "- Stands as a complete, self-contained reference on the topic"
        ),
        agent=critic_agent,
        context=[retrieve_task, research_answer_task, ingest_task],
    )

    # -------------------------
    # Crew
    # -------------------------
    crew = Crew(
        agents=[research_agent, critic_agent],
        tasks=[
            retrieve_task,
            ingest_task,
            research_answer_task,
            critique_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff()


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    question = input("Enter your research question: ")
    try:
        answer = run_research_crew(question)
        print("\n" + "=" * 80)
        print("FINAL ANSWER")
        print("=" * 80)
        print(answer)
    except Exception as e:
        print(f"\nError running crew: {e}")
        import traceback

        traceback.print_exc()
