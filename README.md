An automated scientific research assistant that retrieves, evaluates, and summarizes scientific papers from arXiv using CrewAI LLM-based agents with vector search.

It is designed to ingest new papers automatically, assess retrieval quality, and generate evidence-based, citation-rich reports.

Features:

1. Query refinement: User can ask informally for the topic, and the LLM would reformulate it for the VectorDB and Arxiv querying.
2. Vector search retrieval: Searches a vector database of scientific papers and extracts relevant excerpts. 
3. Ingestion pipeline: the system intelligently asses the necessity of new data parsing, and retrieves the relevant papers in latex format from arxiv
4. Evidence-based summarization: Generates structured, citation-rich answers grounded in retrieved papers with provided metrics
5. Validation & Critique: The second agent ensures final answers are accurate, fully sourced, and scientifically rigorous.