from crewai.tools import tool
import logging
import json
import re
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import arxiv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import requests


BASE_DIR = Path("knowledge/data")
CHROMA_DIR = BASE_DIR / "chroma_lm"
LATEX_DIR = BASE_DIR / "latex_lm"
METADATA_PATH = BASE_DIR / "metadata_lm.json"

MAX_ARXIV_RESULTS = 6
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
DOWNLOAD_WORKERS = 6  # parallel downloads

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
LMSTUDIO_MODEL = "lm_studio/liquid/lfm2.5-1.2b"


def normalize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ", ".join(map(str, v))
        else:
            clean[k] = str(v)
    return clean


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call LM Studio embeddings API (batch support)."""
    if not texts:
        return []
    data = {"model": LMSTUDIO_MODEL, "input": texts}
    response = requests.post(LMSTUDIO_URL, json=data)
    response.raise_for_status()
    result = response.json()
    return [item["embedding"] for item in result["data"]]


# -------------------------
# Lazy Embeddings wrapper
# -------------------------
from langchain_core.embeddings import Embeddings


class LMStudioEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embeddings([text])[0]


_vectordb = None


import requests

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
LMSTUDIO_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call LM Studio embeddings API."""
    if isinstance(texts, str):
        texts = [texts]
    data = {"model": LMSTUDIO_MODEL, "input": texts}
    response = requests.post(LMSTUDIO_URL, json=data)
    response.raise_for_status()
    result = response.json()
    return [item["embedding"] for item in result["data"]]


from langchain_chroma import Chroma

_vectordb = None


def get_vectordb():
    """Lazy-load Chroma DB with LM Studio embeddings."""
    global _vectordb
    if _vectordb is None:
        embeddings = LMStudioEmbeddings()
        _vectordb = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,  # instance, not function
        )
    return _vectordb


def download_arxiv_latex(arxiv_id: str, save_dir: Path = LATEX_DIR) -> Path | None:
    """Download LaTeX source for an arXiv paper into a unique folder."""
    paper_dir = save_dir / arxiv_id
    paper_dir.mkdir(exist_ok=True)
    try:
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        response = requests.get(url)
        if response.status_code != 200:
            logging.warning(f"Failed to download LaTeX for {arxiv_id}")
            return None

        tar_path = paper_dir / f"{arxiv_id}.tar.gz"
        tar_path.write_bytes(response.content)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=paper_dir)
            tex_files = [f for f in tar.getnames() if f.endswith(".tex")]
            if not tex_files:
                logging.warning(f"No .tex files found for {arxiv_id}")
                return None
            # Pick largest tex file as main
            main_tex = max(tex_files, key=lambda f: tar.getmember(f).size)
            return paper_dir / main_tex
    except Exception as e:
        logging.warning(f"Error downloading LaTeX for {arxiv_id}: {e}")
        return None


def read_full_latex(current_file_path: Path) -> str:
    """Read LaTeX file and recursively resolve \input{} and \include{} commands."""
    content = current_file_path.read_text(encoding="utf-8")
    content = re.sub(r"(?<!\\)%.*", "", content)  # remove comments

    pattern = r"\\(?:input|include)\{(.+?)\}"

    def replace_input(m):
        # included_path = (main_tex_path.parent / m.group(1)).with_suffix(".tex")
        included_path = current_file_path.parent / f"{m.group(1)}.tex"
        # included_path = (current_file_path.parent / m.group(1)).with_suffix(".tex")

        if included_path.exists():
            return read_full_latex(included_path)
        else:
            logging.warning(f"Included file not found: {included_path}")
            return ""

    content = re.sub(pattern, replace_input, content)
    return content


def strip_latex_preamble_and_commands(latex_text: str) -> str:
    """
    Remove LaTeX preamble, \maketitle, \tableofcontents, and everything
    before \begin{document} and after \end{document}.
    """
    # Keep only content between \begin{document} and \end{document}
    parts = re.split(r"\\begin\{document\}", latex_text, maxsplit=1)
    if len(parts) == 2:
        latex_text = parts[1]
    latex_text = re.split(r"\\end\{document\}", latex_text, maxsplit=1)[0]

    # Remove \maketitle, \tableofcontents, \author, \title, \date commands
    latex_text = re.sub(
        r"\\(maketitle|tableofcontents|author|title|date|cite)(\{.*?\})?",
        "",
        latex_text,
    )

    # Remove remaining LaTeX comments
    latex_text = re.sub(r"(?<!\\)%.*", "", latex_text)

    return latex_text.strip()


def parse_latex_to_chunks(latex_content: str) -> list[dict]:
    """Parse LaTeX content into section/subsection chunks."""
    sections = re.split(r"\\section\{(.+?)\}", latex_content)
    chunks = []
    current_section = "Introduction"
    current_subsection = None

    for i, sec_content in enumerate(sections):
        if i == 0:
            text = sec_content.strip()
            if text:
                chunks.append(
                    {"section": current_section, "subsection": None, "text": text}
                )
        else:
            if i % 2 == 1:
                current_section = sec_content.strip()
            else:
                subsections = re.split(r"\\subsection\{(.+?)\}", sec_content)
                for j, sub_content in enumerate(subsections):
                    if j % 2 == 0:
                        text = sub_content.strip()
                        if text:
                            chunks.append(
                                {
                                    "section": current_section,
                                    "subsection": current_subsection,
                                    "text": text,
                                }
                            )
                    else:
                        current_subsection = sub_content.strip()
    return chunks


def enrich_chunk_with_captions(latex_chunk: str, latex_source: str) -> str:
    """Replace \ref{} references in chunk with figure/table captions."""
    ref_pattern = r"\\ref\{([^\}]+)\}"
    refs = re.findall(ref_pattern, latex_chunk)

    for ref_label in refs:
        # Figure
        fig_pattern = rf"\\begin\{{figure\}}.*?\\label\{{{re.escape(ref_label)}\}}.*?\\end\{{figure\}}"
        fig_match = re.search(fig_pattern, latex_source, re.DOTALL)
        if fig_match:
            figure_block = fig_match.group(0)
            caption_match = re.search(r"\\caption\{(.+?)\}", figure_block, re.DOTALL)
            caption = (
                caption_match.group(1).strip() if caption_match else "[No caption]"
            )
            replacement = f"[Figure: {caption}]"
            latex_chunk = re.sub(
                rf"\\ref\{{{re.escape(ref_label)}\}}",
                lambda m: replacement,
                latex_chunk,
            )
            continue

        # Table
        tab_pattern = rf"\\begin\{{table\}}.*?\\label\{{{re.escape(ref_label)}\}}.*?\\end\{{table\}}"
        tab_match = re.search(tab_pattern, latex_source, re.DOTALL)
        if tab_match:
            table_block = tab_match.group(0)
            caption_match = re.search(r"\\caption\{(.+?)\}", table_block, re.DOTALL)
            caption = (
                caption_match.group(1).strip() if caption_match else "[No caption]"
            )
            replacement = f"[Table: {caption}]"
            latex_chunk = re.sub(
                rf"\\ref\{{{re.escape(ref_label)}\}}",
                lambda m: replacement,
                latex_chunk,
            )
    return latex_chunk


# -------------------------
# Ingest LaTeX using LM Studio embeddings
# -------------------------
def ingest_latex(metadata, print_samples: int = 5):
    """Ingest LaTeX sources into Chroma using LM Studio embeddings."""
    vectordb = get_vectordb()
    all_docs = []
    sample_count = 0

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for meta in tqdm(metadata, desc="Processing papers"):
        arxiv_id = meta["arxiv_id"]
        tex_path = download_arxiv_latex(arxiv_id)
        if not tex_path or not tex_path.exists():
            logging.warning(f"LaTeX not found for {arxiv_id}, skipping.")
            continue

        full_latex = read_full_latex(tex_path)
        full_latex = strip_latex_preamble_and_commands(full_latex)
        chunks = parse_latex_to_chunks(full_latex)
        enriched_chunks = [
            enrich_chunk_with_captions(c["text"], full_latex) for c in chunks
        ]

        for c, text in zip(chunks, enriched_chunks):
            sub_chunks = char_splitter.split_text(text)
            for sub_text in sub_chunks:
                doc = Document(
                    page_content=sub_text,
                    metadata={
                        **meta,
                        "section": c["section"],
                        "subsection": c["subsection"],
                    },
                )
                all_docs.append(doc)

                if sample_count < print_samples:
                    print(f"Sample {sample_count+1}")
                    print(f"Title: {meta['title']}")
                    print(f"Section: {c['section']}")
                    print(f"Subsection: {c['subsection']}")
                    print(f"Text:\n{sub_text[:500]}...\n{'-'*80}")
                    sample_count += 1

    if all_docs:
        vectordb.add_documents(all_docs)
        # vectordb.persist()
        logging.info(f"Stored {len(all_docs)} LaTeX chunks in Chroma.")
    else:
        logging.warning("No LaTeX documents were ingested!")


def download_single_paper(paper):
    """Download metadata for a single arXiv paper."""
    try:
        meta = {
            "arxiv_id": paper.get_short_id(),
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "year": paper.published.year,
            "categories": paper.categories,
            "abstract": paper.summary,
        }
        return normalize_metadata(meta)
    except Exception as e:
        logging.warning(f"Failed to download metadata for {paper.title}: {e}")
        return None


def download_pdfs_parallel(query: str, max_results: int = MAX_ARXIV_RESULTS):
    """Download metadata for multiple papers in parallel."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results)
    papers = list(client.results(search))

    metadata = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_single_paper, p): p for p in papers}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading metadata"
        ):
            result = future.result()
            if result:
                metadata.append(result)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Downloaded {len(metadata)} metadata entries and saved metadata.")
    return metadata


@tool("ingest_from_arxiv")
def ingest_from_arxiv(keywords: str) -> str:
    """
    USE THIS ONLY WHEN vector_search returns insufficient or irrelevant results.

    This tool dynamically fetches new arXiv papers based on the user's natural language query,
    downloads their LaTeX source, parses sections, enriches content (e.g., figure captions),
    chunks the text, and ingests it into the vector database for future retrieval.

    Input: A clear, concise research topic in natural language (e.g., "contrastive learning for ECG classification").

    Output: Confirmation message with number of papers ingested.

    Expensive operation (takes 1–5 minutes). Only invoke when necessary!
    """

    # reuse YOUR existing functions
    import re

    clean_query = re.sub(r"[^\w\s\-]", " ", keywords)
    terms = [f'"{term}"' if " " in term else term for term in clean_query.split()]
    arxiv_query = " AND ".join(terms)

    metadata = download_pdfs_parallel(arxiv_query)
    ingest_latex(metadata)

    return f"Ingested {len(metadata)} new papers from arXiv."
