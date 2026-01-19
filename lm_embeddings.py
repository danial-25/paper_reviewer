import logging
import json
import re
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import arxiv
import requests

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# -------------------------
# Setup paths & logging
# -------------------------
BASE_DIR = Path("knowledge/data")
CHROMA_DIR = BASE_DIR / "chroma_lm"
LATEX_DIR = BASE_DIR / "latex_lm"
METADATA_PATH = BASE_DIR / "metadata_lm.json"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -------------------------
# Config
# -------------------------
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
LMSTUDIO_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"

MAX_ARXIV_RESULTS = 60
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
DOWNLOAD_WORKERS = 8

KEYWORDS = ["EEG", "seizure detection", "machine learning"]
ARXIV_QUERY = " AND ".join(KEYWORDS + ["epilepsy", "prediction"])


# -------------------------
# LM Studio Embedding Wrapper
# -------------------------
class LMStudioEmbeddings:
    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            texts = [texts]
        data = {"model": self.model, "input": texts}
        response = requests.post(self.url, json=data)
        response.raise_for_status()
        result = response.json()
        return [item["embedding"] for item in result["data"]]


# -------------------------
# Helpers
# -------------------------
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


def download_arxiv_latex(arxiv_id: str, save_dir: Path = LATEX_DIR) -> Path | None:
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
            main_tex = max(tex_files, key=lambda f: tar.getmember(f).size)
            return paper_dir / main_tex
    except Exception as e:
        logging.warning(f"Error downloading LaTeX for {arxiv_id}: {e}")
        return None


def read_full_latex(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    content = re.sub(r"(?<!\\)%.*", "", content)

    def repl(m):
        inc = path.parent / f"{m.group(1)}.tex"
        return read_full_latex(inc) if inc.exists() else ""

    return re.sub(r"\\(?:input|include)\{(.+?)\}", repl, content)


def strip_latex_preamble_and_commands(text: str) -> str:
    text = re.split(r"\\begin\{document\}", text, maxsplit=1)[-1]
    text = re.split(r"\\end\{document\}", text, maxsplit=1)[0]
    text = re.sub(
        r"\\(maketitle|tableofcontents|author|title|date|cite)(\{.*?\})?", "", text
    )
    text = re.sub(r"(?<!\\)%.*", "", text)
    return text.strip()


def parse_latex_to_chunks(text: str) -> list[dict]:
    sections = re.split(r"\\section\{(.+?)\}", text)
    chunks = []
    current_section = "Introduction"
    current_subsection = None

    for i, part in enumerate(sections):
        if i == 0:
            if part.strip():
                chunks.append(
                    {"section": current_section, "subsection": None, "text": part}
                )
        elif i % 2 == 1:
            current_section = part.strip()
        else:
            subs = re.split(r"\\subsection\{(.+?)\}", part)
            for j, sub in enumerate(subs):
                if j % 2 == 0 and sub.strip():
                    chunks.append(
                        {
                            "section": current_section,
                            "subsection": current_subsection,
                            "text": sub,
                        }
                    )
                elif j % 2 == 1:
                    current_subsection = sub.strip()
    return chunks


def download_single_paper(paper):
    try:
        return normalize_metadata(
            {
                "arxiv_id": paper.get_short_id(),
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "year": paper.published.year,
                "categories": paper.categories,
                "abstract": paper.summary,
            }
        )
    except Exception as e:
        logging.warning(e)
        return None


def download_pdfs_parallel(query: str):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=MAX_ARXIV_RESULTS)
    papers = list(client.results(search))

    metadata = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = [ex.submit(download_single_paper, p) for p in papers]
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading metadata"
        ):
            if f.result():
                metadata.append(f.result())

    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


# -------------------------
# Ingestion
# -------------------------
def ingest_latex(metadata):
    embeddings = LMStudioEmbeddings(url=LMSTUDIO_URL, model=LMSTUDIO_MODEL)
    vectordb = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    docs = []
    for meta in tqdm(metadata, desc="Processing papers"):
        tex = download_arxiv_latex(meta["arxiv_id"])
        if not tex:
            continue
        latex = strip_latex_preamble_and_commands(read_full_latex(tex))
        chunks = parse_latex_to_chunks(latex)

        for c in chunks:
            for piece in splitter.split_text(c["text"]):
                docs.append(
                    Document(
                        page_content=piece,
                        metadata={
                            **meta,
                            "section": c["section"],
                            "subsection": c["subsection"],
                        },
                    )
                )

    vectordb.add_documents(docs)
    vectordb.persist()
    logging.info(f"Stored {len(docs)} chunks in Chroma")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Download metadata if not exists
    if not METADATA_PATH.exists():
        metadata = download_pdfs_parallel(ARXIV_QUERY)
    else:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    ingest_latex(metadata)
