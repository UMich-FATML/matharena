import requests
import os
import time
import json
import re
from ..api_client import APIClient
import yaml
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import threading
from loguru import logger

STORE_FOLDER = "paper_search_cache/"
MODEL_CONFIG_PATH = "configs/models/glm/ocr.yaml"
os.makedirs(STORE_FOLDER, exist_ok=True)
_s2_rate_lock = threading.Lock()
_s2_last_call = 0.0  # monotonic seconds
S2_API_KEY = os.getenv("S2_API_KEY")
if not S2_API_KEY:
    logger.warning("Semantic Scholar API key (S2_API_KEY) not set. You may hit rate limits.")

def _rate_limit_one_call_per_10s():
    global _s2_last_call
    with _s2_rate_lock:
        now = time.monotonic()
        wait = 2.0 - (now - _s2_last_call)
        if wait > 0:
            time.sleep(wait)
        # Reserve the slot for this thread (IMPORTANT: do this while holding the lock)
        _s2_last_call = time.monotonic()

def _abstract_snippet(abstract: str, max_chars: int = 1800, max_sentences: int = 10) -> str:
    if not abstract:
        return "Not available"
    abstract = " ".join(abstract.split())
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    picked = []
    total_chars = 0
    for sentence in sentences:
        if not sentence:
            continue
        candidate_len = len(sentence) + (1 if picked else 0)
        if picked and (total_chars + candidate_len > max_chars or len(picked) >= max_sentences):
            break
        picked.append(sentence)
        total_chars += candidate_len
        if total_chars >= max_chars or len(picked) >= max_sentences:
            break
    if not picked:
        return abstract[:max_chars]
    snippet = " ".join(picked)
    if len(snippet) < len(abstract):
        snippet += " ..."
    return snippet


def query_semantic_scholar(query, result_limit=100):
    """Extract paper data from Semantic Scholar API given a query string."""
    n_retries = 10
    for attempt in range(1, n_retries + 1):
        try:
            # Enforce global rate limit across threads
            _rate_limit_one_call_per_10s()
            rsp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers={"X-API-KEY": os.getenv("S2_API_KEY")} if os.getenv("S2_API_KEY") else {},
                params={
                    "query": query,
                    "limit": result_limit,
                    "fields": "paperId,title,authors,url,abstract,publicationVenue,textAvailability,publicationDate,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy",
                    "publicationDateOrYear": ":2025-08-31",
                },
            )
            rsp.raise_for_status()
            results = rsp.json()
            if "total" in results:
                break
        except Exception as e:
            # backoff a bit (don't hammer the API). keep it short; rate limiter already spaces calls.
            logger.warning(f"Attempt {attempt} failed to query Semantic Scholar: {str(e)}")
            time.sleep(min(2 ** attempt, 60))
    else:
        logger.error(f"Failed to query Semantic Scholar after {n_retries} attempts.")
        return "No results found."

    if results.get("total", 0) == 0:
        return "No results found."

    # Local filtering: only keep papers that have open-access PDF.
    filtered = []
    for paper in results.get("data", []):
        oap = paper.get("openAccessPdf") or {}
        url = oap.get("url")
        if "arxiv" in oap.get("disclaimer", "").lower():
            arxiv_id = oap["disclaimer"].split("/")[-1].split(",")[0]
            url = f"https://arxiv.org/pdf/{arxiv_id}"
        if not url:
            continue
        fields = set()
        fos = paper.get("fieldsOfStudy") or []
        if isinstance(fos, list):
            for f in fos:
                if isinstance(f, str):
                    fields.add(f.lower())
        s2_fos = paper.get("s2FieldsOfStudy") or []
        if isinstance(s2_fos, list):
            for item in s2_fos:
                if isinstance(item, dict):
                    cat = item.get("category")
                    if isinstance(cat, str):
                        fields.add(cat.lower())
                elif isinstance(item, str):
                    fields.add(item.lower())
        if "mathematics" not in fields:
            continue
        
        paper["url"] = url
        filtered.append(paper)

    top_papers = filtered[:15]
    if not top_papers:
        return "No results found with open-access PDF in the Mathematics topic."

    paper_string = ""
    for paper in top_papers:
        paper_id = (paper.get("paperId") or "unknown")[:10]
        paper_string += f"### {paper.get('title','(no title)')} (Paper ID: {paper_id}) ###\n"
        authors = paper.get("authors") or []
        author_names = []
        if isinstance(authors, list):
            for a in authors:
                if isinstance(a, dict):
                    n = a.get("name")
                    if isinstance(n, str) and n.strip():
                        author_names.append(n.strip())
        if author_names:
            shown = author_names[:12]
            author_line = ", ".join(shown)
            if len(author_names) > len(shown):
                author_line += ", et al."
            paper_string += f"Authors: {author_line}\n"
        else:
            paper_string += "Authors: Not available\n"
        paper_string += f"Abstract:\n{_abstract_snippet(paper.get('abstract'))}\n\n"

        json_path = os.path.join(STORE_FOLDER, f"{paper_id}.json")
        with open(json_path, "w") as f:
            json.dump(paper, f)

    return paper_string

def download_paper_pdf(paper_id):
    """Downloads the PDF of a paper given its Paper ID from the cache folder."""
    json_path = os.path.join(STORE_FOLDER, f"{paper_id}.json")
    if not os.path.exists(json_path):
        return None, "Paper ID not found in cache."

    with open(json_path, 'r') as f:
        paper = json.load(f)

    if paper["url"] is None:
        return None, "PDF not available for this paper."

    try:
        rsp = requests.get(paper['url'])
        rsp.raise_for_status()
        # check if content-type is pdf
        if 'application/pdf' not in rsp.headers.get('Content-Type', ''):
            return None, "The URL does not point to a valid PDF. PDF is therefore not available for this paper."
        pdf_path = os.path.join(STORE_FOLDER, f"{paper_id}.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(rsp.content)
        return pdf_path, "PDF downloaded successfully."
    except Exception as e:
        return None, f"Failed to download PDF: {str(e)}. Do not attempt again, since the URL is likely invalid, making the PDF unavailable for this paper."


def pdf_page_to_pil(doc: fitz.Document, page_index: int, dpi: int = 300) -> Image.Image:
    page = doc.load_page(page_index)
    # Render at requested DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    mime = "image/png"

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def ocr_batch(papers, store_folder=STORE_FOLDER, page_batch_size=None, dpi=300):
    papers = list(papers)
    if not papers:
        return {}

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    model_config.pop("human_readable_id", None)
    model_config.pop("date", None)
    client = APIClient(**model_config)
    if page_batch_size is None:
        page_batch_size = max(1, client.concurrent_requests * 4)

    docs = []
    output_paths = {}
    saved_papers = set()
    try:
        os.makedirs(store_folder, exist_ok=True)
        for pdf_path, store_filename in papers:
            doc = fitz.open(pdf_path)
            docs.append((store_filename, doc, [None] * doc.page_count))

        remaining_pages = [doc.page_count for _, doc, _ in docs]

        def save_paper(paper_idx):
            store_filename, _, md_pages = docs[paper_idx]
            output_path = os.path.join(store_folder, f"{store_filename}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(page or "" for page in md_pages))
            output_paths[store_filename] = output_path
            saved_papers.add(paper_idx)

        page_refs = [
            (paper_idx, page_idx)
            for paper_idx, (_, doc, _) in enumerate(docs)
            for page_idx in range(doc.page_count)
        ]
        logger.info(
            f"OCR batch: {len(docs)} PDFs, {len(page_refs)} pages, page_batch_size={page_batch_size}."
        )

        for start in range(0, len(page_refs), page_batch_size):
            chunk_refs = page_refs[start : start + page_batch_size]
            pages_queries = []
            for paper_idx, page_idx in chunk_refs:
                img = pdf_page_to_pil(docs[paper_idx][1], page_idx, dpi=dpi)
                data_uri = pil_to_data_uri(img)
                pages_queries.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_uri}},
                                {"type": "text", "text": "Text Recognition:"},
                            ],
                        }
                    ]
                )

            for local_idx, result, _ in client.run_queries(pages_queries, no_tqdm=True):
                paper_idx, page_idx = chunk_refs[local_idx]
                markdown = (result[-1].get("content") or "").strip() if result else ""
                if docs[paper_idx][2][page_idx] is None:
                    remaining_pages[paper_idx] -= 1
                docs[paper_idx][2][page_idx] = f"##### Page {page_idx + 1} #####\n\n{markdown}"
                if remaining_pages[paper_idx] == 0 and paper_idx not in saved_papers:
                    save_paper(paper_idx)

        for paper_idx in range(len(docs)):
            if paper_idx not in saved_papers:
                save_paper(paper_idx)
        return output_paths
    finally:
        for _, doc, _ in docs:
            doc.close()


def ocr(pdf_path, store_filename, store_folder=STORE_FOLDER):
    ocr_batch([(pdf_path, store_filename)], store_folder=store_folder)

def ocr_paper(paper_id):
    pdf_path = os.path.join(STORE_FOLDER, f"{paper_id}.pdf")
    ocr(pdf_path, store_filename=paper_id)

def check_and_prepare_paper(paper_id):
    json_path = os.path.join(STORE_FOLDER, f"{paper_id}.json")
    if not os.path.exists(json_path):
        return "Paper ID not found."
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    if not os.path.exists(md_path):
        done, message = download_paper_pdf(paper_id)
        if not done:
            return message
        ocr_paper(paper_id)
    return "Paper is ready."

def read_paper(paper_id):
    output_prepare = check_and_prepare_paper(paper_id)
    if output_prepare != "Paper is ready.":
        return output_prepare
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    
    with open(md_path, 'r') as f:
        content = f.read()
    return content

def read_pages(paper_id, min_page=None, max_page=None):
    output_prepare = check_and_prepare_paper(paper_id)
    if output_prepare != "Paper is ready.":
        return output_prepare
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    
    with open(md_path, 'r') as f:
        content = f.read()
    
    pages = content.split("##### Page ")
    selected_pages = []
    for page in pages[1:]:
        page_number_str, page_content = page.split(" #####\n\n", 1)
        page_number = int(page_number_str.strip())
        if (min_page is None or page_number >= min_page) and (max_page is None or page_number <= max_page):
            selected_pages.append(f"##### Page {page_number} #####\n\n{page_content}")
    
    return "\n\n".join(selected_pages)

def find_in_paper(paper_id, string):
    # returns the first 10 occurrences of string in the paper markdown
    # plus five sentences before and after it, as well as the page number
    output_prepare = check_and_prepare_paper(paper_id)
    if output_prepare != "Paper is ready.":
        return output_prepare
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    with open(md_path, 'r') as f:
        content = f.read()
    pages = content.split("##### Page ")
    occurrences = []
    for page in pages[1:]:
        page_number_str, page_content = page.split(" #####\n\n", 1)
        page_number = int(page_number_str.strip())
        sentences = page_content.split('. ')
        for i, sentence in enumerate(sentences):
            if str(string).lower() in sentence.lower():
                start = max(0, i - 3)
                end = min(len(sentences), i + 3)
                context = '. '.join(sentences[start:end])
                occurrences.append(f"### Occurrence on Page {page_number} ###\n...\n{context}\n...")
                if len(occurrences) >= 10:
                    break
        if len(occurrences) >= 10:
            break
    if not occurrences:
        return "No occurrences found."
    return "\n\n".join(occurrences)
