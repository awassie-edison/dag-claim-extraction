#!/usr/bin/env python3
"""
Run the claim extraction pipeline (Stage 0 → 1 → 2 → 3 → 4) via the Anthropic API.

V10: Adds Stage 4 (analysis protocol enhancement via code repository mining).
Uses temperature=0 for deterministic output.

Usage:
    # Run all papers (all stages):
    python run_pipeline.py Papers_claims/*/

    # Run specific papers:
    python run_pipeline.py Papers_claims/TMS_0 Papers_claims/TMS_1

    # Resume from Stage 4 (code enhancement) on existing DAGs:
    python run_pipeline.py --from-stage 4 Papers_claims/TMS_0

Each paper directory should contain a PDF file. All outputs (dataset_links.csv,
dataset_profile.md, section_inventory.csv, claims.csv, key_terms.csv,
claim_dag.json) are written into the same directory alongside the PDF.

Stage 4 fetches the paper's code repository and enhances analysis_detail fields
in claim_dag.json. If no code repo is found in the paper, Stage 4 is skipped.
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL = "claude-opus-4-6"
TEMPERATURE = 0.0
MAX_TOKENS = 128000       # Maximum output tokens — no cost concern
MAX_STAGE_RETRIES = 3     # Retries per stage on any failure (API, parse, etc.)
STATUS_FILE = PROJECT_ROOT / "pipeline_status.json"

STAGE_PROMPTS = {
    0: PROJECT_ROOT / "stage0_dataset_link_extraction_prompt.txt",
    1: PROJECT_ROOT / "stage1_dataset_identification_prompt.txt",
    2: PROJECT_ROOT / "stage2_claim_extraction_prompt.txt",
    3: PROJECT_ROOT / "stage3_claim_dag_construction_prompt.txt",
    4: PROJECT_ROOT / "stage4_analysis_protocol_prompt.txt",
}

STAGE_NAMES = {
    0: "Dataset link extraction",
    1: "Dataset identification",
    2: "Claim extraction",
    3: "DAG construction",
    4: "Analysis protocol enhancement",
}

# Stage 4 Pass 2: Prompt for improving unmatched analysis_detail from paper text
STAGE4_PASS2_PROMPT = """TASK: ANALYSIS DETAIL IMPROVEMENT

You are a computational biology expert. Below are DAG nodes whose
analysis_detail fields need improvement. Each node represents a scientific
claim derived from the paper. Your task is to rewrite each analysis_detail
into a detailed, self-contained analysis protocol using ONLY information
from the paper's Methods, Results, and Supplementary sections.

QUALITY STANDARDS FOR analysis_detail:
Each analysis_detail must be a step-by-step protocol that an AI agent could
follow to reproduce the analysis. It must include:

1. INPUT DATA — What data goes in (file types, which dataset, any subsetting
   or filtering applied before the analysis begins).
2. PREPROCESSING — Any normalization, transformation, batch correction, or
   QC steps applied to the data before the core analysis.
3. CORE ANALYSIS — The specific analytical method or statistical test, with:
   - The tool/package/algorithm used (e.g., MAST, DESeq2, Louvain clustering)
   - All parameters and thresholds mentioned in the paper (p-value cutoffs,
     fold-change thresholds, number of components, resolution, etc.)
   - Any covariates, contrasts, or groupings used
4. OUTPUT — What the analysis produces (e.g., a list of DE genes, a UMAP
   embedding, cluster assignments, a p-value).

WHAT TO AVOID:
- Do NOT include results, conclusions, or findings (e.g., "which showed
  upregulation of X" or "revealing that Y increases with age"). The
  analysis_detail describes HOW to do the analysis, not WHAT was found.
- Do NOT reference other nodes ("See L7", "as in I3").
- Do NOT reference specific script file names.
- Do NOT include vague descriptions like "standard bioinformatics pipeline"
  or "data was analyzed" — be specific about what was done.

RULES:
- Only use information from the paper. Do not invent parameters.
- If the paper does not specify a parameter, say so explicitly (e.g.,
  "resolution parameter not specified in the paper").
- Preserve the analysis_type field unchanged.
- Use ASCII characters only (codes 32-126).

OUTPUT FORMAT:
Return a JSON array with one object per node. Each object must have:
  "id": the node ID (unchanged)
  "analysis_detail": the improved protocol text

Return ONLY the JSON array."""

# Stage 4: Code repository file extensions to include
CODE_EXTENSIONS = {
    ".r", ".rmd", ".py", ".ipynb", ".sh", ".bash",
    ".nf", ".smk", ".wdl", ".jl",
}
# Always include these filenames regardless of extension
CODE_SPECIAL_FILES = {
    "readme.md", "readme.rst", "readme.txt", "readme",
    "requirements.txt", "environment.yml", "environment.yaml",
    "renv.lock", "description", "setup.py", "setup.cfg",
    "pyproject.toml", "snakefile", "makefile",
}
# Directories to skip entirely
CODE_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".snakemake",
    ".nextflow", "work", ".ipynb_checkpoints", "renv",
}


# ---------------------------------------------------------------------------
# Status tracker (thread-safe)
# ---------------------------------------------------------------------------
class StatusTracker:
    """Thread-safe status tracker that writes to pipeline_status.json."""

    def __init__(self, paper_names, from_stage=0):
        self._lock = threading.Lock()
        self._start_time = datetime.now().isoformat()
        self._status = {
            "pipeline_start": self._start_time,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "from_stage": from_stage,
            "papers": {},
        }
        for name in paper_names:
            self._status["papers"][name] = {
                "status": "pending",
                "stages": {},
            }
        self._write()

    def stage_start(self, paper_name, stage_num):
        with self._lock:
            paper = self._status["papers"][paper_name]
            paper["status"] = "running"
            paper["stages"][str(stage_num)] = {
                "name": STAGE_NAMES[stage_num],
                "status": "running",
                "start": datetime.now().isoformat(),
            }
            self._write()

    def stage_done(self, paper_name, stage_num, *, tokens_in=0, tokens_out=0,
                   elapsed=0, detail=""):
        with self._lock:
            stage = self._status["papers"][paper_name]["stages"][str(stage_num)]
            stage["status"] = "done"
            stage["end"] = datetime.now().isoformat()
            stage["elapsed_s"] = round(elapsed, 1)
            stage["tokens_in"] = tokens_in
            stage["tokens_out"] = tokens_out
            if detail:
                stage["detail"] = detail
            self._write()

    def stage_failed(self, paper_name, stage_num, error):
        with self._lock:
            paper = self._status["papers"][paper_name]
            paper["status"] = "failed"
            stage = paper["stages"].get(str(stage_num), {})
            stage["status"] = "failed"
            stage["error"] = str(error)
            stage["end"] = datetime.now().isoformat()
            paper["stages"][str(stage_num)] = stage
            self._write()

    def paper_done(self, paper_name, elapsed):
        with self._lock:
            paper = self._status["papers"][paper_name]
            paper["status"] = "done"
            paper["total_elapsed_s"] = round(elapsed, 1)
            self._write()

    def paper_failed(self, paper_name):
        with self._lock:
            self._status["papers"][paper_name]["status"] = "failed"
            self._write()

    def finish(self):
        with self._lock:
            self._status["pipeline_end"] = datetime.now().isoformat()
            self._write()

    def _write(self):
        with open(STATUS_FILE, "w") as f:
            json.dump(self._status, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF, with page markers."""
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages.append(f"--- Page {i} ---\n{text}")
    return "\n\n".join(pages)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def find_pdf(paper_dir: Path) -> Path:
    """Find the PDF file in a paper directory."""
    pdfs = list(paper_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {paper_dir}")
    if len(pdfs) > 1:
        print(f"  WARNING: Multiple PDFs in {paper_dir}, using {pdfs[0].name}")
    return pdfs[0]


def call_claude(client, messages, *, max_retries=3):
    """
    Call the Anthropic API with temperature=0 and retry on transient errors.
    Uses streaming to handle long Opus responses.
    Returns (text, input_tokens, output_tokens, elapsed).
    """
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=messages,
            ) as stream:
                for _ in stream:
                    pass
                response = stream.get_final_message()

            elapsed = time.time() - t0
            text = response.content[0].text
            usage = response.usage
            print(f"    Response: {usage.input_tokens:,} in / {usage.output_tokens:,} out "
                  f"({elapsed:.1f}s, stop={response.stop_reason})")

            if response.stop_reason != "end_turn":
                print(f"    WARNING: stop_reason={response.stop_reason} — response may be truncated")

            return text, usage.input_tokens, usage.output_tokens, elapsed

        except (anthropic.APIError, anthropic.APIConnectionError) as e:
            print(f"    Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def extract_csv_from_response(text, expected_headers=None):
    """Extract CSV content from a response that may contain markdown fences."""
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.startswith("csv"):
                candidate = candidate[3:].strip()
            if "," in candidate.split("\n")[0]:
                return candidate

    lines = text.strip().split("\n")
    csv_lines = []
    in_csv = False
    for line in lines:
        if not in_csv:
            if expected_headers and any(h in line for h in expected_headers[:2]):
                in_csv = True
                csv_lines.append(line)
            elif "," in line and not line.startswith("#") and not line.startswith("*"):
                in_csv = True
                csv_lines.append(line)
        else:
            if line.strip() == "" or line.startswith("#") or line.startswith("*"):
                break
            csv_lines.append(line)

    if csv_lines:
        return "\n".join(csv_lines)
    return text.strip()


def extract_json_from_response(text):
    """Extract a JSON object from a response that may contain markdown fences."""
    raw = text.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                raw = candidate
                break

    if raw.startswith("{"):
        depth = 0
        for i, ch in enumerate(raw):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = raw[: i + 1]
                    break

    return json.loads(raw)


def extract_markdown_from_response(text):
    """Extract markdown content, stripping any code fences."""
    if text.strip().startswith("```"):
        parts = text.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.startswith("markdown"):
                candidate = candidate[8:].strip()
            return candidate
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 4 helpers: code repository fetching and processing
# ---------------------------------------------------------------------------
def extract_code_repo_urls(pdf_text):
    """Extract GitHub/GitLab/Bitbucket repository URLs from paper text.

    Prioritizes URLs found near 'code availability' sections and deprioritizes
    URLs that appear in tool-reference contexts ('we used X', 'available from X').
    """
    url_patterns = [
        r'https?://github\.com/[\w\-\.]+/[\w\-\.]+',
        r'https?://gitlab\.com/[\w\-\.]+/[\w\-\.]+',
        r'https?://bitbucket\.org/[\w\-\.]+/[\w\-\.]+',
        r'https?://zenodo\.org/records?/\d+',
        r'https?://doi\.org/10\.5281/zenodo\.\d+',
        r'https?://figshare\.com/articles/[\w\-\.]+/\d+',
        r'https?://doi\.org/10\.6084/m9\.figshare\.\d+',
        r'https?://datadryad\.org/stash/dataset/doi:[\w\-\./:]+',
    ]

    # Collect all URLs with their surrounding context
    url_entries = []  # list of (url, context_before, context_after)
    for pattern in url_patterns:
        for match in re.finditer(pattern, pdf_text):
            url = match.group(0).rstrip('.,;:)]}\'\"')
            if url.endswith('.git'):
                url = url[:-4]
            start = max(0, match.start() - 200)
            end = min(len(pdf_text), match.end() + 200)
            context = pdf_text[start:end].lower()
            url_entries.append((url, context))

    # Deduplicate preserving first occurrence
    seen = set()
    unique_entries = []
    for url, context in url_entries:
        normalized = url.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_entries.append((url, context))

    # Score each URL: higher = more likely the paper's own code repo
    def score_url(url, context):
        s = 0
        # Strong positive: near "code availability" / "data and code" sections
        if re.search(r'code.*availab|data.*and.*code|code.*deposit|our.*code|'
                      r'analysis.*scripts?.*availab|code.*accessible', context):
            s += 10
        # Moderate positive: near "reproducib" or "pipeline"
        if re.search(r'reproduc|pipeline.*availab|workflow.*availab', context):
            s += 5
        # Negative: tool-reference context ("we used X", "available from X",
        # "implemented in X", "using the X package/tool/library")
        if re.search(r'we used|available from|implemented in|using the|'
                      r'package|library|tool|software|v\d+\.\d+', context):
            s -= 5
        # Negative: URL appears in references/citations section context
        if re.search(r'^\d+\.\s|et al\.|doi:|pmid:', context):
            s -= 3
        return s

    scored = [(score_url(url, ctx), url) for url, ctx in unique_entries]
    scored.sort(key=lambda x: -x[0])  # highest score first

    return [url for _, url in scored]


def parse_repo_url(repo_url):
    """Parse a GitHub/GitLab/Bitbucket URL into (host, owner, repo)."""
    parsed = urlparse(repo_url)
    path_parts = parsed.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError(f"Cannot parse owner/repo from URL: {repo_url}")
    return parsed.netloc, path_parts[0], path_parts[1]


def fetch_repo_tree(repo_url):
    """Fetch the file tree of a GitHub repo via the API (no download).

    Returns (branch, file_list) where file_list is a list of dicts with
    'path', 'size', and 'type' keys.
    """
    host, owner, repo = parse_repo_url(repo_url)

    if 'github.com' not in host:
        raise ValueError(f"Only GitHub repos supported for API access, got: {host}")

    headers = {"User-Agent": "ClaimPipeline/1.0", "Accept": "application/vnd.github.v3+json"}

    # Try common default branches
    for branch in ['main', 'master']:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        try:
            req = Request(api_url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            files = [
                {"path": item["path"], "size": item.get("size", 0), "type": item["type"]}
                for item in data.get("tree", [])
            ]
            return branch, files
        except HTTPError as e:
            if e.code == 404:
                continue
            raise

    raise RuntimeError(f"Could not fetch tree for {repo_url} (tried main, master)")


def fetch_file_content(repo_url, branch, file_path):
    """Fetch a single file's content from GitHub via raw.githubusercontent.com."""
    host, owner, repo = parse_repo_url(repo_url)
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    req = Request(raw_url, headers={"User-Agent": "ClaimPipeline/1.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def collect_code_files_remote(repo_url, branch, file_tree, max_tokens=60000):
    """Filter the repo file tree and fetch analysis-relevant files via API.

    Returns:
        code_files: list of {"path": relative_path, "content": text}
        metadata: dict with repo stats
    """
    # Filter to relevant files
    candidates = []
    for f in file_tree:
        if f["type"] != "blob":
            continue
        path = f["path"]
        size = f.get("size", 0)

        # Skip files in excluded directories
        path_parts = path.lower().split('/')
        if any(part in CODE_SKIP_DIRS for part in path_parts[:-1]):
            continue

        fname = path_parts[-1]
        ext = Path(path).suffix.lower()

        # Include if extension matches or it's a special file
        if ext not in CODE_EXTENSIONS and fname not in CODE_SPECIAL_FILES:
            continue

        # Skip large files (>500KB) and empty files
        if size > 500_000 or size == 0:
            continue

        candidates.append({"path": path, "size": size, "tokens": size // 4})

    total_files_in_repo = len(candidates)

    # Sort: README and env files first, then by size (smaller first)
    def sort_key(f):
        name = Path(f["path"]).name.lower()
        if name.startswith("readme"):
            return (0, f["size"])
        if name in CODE_SPECIAL_FILES:
            return (1, f["size"])
        return (2, f["size"])

    candidates.sort(key=sort_key)

    # Select files up to token budget, then fetch their content
    selected = []
    total_tokens = 0
    for f in candidates:
        if total_tokens + f["tokens"] > max_tokens:
            break
        selected.append(f)
        total_tokens += f["tokens"]

    # Fetch file contents
    code_files = []
    for f in selected:
        try:
            content = fetch_file_content(repo_url, branch, f["path"])
            actual_tokens = len(content) // 4
            code_files.append({"path": f["path"], "content": content})
        except Exception as e:
            print(f"      Skipping {f['path']}: {e}")

    actual_tokens = sum(len(f["content"]) // 4 for f in code_files)
    metadata = {
        "total_files_in_repo": total_files_in_repo,
        "files_included": len(code_files),
        "approx_tokens": actual_tokens,
    }
    return code_files, metadata


def download_and_collect_code_files(repo_url, max_tokens=60000):
    """Download a ZIP archive from Zenodo/Figshare/Dryad/Bitbucket, extract it,
    and collect analysis-relevant code files.

    Returns (code_files, metadata, branch_or_version) or raises on failure.
    """
    headers = {"User-Agent": "ClaimPipeline/1.0"}
    parsed = urlparse(repo_url)
    host = parsed.netloc.lower()

    # Resolve the download URL based on host
    zip_url = None
    version = "latest"

    if "zenodo.org" in host:
        # Zenodo: /records/12345 or /record/12345 -> API gives download links
        record_id = re.search(r'/records?/(\d+)', parsed.path)
        if not record_id:
            # DOI format: 10.5281/zenodo.12345
            record_id = re.search(r'zenodo\.(\d+)', repo_url)
        if record_id:
            api_url = f"https://zenodo.org/api/records/{record_id.group(1)}"
            req = Request(api_url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                record = json.loads(resp.read().decode())
            version = record.get("metadata", {}).get("version", "latest")
            # Find the first ZIP file in the record's files
            for f in record.get("files", []):
                if f["key"].endswith(".zip"):
                    zip_url = f["links"]["self"]
                    break
            # If no ZIP, try the first tar.gz
            if not zip_url:
                for f in record.get("files", []):
                    if f["key"].endswith((".tar.gz", ".tgz")):
                        zip_url = f["links"]["self"]
                        break

    elif "figshare.com" in host or "10.6084" in repo_url:
        # Figshare: try the download endpoint
        article_id = re.search(r'/(\d+)(?:\?|$|/)', repo_url)
        if article_id:
            api_url = f"https://api.figshare.com/v2/articles/{article_id.group(1)}"
            req = Request(api_url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                article = json.loads(resp.read().decode())
            version = str(article.get("version", "latest"))
            for f in article.get("files", []):
                if f["name"].endswith((".zip", ".tar.gz", ".tgz")):
                    zip_url = f["download_url"]
                    break

    elif "bitbucket.org" in host:
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2:
            owner, repo = path_parts[0], path_parts[1]
            for branch in ['main', 'master']:
                zip_url = f"https://bitbucket.org/{owner}/{repo}/get/{branch}.zip"
                try:
                    req = Request(zip_url, headers=headers, method="HEAD")
                    with urlopen(req, timeout=10):
                        version = branch
                        break
                except (HTTPError, URLError):
                    zip_url = None

    if not zip_url:
        raise RuntimeError(f"Could not resolve download URL for {repo_url}")

    # Download and extract to a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="claim_pipeline_"))
    try:
        zip_path = tmp_dir / "archive.zip"
        req = Request(zip_url, headers=headers)
        with urlopen(req, timeout=120) as resp, open(zip_path, 'wb') as f:
            f.write(resp.read())

        # Extract (handle both ZIP and tar.gz)
        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir()
        if zip_path.suffix == ".zip" or zip_url.endswith(".zip"):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        else:
            import tarfile
            with tarfile.open(zip_path) as tf:
                tf.extractall(extract_dir)

        # Find the root directory (archives often have a single top-level dir)
        contents = [d for d in extract_dir.iterdir() if d.is_dir()]
        repo_root = contents[0] if len(contents) == 1 else extract_dir

        # Walk and collect files (same logic as the remote version)
        candidates = []
        for root, dirs, files in os.walk(repo_root):
            root_path = Path(root)
            rel_root = root_path.relative_to(repo_root)
            dirs[:] = [d for d in dirs if d.lower() not in CODE_SKIP_DIRS]

            for fname in files:
                fpath = root_path / fname
                rel_path = str(rel_root / fname)
                ext = fpath.suffix.lower()
                fname_lower = fname.lower()

                if ext not in CODE_EXTENSIONS and fname_lower not in CODE_SPECIAL_FILES:
                    continue
                try:
                    size = fpath.stat().st_size
                    if size > 500_000 or size == 0:
                        continue
                except OSError:
                    continue
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                candidates.append({
                    "path": rel_path, "content": content,
                    "tokens": len(content) // 4, "size": size,
                })

        total_files = len(candidates)

        # Sort and budget
        def sort_key(f):
            name = Path(f["path"]).name.lower()
            if name.startswith("readme"):
                return (0, f["size"])
            if name in CODE_SPECIAL_FILES:
                return (1, f["size"])
            return (2, f["size"])

        candidates.sort(key=sort_key)
        code_files = []
        total_tokens = 0
        for f in candidates:
            if total_tokens + f["tokens"] > max_tokens:
                break
            code_files.append({"path": f["path"], "content": f["content"]})
            total_tokens += f["tokens"]

        metadata = {
            "total_files_in_repo": total_files,
            "files_included": len(code_files),
            "approx_tokens": total_tokens,
        }
        return code_files, metadata, version

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def fetch_code_files(repo_url):
    """Unified entry point: fetch code files via API (GitHub/GitLab) or
    download (Zenodo/Figshare/Bitbucket/other).

    Returns (code_files, metadata, branch_or_version).
    """
    parsed = urlparse(repo_url)
    host = parsed.netloc.lower()

    if "github.com" in host:
        branch, file_tree = fetch_repo_tree(repo_url)
        code_files, metadata = collect_code_files_remote(repo_url, branch, file_tree)
        return code_files, metadata, branch

    # All other hosts: download archive
    return download_and_collect_code_files(repo_url)


def build_code_digest(code_files):
    """Format code files into a structured text block for the Claude prompt."""
    parts = []
    for f in code_files:
        parts.append(f"===CODE_FILE: {f['path']}===")
        parts.append(f["content"])
        parts.append("===END_CODE_FILE===")
    return "\n\n".join(parts)


def verify_code_repo(client, pdf_text, pdf_name, candidate_urls):
    """Ask Claude whether candidate URLs are the paper's own analysis code.

    Returns the confirmed URL, or None if no candidate is the paper's code.
    Conservative: only returns a URL if Claude is confident it's the paper's
    own analysis repository, not a third-party tool or library.
    """
    urls_text = "\n".join(f"  {i+1}. {url}" for i, url in enumerate(candidate_urls))

    messages = [
        {
            "role": "user",
            "content": f"""I found these GitHub repository URLs in the paper "{pdf_name}":

{urls_text}

Paper title and abstract (first 2000 chars):
{pdf_text[:2000]}

For each URL, determine whether it is:
(A) The paper's OWN analysis/code repository — containing the scripts the
    authors wrote to produce the results in this specific paper.
(B) A third-party tool, library, or software package that the authors used
    but did not create for this paper.
(C) Uncertain / cannot determine.

Be CONSERVATIVE: only classify as (A) if you are confident. If in doubt,
classify as (B) or (C).

Respond with ONLY a JSON object in this format:
{{"confirmed_url": "<url>" or null, "reasoning": "<brief explanation>"}}

Set confirmed_url to the URL classified as (A), or null if none qualifies.""",
        }
    ]

    try:
        text, _, _, _ = call_claude(client, messages, max_retries=2)
        result = extract_json_from_response(text)
        confirmed = result.get("confirmed_url")
        reasoning = result.get("reasoning", "")
        if confirmed:
            print(f"    Verified: {confirmed}")
            print(f"    Reason: {reasoning}")
            return confirmed
        else:
            print(f"    No confirmed analysis repo: {reasoning}")
            return None
    except Exception as e:
        print(f"    Verification call failed: {e} — skipping code enhancement")
        return None


def _run_stage4_paper_only(client, pdf_text, pdf_name, paper_dir, t0):
    """Run only Pass 2 (paper-based improvement) when no code repo is available."""
    dag_path = paper_dir / "claim_dag.json"
    dag_text = read_text(dag_path)

    # Save backup
    backup_path = paper_dir / "claim_dag_pre_stage4.json"
    backup_path.write_text(dag_text, encoding="utf-8")

    dag = json.loads(dag_text)

    # Collect nodes that have analysis_detail
    unmatched_nodes = [
        n for n in dag.get("nodes", [])
        if n.get("analysis_detail") is not None
    ]

    if not unmatched_nodes:
        print("    No nodes with analysis_detail — nothing to improve")
        return {"tokens_in": 0, "tokens_out": 0, "elapsed": time.time() - t0,
                "detail": "skipped: no analysis_detail nodes"}

    print(f"    Paper-only mode: improving {len(unmatched_nodes)} nodes from paper text")

    nodes_for_improvement = [
        {
            "id": n["id"],
            "claim_text": n.get("claim_text", ""),
            "analysis_detail": n.get("analysis_detail", ""),
            "analysis_type": n.get("analysis_type", ""),
            "experiment_type": n.get("experiment_type", ""),
            "evidence_location": n.get("evidence_location", ""),
        }
        for n in unmatched_nodes
    ]

    pass2_prompt = STAGE4_PASS2_PROMPT
    nodes_json = json.dumps(nodes_for_improvement, indent=2, ensure_ascii=True)

    messages = [
        {
            "role": "user",
            "content": f"""Below is the full text of the paper "{pdf_name}".

{pdf_text}

---

{pass2_prompt}

Nodes to improve:
{nodes_json}

Return ONLY the JSON array. Do not wrap it in markdown code fences or include any other text.""",
        }
    ]

    text, tok_in, tok_out, elapsed_api = call_claude(client, messages)
    n_improved = 0

    try:
        raw = text.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts[1::2]:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("["):
                    raw = candidate
                    break
        improved_nodes = json.loads(raw)

        improved_lookup = {n["id"]: n for n in improved_nodes}
        dag_node_lookup = {n["id"]: n for n in dag.get("nodes", [])}

        for nid, improved in improved_lookup.items():
            if nid in dag_node_lookup:
                node = dag_node_lookup[nid]
                new_ad = improved.get("analysis_detail")
                if new_ad and len(new_ad) > len(node.get("analysis_detail", "") or ""):
                    node["analysis_detail"] = new_ad
                    node["analysis_detail_enhanced"] = False
                    n_improved += 1

        # Post-process: strip cross-refs and script names
        for node in dag.get("nodes", []):
            ad = node.get("analysis_detail")
            if not ad:
                continue
            ad = re.sub(r'[Ss]ee [LIR]+\d+\b[^.]*\.?\s*', '', ad)
            ad = re.sub(
                r'(?:the [A-Z]+ script |via |executed via |run |see )?'
                r'[\w/\-]+\.(?:R|py|sh|ipynb|Rmd|jl|nf|smk)\b'
                r'(?:,? which)?',
                '', ad
            )
            ad = re.sub(r'  +', ' ', ad)
            ad = re.sub(r'\(\s*\)', '', ad)
            ad = re.sub(r'\(\s*,', '(', ad)
            node["analysis_detail"] = ad.strip()

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"    Paper improvement parse failed: {e}")

    # Write updated DAG
    out_path = paper_dir / "claim_dag.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dag, f, indent=2, ensure_ascii=True)

    elapsed = time.time() - t0
    detail = f"paper-only: {n_improved}/{len(unmatched_nodes)} nodes improved (no code repo)"
    print(f"    → {detail}")
    return {"tokens_in": tok_in, "tokens_out": tok_out, "elapsed": elapsed,
            "detail": detail}


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def run_stage0(client, pdf_text, pdf_name, paper_dir):
    """Stage 0: Extract dataset links from the paper."""
    print("  Stage 0: Dataset link extraction")
    prompt_text = read_text(STAGE_PROMPTS[0])

    messages = [
        {
            "role": "user",
            "content": f"""Below is the full text of the paper "{pdf_name}".

{pdf_text}

---

{prompt_text}

Return ONLY the CSV output (with header row). Do not include any other text, explanation, or markdown fences.""",
        }
    ]

    text, tok_in, tok_out, elapsed = call_claude(client, messages)
    csv_text = extract_csv_from_response(text, expected_headers=["dataset_id", "dataset_type"])

    out_path = paper_dir / "dataset_links.csv"
    out_path.write_text(csv_text, encoding="utf-8")

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    print(f"    → {len(rows)} dataset links extracted")
    return {"tokens_in": tok_in, "tokens_out": tok_out, "elapsed": elapsed,
            "detail": f"{len(rows)} dataset links"}


def run_stage1(client, pdf_text, pdf_name, paper_dir):
    """Stage 1: Dataset identification — profile + section inventory."""
    print("  Stage 1: Dataset identification")
    prompt_text = read_text(STAGE_PROMPTS[1])
    dataset_links = read_text(paper_dir / "dataset_links.csv")

    messages = [
        {
            "role": "user",
            "content": f"""Below is the full text of the paper "{pdf_name}".

{pdf_text}

---

{prompt_text}

Dataset links from Stage 0:
{dataset_links}

You must produce TWO outputs. Separate them clearly with the exact delimiters shown below.

===DATASET_PROFILE_START===
(the full dataset profile in markdown)
===DATASET_PROFILE_END===

===SECTION_INVENTORY_START===
(the section inventory as CSV with header row, including a dataset_source column)
===SECTION_INVENTORY_END===

Do not include any other text outside these delimiters.""",
        }
    ]

    text, tok_in, tok_out, elapsed = call_claude(client, messages)

    if "===DATASET_PROFILE_START===" in text:
        profile = text.split("===DATASET_PROFILE_START===")[1].split("===DATASET_PROFILE_END===")[0].strip()
    else:
        parts = text.split("===SECTION_INVENTORY_START===")
        profile = parts[0].strip()

    if "===SECTION_INVENTORY_START===" in text:
        inventory = text.split("===SECTION_INVENTORY_START===")[1].split("===SECTION_INVENTORY_END===")[0].strip()
    else:
        inventory = ""

    profile = extract_markdown_from_response(profile)
    if inventory:
        inventory = extract_csv_from_response(inventory)

    (paper_dir / "dataset_profile.md").write_text(profile, encoding="utf-8")
    (paper_dir / "section_inventory.csv").write_text(inventory, encoding="utf-8")

    n_inv = len(inventory.splitlines()) - 1 if inventory else 0
    print(f"    → Profile: {len(profile)} chars, Inventory: {n_inv} rows")
    return {"tokens_in": tok_in, "tokens_out": tok_out, "elapsed": elapsed,
            "detail": f"profile {len(profile)} chars, {n_inv} inventory rows"}


def run_stage2(client, pdf_text, pdf_name, paper_dir):
    """Stage 2: Claim extraction."""
    print("  Stage 2: Claim extraction")
    prompt_text = read_text(STAGE_PROMPTS[2])
    profile = read_text(paper_dir / "dataset_profile.md")
    inventory = read_text(paper_dir / "section_inventory.csv")

    messages = [
        {
            "role": "user",
            "content": f"""Below is the full text of the paper "{pdf_name}".

{pdf_text}

---

{prompt_text}

Combined dataset profile from Stage 1:
{profile}

Section inventory from Stage 1:
{inventory}

You must produce TWO outputs. Separate them clearly with the exact delimiters shown below.

===CLAIMS_CSV_START===
(the claims CSV with header row)
===CLAIMS_CSV_END===

===KEY_TERMS_CSV_START===
(the key terms CSV with header row)
===KEY_TERMS_CSV_END===

Do not include any other text outside these delimiters.""",
        }
    ]

    text, tok_in, tok_out, elapsed = call_claude(client, messages)

    if "===CLAIMS_CSV_START===" in text:
        claims = text.split("===CLAIMS_CSV_START===")[1].split("===CLAIMS_CSV_END===")[0].strip()
    else:
        claims = extract_csv_from_response(text, expected_headers=["claim_id", "claim_text"])

    if "===KEY_TERMS_CSV_START===" in text:
        key_terms = text.split("===KEY_TERMS_CSV_START===")[1].split("===KEY_TERMS_CSV_END===")[0].strip()
    else:
        key_terms = ""

    claims = extract_csv_from_response(claims, expected_headers=["claim_id"])
    if key_terms:
        key_terms = extract_csv_from_response(key_terms)

    (paper_dir / "claims.csv").write_text(claims, encoding="utf-8")
    (paper_dir / "key_terms.csv").write_text(key_terms, encoding="utf-8")

    reader = csv.DictReader(io.StringIO(claims))
    rows = list(reader)
    print(f"    → {len(rows)} claims extracted")
    return {"tokens_in": tok_in, "tokens_out": tok_out, "elapsed": elapsed,
            "detail": f"{len(rows)} claims"}


def run_stage3(client, pdf_text, pdf_name, paper_dir):
    """Stage 3: DAG construction."""
    print("  Stage 3: DAG construction")
    prompt_text = read_text(STAGE_PROMPTS[3])
    profile = read_text(paper_dir / "dataset_profile.md")
    claims = read_text(paper_dir / "claims.csv")

    messages = [
        {
            "role": "user",
            "content": f"""Below is the full text of the paper "{pdf_name}".

{pdf_text}

---

{prompt_text}

Combined dataset profile from Stage 1:
{profile}

Claims from Stage 2:
{claims}

Return ONLY the JSON object. Do not wrap it in markdown code fences or include any other text.""",
        }
    ]

    text, tok_in, tok_out, elapsed = call_claude(client, messages)
    dag = extract_json_from_response(text)

    out_path = paper_dir / "claim_dag.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dag, f, indent=2, ensure_ascii=True)

    # Report
    n_nodes = len(dag.get("nodes", []))
    n_edges = len(dag.get("edges", []))
    node_types = {}
    for n in dag.get("nodes", []):
        nt = n.get("node_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1
    sources = {}
    for n in dag.get("nodes", []):
        s = n.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    depths = [n.get("depth", 0) for n in dag.get("nodes", [])]
    max_depth = max(depths) if depths else 0

    detail = f"{n_nodes} nodes, {n_edges} edges, depth {max_depth}"
    print(f"    → {detail}")
    print(f"      Node types: {node_types}")
    print(f"      Sources: {sources}")
    return {"tokens_in": tok_in, "tokens_out": tok_out, "elapsed": elapsed,
            "detail": detail}


def run_stage4(client, pdf_text, pdf_name, paper_dir, code_repo_url=None):
    """Stage 4: Enhance analysis_detail using the paper's code repository."""
    print("  Stage 4: Analysis protocol enhancement")

    # Step 1: Determine code repo URL
    t0 = time.time()

    if code_repo_url:
        repo_url = code_repo_url
        print(f"    Code repository (user-specified): {repo_url}")
    else:
        repo_urls = extract_code_repo_urls(pdf_text)
        if not repo_urls:
            print("    No code repository URLs found in paper")
            repo_url = None
        else:
            print(f"    Candidate URLs found: {', '.join(repo_urls)}")
            # Verify with Claude that the top candidate is the paper's own code
            repo_url = verify_code_repo(client, pdf_text, pdf_name, repo_urls)

        if not repo_url:
            print("    No confirmed analysis code repository — running paper-only enhancement")
            # Skip to pass 2 (paper-only improvement) without code enhancement
            return _run_stage4_paper_only(client, pdf_text, pdf_name, paper_dir, t0)

    # Step 2: Fetch code files (API for GitHub/GitLab, download for others)
    try:
        code_files, repo_meta, branch = fetch_code_files(repo_url)
        print(f"    Code files: {len(code_files)} fetched "
              f"(of {repo_meta['total_files_in_repo']} relevant), "
              f"~{repo_meta['approx_tokens']:,} tokens (branch/version: {branch})")
    except Exception as e:
        print(f"    Failed to fetch code files: {e}")
        print("    Falling back to paper-only enhancement")
        return _run_stage4_paper_only(client, pdf_text, pdf_name, paper_dir, t0)

    if not code_files:
        print("    No analysis-relevant code files found — paper-only enhancement")
        return _run_stage4_paper_only(client, pdf_text, pdf_name, paper_dir, t0)

    code_digest = build_code_digest(code_files)
    files_read = [f["path"] for f in code_files]

    # Step 4: Load existing claim_dag.json
    dag_path = paper_dir / "claim_dag.json"
    dag_text = read_text(dag_path)

    # Save a backup of the original DAG before enhancement
    backup_path = paper_dir / "claim_dag_pre_stage4.json"
    backup_path.write_text(dag_text, encoding="utf-8")

    # Step 5: Build and send Claude message
    prompt_text = read_text(STAGE_PROMPTS[4])
    today = datetime.now().strftime("%Y-%m-%d")

    messages = [
        {
            "role": "user",
            "content": f"""Below is the full text of the paper "{pdf_name}".

{pdf_text}

---

{prompt_text}

The claim DAG from Stage 3:
{dag_text}

The paper's code repository ({repo_url}, branch: {branch}, fetched: {today}):
{code_digest}

Return ONLY the JSON object. Do not wrap it in markdown code fences or include any other text.""",
        }
    ]

    text, tok_in, tok_out, elapsed_api = call_claude(client, messages)
    dag = extract_json_from_response(text)

    # Step 6: Post-processing validation
    original_dag = json.loads(dag_text)
    original_nodes = {n["id"]: n for n in original_dag.get("nodes", [])}
    immutable_fields = [
        "id", "claim_text", "claim_type", "node_type", "depth", "source",
        "datasets", "dataset_accessions", "experiment_type", "experiment_detail",
        "evidence_location", "supporting_quote", "dataset_link",
    ]
    warnings_count = 0
    for node in dag.get("nodes", []):
        nid = node.get("id")
        if nid not in original_nodes:
            continue
        orig = original_nodes[nid]

        # 6a: Restore any immutable fields that Claude changed
        for field in immutable_fields:
            if field in orig and node.get(field) != orig[field]:
                print(f"    WARNING: Stage 4 modified immutable field '{field}' on {nid} — restoring")
                node[field] = orig[field]
                warnings_count += 1

        # 6b: For unmatched nodes, restore original analysis_detail exactly
        if node.get("analysis_detail_enhanced") is False:
            orig_ad = orig.get("analysis_detail")
            if node.get("analysis_detail") != orig_ad:
                node["analysis_detail"] = orig_ad
                warnings_count += 1

        # 6c: Strip script/file name references from enhanced analysis_detail
        ad = node.get("analysis_detail")
        if ad and node.get("analysis_detail_enhanced") is True:
            # Remove patterns like "the R script DGE_analysis.R" or
            # "via DGE_analysis.R" or "(DGE_analysis.R)"
            ad = re.sub(
                r'(?:the [A-Z]+ script |via |executed via |run |see )?'
                r'[\w/\-]+\.(?:R|py|sh|ipynb|Rmd|jl|nf|smk)\b'
                r'(?:,? which)?',
                '', ad
            )
            # Clean up resulting double spaces, leading/trailing commas
            ad = re.sub(r'  +', ' ', ad)
            ad = re.sub(r'\(\s*\)', '', ad)
            ad = re.sub(r'\(\s*,', '(', ad)
            ad = ad.strip()
            node["analysis_detail"] = ad

    # Step 7: Second pass — improve unmatched analysis_detail from paper text
    unmatched_nodes = [
        n for n in dag.get("nodes", [])
        if n.get("analysis_detail") is not None
        and n.get("analysis_detail_enhanced") is not True
    ]

    tok_in_2, tok_out_2, elapsed_api_2 = 0, 0, 0
    n_paper_improved = 0

    if unmatched_nodes:
        print(f"    Pass 2: Improving {len(unmatched_nodes)} unmatched nodes from paper text")

        # Build a compact JSON of just the nodes that need improvement
        nodes_for_improvement = []
        for n in unmatched_nodes:
            nodes_for_improvement.append({
                "id": n["id"],
                "claim_text": n.get("claim_text", ""),
                "analysis_detail": n.get("analysis_detail", ""),
                "analysis_type": n.get("analysis_type", ""),
                "experiment_type": n.get("experiment_type", ""),
                "evidence_location": n.get("evidence_location", ""),
            })

        pass2_prompt = STAGE4_PASS2_PROMPT
        nodes_json = json.dumps(nodes_for_improvement, indent=2, ensure_ascii=True)

        messages_2 = [
            {
                "role": "user",
                "content": f"""Below is the full text of the paper "{pdf_name}".

{pdf_text}

---

{pass2_prompt}

Nodes to improve:
{nodes_json}

Return ONLY the JSON array. Do not wrap it in markdown code fences or include any other text.""",
            }
        ]

        text_2, tok_in_2, tok_out_2, elapsed_api_2 = call_claude(client, messages_2)

        # Parse the response — expect a JSON array of updated nodes
        try:
            raw_2 = text_2.strip()
            if "```" in raw_2:
                parts = raw_2.split("```")
                for part in parts[1::2]:
                    candidate = part.strip()
                    if candidate.startswith("json"):
                        candidate = candidate[4:].strip()
                    if candidate.startswith("["):
                        raw_2 = candidate
                        break
            improved_nodes = json.loads(raw_2)

            # Build lookup and apply improvements
            improved_lookup = {n["id"]: n for n in improved_nodes}
            dag_node_lookup = {n["id"]: n for n in dag.get("nodes", [])}

            for nid, improved in improved_lookup.items():
                if nid in dag_node_lookup:
                    node = dag_node_lookup[nid]
                    new_ad = improved.get("analysis_detail")
                    if new_ad and len(new_ad) > len(node.get("analysis_detail", "") or ""):
                        node["analysis_detail"] = new_ad
                        node["analysis_detail_enhanced"] = False  # still not code-enhanced
                        n_paper_improved += 1

            # Post-process pass 2: strip any cross-refs or script names
            for node in dag.get("nodes", []):
                ad = node.get("analysis_detail")
                if not ad:
                    continue
                # Strip cross-references
                ad = re.sub(r'[Ss]ee [LIR]+\d+\b[^.]*\.?\s*', '', ad)
                # Strip script names
                ad = re.sub(
                    r'(?:the [A-Z]+ script |via |executed via |run |see )?'
                    r'[\w/\-]+\.(?:R|py|sh|ipynb|Rmd|jl|nf|smk)\b'
                    r'(?:,? which)?',
                    '', ad
                )
                ad = re.sub(r'  +', ' ', ad)
                ad = re.sub(r'\(\s*\)', '', ad)
                ad = re.sub(r'\(\s*,', '(', ad)
                node["analysis_detail"] = ad.strip()

            print(f"    Pass 2: {n_paper_improved}/{len(unmatched_nodes)} nodes improved")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"    Pass 2 parse failed: {e} — keeping original analysis_detail")

    # Step 8: Write updated claim_dag.json
    out_path = paper_dir / "claim_dag.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dag, f, indent=2, ensure_ascii=True)

    # Step 9: Report
    elapsed = time.time() - t0
    n_enhanced = sum(1 for n in dag.get("nodes", [])
                     if n.get("analysis_detail_enhanced") is True)
    n_with_detail = sum(1 for n in dag.get("nodes", [])
                        if n.get("analysis_detail") is not None)

    total_tok_in = tok_in + tok_in_2
    total_tok_out = tok_out + tok_out_2

    detail = (f"{n_enhanced}/{n_with_detail} code-enhanced, "
              f"{n_paper_improved} paper-improved, from {len(code_files)} code files")
    if warnings_count:
        detail += f" ({warnings_count} field corrections)"
    print(f"    → {detail}")
    return {"tokens_in": total_tok_in, "tokens_out": total_tok_out, "elapsed": elapsed,
            "detail": detail}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STAGES = [
    (0, run_stage0),
    (1, run_stage1),
    (2, run_stage2),
    (3, run_stage3),
    (4, run_stage4),
]


def run_paper(client, paper_dir, from_stage=0, tracker=None, code_repo_url=None):
    """Run the full pipeline for a single paper directory."""
    paper_dir = Path(paper_dir).resolve()
    name = paper_dir.name
    pdf_path = find_pdf(paper_dir)
    pdf_text = extract_pdf_text(pdf_path)

    print(f"\n[{name}] PDF: {pdf_path.name} "
          f"({pdf_path.stat().st_size / 1024 / 1024:.1f} MB, "
          f"~{len(pdf_text) // 4:,} tokens extracted)")

    t_total = time.time()
    for stage_num, stage_fn in STAGES:
        if stage_num < from_stage:
            continue

        # Verify prerequisites
        prereqs = {
            1: ["dataset_links.csv"],
            2: ["dataset_profile.md", "section_inventory.csv"],
            3: ["claims.csv"],
            4: ["claim_dag.json"],
        }
        missing = [f for f in prereqs.get(stage_num, []) if not (paper_dir / f).exists()]
        if missing:
            print(f"  ERROR: Missing prerequisites for Stage {stage_num}: {', '.join(missing)}")
            if tracker:
                tracker.stage_failed(name, stage_num, f"Missing: {', '.join(missing)}")
                tracker.paper_failed(name)
            return False

        stage_succeeded = False
        for attempt in range(1, MAX_STAGE_RETRIES + 1):
            if tracker:
                tracker.stage_start(name, stage_num)

            try:
                if stage_num == 4:
                    result = stage_fn(client, pdf_text, pdf_path.name, paper_dir,
                                      code_repo_url=code_repo_url)
                else:
                    result = stage_fn(client, pdf_text, pdf_path.name, paper_dir)
                if tracker and result:
                    tracker.stage_done(name, stage_num, **result)
                stage_succeeded = True
                break
            except Exception as e:
                print(f"  Stage {stage_num} attempt {attempt}/{MAX_STAGE_RETRIES} failed: {e}")
                if attempt < MAX_STAGE_RETRIES:
                    wait = 15 * attempt
                    print(f"    Retrying stage in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  FAILED: Stage {stage_num} after {MAX_STAGE_RETRIES} attempts")
                    if tracker:
                        tracker.stage_failed(name, stage_num, str(e))
                        tracker.paper_failed(name)

        if not stage_succeeded:
            return False

    elapsed = time.time() - t_total
    print(f"[{name}] Complete ({elapsed:.1f}s total)")
    if tracker:
        tracker.paper_done(name, elapsed)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the claim extraction pipeline via the Anthropic API (temperature=0)."
    )
    parser.add_argument("paper_dirs", nargs="+",
                        help="Paper directories under Papers_claims/, each containing a PDF")
    parser.add_argument("--from-stage", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Resume from this stage (prior outputs must exist)")
    parser.add_argument("--max-parallel", type=int, default=5,
                        help="Max papers to process in parallel (default: 5)")
    parser.add_argument("--code-repo", type=str, default=None,
                        help="Code repository URL for Stage 4 (overrides auto-detection)")
    args = parser.parse_args()

    client = anthropic.Anthropic()

    # Resolve all paper paths
    paper_paths = []
    for paper_dir in args.paper_dirs:
        paper_path = Path(paper_dir)
        if not paper_path.is_absolute():
            paper_path = PROJECT_ROOT / paper_path
        paper_path = paper_path.resolve()
        if not paper_path.is_dir():
            print(f"WARNING: Not a directory: {paper_path} — skipping")
            continue
        paper_paths.append(paper_path)

    if not paper_paths:
        print("ERROR: No valid paper directories provided.")
        sys.exit(1)

    n_workers = min(args.max_parallel, len(paper_paths))
    print("=" * 60)
    print("Claim Extraction Pipeline (Anthropic API, temperature=0)")
    print(f"Model: {MODEL}")
    print(f"Papers: {len(paper_paths)}")
    print(f"Parallel workers: {n_workers}")
    if args.from_stage > 0:
        print(f"Resuming from: Stage {args.from_stage}")
    print(f"Status file: {STATUS_FILE}")
    print("=" * 60)

    # Initialize status tracker
    tracker = StatusTracker(
        [pp.name for pp in paper_paths],
        from_stage=args.from_stage,
    )

    # Run papers in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    t_all = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_paper, client, pp, args.from_stage, tracker,
                            args.code_repo): pp.name
            for pp in paper_paths
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                success = future.result()
                results[name] = success
            except Exception as e:
                print(f"\n[{name}] EXCEPTION: {e}")
                results[name] = False
                tracker.paper_failed(name)

    tracker.finish()

    # Summary
    elapsed = time.time() - t_all
    ok = sum(1 for v in results.values() if v)
    print("\n" + "=" * 60)
    print(f"SUMMARY ({elapsed:.0f}s total)")
    print("=" * 60)
    for name in sorted(results):
        status = "OK" if results[name] else "FAILED"
        print(f"  {name}: {status}")
    print(f"\n  {ok}/{len(results)} papers completed successfully")
    print(f"  Status details: {STATUS_FILE}")


if __name__ == "__main__":
    main()
