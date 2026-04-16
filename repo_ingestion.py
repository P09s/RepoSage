import os
import shutil
import tempfile
import ast
from pathlib import Path
from git import Repo

# Extensions we care about, mapped to language label
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".env.example": "text",
    ".sh": "shell",
}

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
    ".pytest_cache", ".mypy_cache", "eggs", "*.egg-info"
}

MAX_FILE_SIZE_KB = 150  # skip files larger than this


def clone_repo(github_url: str, github_token: str = None) -> str:
    """Clone a GitHub repo into a temp directory. Returns the local path."""
    tmp_dir = tempfile.mkdtemp(prefix="reposage_")

    if github_token:
        # Inject token into URL for private repos
        url = github_url.replace("https://", f"https://{github_token}@")
    else:
        url = github_url

    try:
        Repo.clone_from(url, tmp_dir, depth=1)  # shallow clone for speed
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(f"Failed to clone repo: {str(e)}")

    return tmp_dir


def _should_skip(path: Path) -> bool:
    """Return True if this file should be ignored."""
    for part in path.parts:
        if part in IGNORE_DIRS or part.endswith(".egg-info"):
            return True
    if path.stat().st_size > MAX_FILE_SIZE_KB * 1024:
        return True
    return False


def _chunk_by_lines(content: str, filepath: str, language: str,
                    chunk_size: int = 60, overlap: int = 10) -> list[dict]:
    """
    Fallback: split file into overlapping line-based chunks.
    Used for non-code files (md, yaml, json, etc.)
    """
    lines = content.splitlines()
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_text = "\n".join(lines[start:end])
        if chunk_text.strip():
            chunks.append({
                "content": chunk_text,
                "filepath": filepath,
                "language": language,
                "chunk_type": "lines",
                "start_line": start + 1,
                "end_line": end,
            })
        start += chunk_size - overlap
    return chunks


def _chunk_python_by_functions(content: str, filepath: str) -> list[dict]:
    """
    Parse Python file and extract top-level functions and classes as chunks.
    Falls back to line chunking if parsing fails.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_by_lines(content, filepath, "python")

    lines = content.splitlines()
    chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Only top-level and first-level class methods
            start = node.lineno - 1
            end = node.end_lineno
            chunk_text = "\n".join(lines[start:end])
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "filepath": filepath,
                    "language": "python",
                    "chunk_type": type(node).__name__,
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                })

    # If no functions/classes found, fall back to line chunking
    if not chunks:
        return _chunk_by_lines(content, filepath, "python")

    return chunks


def parse_repo_to_chunks(repo_path: str) -> list[dict]:
    """
    Walk the repo, read files, and return a flat list of chunk dicts.
    Each chunk has: content, filepath (relative), language, chunk_type, metadata.
    """
    root = Path(repo_path)
    all_chunks = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if _should_skip(file_path):
            continue

        suffix = file_path.suffix.lower()
        language = SUPPORTED_EXTENSIONS.get(suffix)
        if not language:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if not content.strip():
            continue

        relative_path = str(file_path.relative_to(root))

        if language == "python":
            chunks = _chunk_python_by_functions(content, relative_path)
        else:
            chunks = _chunk_by_lines(content, relative_path, language)

        all_chunks.extend(chunks)

    return all_chunks


def cleanup_repo(repo_path: str):
    """Delete cloned repo from disk."""
    shutil.rmtree(repo_path, ignore_errors=True)