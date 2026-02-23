"""
Repository content processor.

Builds a compact "context pack" from GitHub data to send to the LLM.
Implements the file selection, filtering, and truncation strategy.
Also handles notebook import extraction and technology evidence building.
"""

from __future__ import annotations

import json
import logging
import re

from app.settings import settings

logger = logging.getLogger("app.summarizer")


# ── Files to skip entirely ──────────────────────────────────────
_SKIP_DIRS = {
    "node_modules", "dist", "build", ".venv", "venv", "env",
    "__pycache__", ".git", ".idea", ".vscode", ".tox",
    ".mypy_cache", ".pytest_cache", ".eggs", "vendor",
    ".next", ".nuxt", "target", "out", "bin", "obj",
    "coverage", ".coverage", "htmlcov",
}

_SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".dylib",
    ".exe", ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".min.js", ".min.css", ".map",
    ".lock",  # catches package-lock.json, yarn.lock, etc.
    # Dataset / serialised model files
    ".csv", ".tsv", ".parquet", ".h5", ".hdf5",
    ".npy", ".npz", ".feather", ".arrow",
    ".pkl", ".pickle", ".joblib",
}

_SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "composer.lock",
    "Gemfile.lock", "Cargo.lock", "go.sum",
    ".DS_Store", "Thumbs.db",
}

# ── Config files worth fetching content for ─────────────────────
PRIORITY_FILES = [
    "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg",
    "package.json", "go.mod", "Cargo.toml", "pom.xml", "build.gradle",
    "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "compose.yml", "compose.yaml",
    ".github/workflows",  # presence marker
]

# Maximum number of key files to fetch content for
MAX_KEY_FILES = 6

# ── Stdlib / noise imports to ignore from notebook extraction ───
_STDLIB_NOISE = {
    "os", "sys", "re", "json", "math", "time", "datetime", "typing",
    "pathlib", "collections", "itertools", "random", "functools",
    "logging", "copy", "io", "string", "hashlib", "csv", "glob",
    "shutil", "subprocess", "argparse", "abc", "enum", "struct",
    "warnings", "gc", "operator", "contextlib", "textwrap",
    "unittest", "pprint", "traceback", "inspect", "codecs",
    "tempfile", "socket", "threading", "multiprocessing", "asyncio",
    "IPython", "ipywidgets", "ipykernel",
}

# ── Import normalization → canonical tech name ──────────────────
IMPORT_NORMALIZATION: dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "dash": "dash",
    "plotly": "plotly",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pandas": "pandas",
    "numpy": "numpy",
    "flask": "flask",
    "django": "django",
    "fastapi": "fastapi",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "keras": "keras",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "scipy": "scipy",
    "sqlalchemy": "sqlalchemy",
    "requests": "requests",
    "httpx": "httpx",
    "pydantic": "pydantic",
    "boto3": "boto3",
    "celery": "celery",
    "redis": "redis",
    "pymongo": "pymongo",
    "psycopg2": "psycopg2",
    "scrapy": "scrapy",
    "nltk": "nltk",
    "spacy": "spacy",
    "transformers": "transformers",
    "huggingface_hub": "huggingface-hub",
    "streamlit": "streamlit",
    "gradio": "gradio",
    "networkx": "networkx",
    "sympy": "sympy",
    "statsmodels": "statsmodels",
    "pyarrow": "pyarrow",
    "polars": "polars",
    "dask": "dask",
    "ray": "ray",
    "wandb": "wandb",
    "mlflow": "mlflow",
    "pytest": "pytest",
    "tqdm": "tqdm",
    "click": "click",
    "typer": "typer",
    "rich": "rich",
    "docker": "docker",
    "gunicorn": "gunicorn",
    "uvicorn": "uvicorn",
    "openai": "openai",
}

# ── Banned concept/algorithm words (not technologies) ───────────
_BANNED_PATTERNS = re.compile(
    r"(?i)\b("
    r"logistic.?regression|linear.?regression|decision.?tree|random.?forest|"
    r"gradient.?boost|xgboost.?model|support.?vector|naive.?bayes|"
    r"k.?nearest|knn|svm|neural.?network|deep.?learning|"
    r"classification|clustering|regression|train.?test.?split|"
    r"cross.?validation|feature.?engineering|hyperparameter|"
    r"accuracy|precision|recall|f1.?score|auc|roc|"
    r"overfitting|underfitting|ensemble|bagging|boosting|"
    r"backpropagation|convolution|recurrent|attention.?mechanism|"
    r"tokenization|embedding|fine.?tuning|transfer.?learning"
    r")\b"
)

# Regex for extracting imports from notebook code cells
_IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
_FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import", re.MULTILINE
)


def filter_tree(tree: list[dict]) -> list[dict]:
    """Remove irrelevant entries from the tree listing."""
    filtered = []
    for node in tree:
        path: str = node.get("path", "")
        ntype: str = node.get("type", "")

        # Skip known junk directories
        parts = path.split("/")
        if any(p.lower() in _SKIP_DIRS for p in parts):
            continue

        # Skip known junk files
        base = parts[-1]
        if base in _SKIP_FILES:
            continue

        # Skip by extension
        if ntype == "blob":
            lower = base.lower()
            if any(lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
                continue

        filtered.append(node)

    return filtered


def select_key_files(tree: list[dict]) -> list[str]:
    """Pick the most informative files from the tree to fetch content for."""
    available = {node["path"] for node in tree if node.get("type") == "blob"}
    selected: list[str] = []

    for pf in PRIORITY_FILES:
        if pf in available and len(selected) < MAX_KEY_FILES:
            selected.append(pf)

    return selected


def find_notebooks(tree: list[dict]) -> list[str]:
    """Find .ipynb files in the tree (top-level or one directory deep)."""
    notebooks = []
    for node in tree:
        if node.get("type") == "blob" and node.get("path", "").endswith(".ipynb"):
            # Cap to 3 notebooks to avoid excessive fetching
            notebooks.append(node["path"])
            if len(notebooks) >= 3:
                break
    return notebooks


def extract_notebook_imports(notebook_json: str) -> set[str]:
    """Parse a .ipynb JSON and extract top-level package names from imports.

    Returns normalized package names, excluding stdlib/noise.
    """
    packages: set[str] = set()

    try:
        nb = json.loads(notebook_json)
    except (json.JSONDecodeError, TypeError):
        return packages

    cells = nb.get("cells", [])
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, list):
            code = "".join(source)
        else:
            code = str(source)

        # Extract import statements
        for match in _IMPORT_RE.findall(code):
            top_pkg = match.split(".")[0]
            if top_pkg and top_pkg not in _STDLIB_NOISE:
                packages.add(top_pkg)

        for match in _FROM_IMPORT_RE.findall(code):
            top_pkg = match.split(".")[0]
            if top_pkg and top_pkg not in _STDLIB_NOISE:
                packages.add(top_pkg)

    # Cap to prevent noise
    return set(list(packages)[:25])


def normalize_import(pkg: str) -> str:
    """Convert an import name to its canonical technology name."""
    return IMPORT_NORMALIZATION.get(pkg, pkg)


def build_evidence_whitelist(
    languages: dict,
    file_contents: dict[str, str],
    notebook_imports: set[str],
) -> set[str]:
    """Build the set of allowed technology names from evidence.

    Sources:
    - GitHub languages API (language names)
    - Dependency file contents (requirements.txt, package.json, etc.)
    - Notebook imports (normalized)
    """
    allowed: set[str] = set()

    # 1. Language names from GitHub API
    for lang in languages:
        allowed.add(lang.lower())

    # 2. Notebook imports (normalized)
    for pkg in notebook_imports:
        allowed.add(normalize_import(pkg).lower())

    # 3. Dependencies from config files
    for path, content in file_contents.items():
        basename = path.split("/")[-1].lower()
        if basename in ("requirements.txt", "setup.cfg"):
            _extract_requirements(content, allowed)
        elif basename in ("pyproject.toml",):
            _extract_pyproject(content, allowed)
        elif basename == "package.json":
            _extract_package_json(content, allowed)
        elif basename in ("setup.py",):
            _extract_setup_py(content, allowed)

    # Add some always-allowed structural technologies
    _add_structural_evidence(file_contents, allowed)

    return allowed


def _extract_requirements(content: str, allowed: set[str]) -> None:
    """Extract package names from requirements.txt-style content."""
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip version specifiers and extras
        pkg = re.split(r"[>=<!\[;]", line)[0].strip()
        if pkg:
            allowed.add(pkg.lower())


def _extract_pyproject(content: str, allowed: set[str]) -> None:
    """Extract dependency names from pyproject.toml (best-effort regex)."""
    # Match lines like: "package>=1.0" or 'package'
    dep_re = re.compile(r'["\']([a-zA-Z0-9_-]+)', re.MULTILINE)
    for match in dep_re.findall(content):
        allowed.add(match.lower())


def _extract_package_json(content: str, allowed: set[str]) -> None:
    """Extract dependency names from package.json."""
    try:
        pkg = json.loads(content)
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            deps = pkg.get(key, {})
            if isinstance(deps, dict):
                for name in deps:
                    allowed.add(name.lower())
    except (json.JSONDecodeError, TypeError):
        pass


def _extract_setup_py(content: str, allowed: set[str]) -> None:
    """Extract package names from setup.py install_requires (best-effort)."""
    # Match strings inside install_requires=[...] or similar
    dep_re = re.compile(r'["\']([a-zA-Z0-9_-]+)', re.MULTILINE)
    for match in dep_re.findall(content):
        allowed.add(match.lower())


def _add_structural_evidence(file_contents: dict[str, str], allowed: set[str]) -> None:
    """Add technology names evidenced by file presence."""
    for path in file_contents:
        basename = path.split("/")[-1].lower()
        if basename == "dockerfile":
            allowed.add("docker")
        elif basename in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
            allowed.add("docker")
            allowed.add("docker-compose")
        elif basename == "makefile":
            allowed.add("make")


def filter_technologies(
    technologies: list[str],
    evidence: set[str],
    languages: dict,
) -> list[str]:
    """Post-filter LLM technologies against evidence whitelist.

    Removes hallucinated items (algorithms, concepts, un-evidenced tech).
    """
    filtered = []
    seen: set[str] = set()

    for tech in technologies:
        tech_clean = tech.strip()
        if not tech_clean:
            continue

        low = tech_clean.lower()

        # Skip duplicates
        if low in seen:
            continue

        # Ban algorithm/concept names
        if _BANNED_PATTERNS.search(tech_clean):
            logger.debug("Filtered out banned concept: %s", tech_clean)
            continue

        # Multi-word items that look like concepts (heuristic)
        if " " in tech_clean and low not in evidence:
            # Allow known multi-word tech like "Jupyter Notebook", "Google Cloud"
            # But block unknown multi-word items
            logger.debug("Filtered out unknown multi-word tech: %s", tech_clean)
            continue

        # Check against evidence whitelist
        if low in evidence:
            seen.add(low)
            filtered.append(tech_clean)
            continue

        logger.debug("Filtered out unevidenced tech: %s", tech_clean)

    # If we filtered everything, fall back to languages-only
    if not filtered:
        filtered = list(languages.keys())[:5]

    # Cap at 12 items
    return filtered[:12]


def build_context(
    *,
    owner: str,
    repo: str,
    repo_data: dict,
    languages: dict,
    readme: str | None,
    tree: list[dict],
    file_contents: dict[str, str],
    notebook_imports: set[str] | None = None,
) -> str:
    """Assemble the LLM context from all fetched data.

    Sections are appended in priority order and accumulation stops
    when ``settings.max_context_chars`` is reached.
    """
    budget = settings.max_context_chars
    parts: list[str] = []
    used = 0

    def _append(section: str, label: str) -> bool:
        nonlocal used
        cost = len(section) + 2  # account for "\n\n" separator
        if used + cost > budget:
            logger.warning(
                "Context budget reached (%d/%d chars) — skipping %s",
                used, budget, label,
            )
            return False
        parts.append(section)
        used += cost
        return True

    # ── 1. Repository metadata (always included) ────────────
    desc = repo_data.get("description") or "No description provided."
    topics = repo_data.get("topics", [])
    _append(
        f"## Repository: {owner}/{repo}\n"
        f"Description: {desc}\n"
        f"Topics: {', '.join(topics) if topics else 'none'}",
        "metadata",
    )

    # ── 2. Languages ────────────────────────────────────────
    if languages:
        total = sum(languages.values())
        lang_parts = []
        for lang, size in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = round(size / total * 100, 1) if total else 0
            lang_parts.append(f"{lang} ({pct}%)")
        _append(f"## LANGUAGES_BYTES\n{', '.join(lang_parts)}", "languages")

    # ── 3. Directory structure ──────────────────────────────
    filtered = filter_tree(tree)
    if filtered:
        tree_lines = []
        for node in filtered[:settings.tree_max_entries]:
            prefix = "📁 " if node.get("type") == "tree" else "📄 "
            size_info = ""
            if node.get("size"):
                size_info = f" ({node['size']} bytes)"
            tree_lines.append(f"{prefix}{node['path']}{size_info}")
        _append(f"## Directory Structure\n" + "\n".join(tree_lines), "tree")

    # ── 4. README ───────────────────────────────────────────
    if readme and readme.strip():
        readme_text = readme[: settings.readme_max_chars]
        _append(f"## README Content\n{readme_text}", "README")
    else:
        _append("## README Content\nNo README found.", "README")

    # ── 5. Key file contents (lower priority) ──────────────
    if file_contents:
        file_sections = []
        for path, content in file_contents.items():
            truncated = content[: settings.file_max_chars]
            file_sections.append(f"### {path}\n```\n{truncated}\n```")
        _append("## DEPENDENCY_SNIPPETS\n" + "\n\n".join(file_sections), "dep-snippets")

    # ── 6. Notebook imports (evidence section) ──────────────
    if notebook_imports:
        normalized = sorted({normalize_import(pkg) for pkg in notebook_imports})
        _append(
            f"## NOTEBOOK_IMPORTS\n"
            f"The following libraries are imported in the project's Jupyter notebooks:\n"
            f"{', '.join(normalized)}",
            "notebook-imports",
        )

    return "\n\n".join(parts)

