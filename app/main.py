"""
FastAPI application — GitHub Repository Summarizer API.

Endpoints:
    GET  /health      → {"status": "ok"}
    POST /summarize   → SummarizeResponse
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from app.cache import TTLCache
from app.github_client import (
    GitHubClient,
    RateLimitError,
    RepoNotFoundError,
    UpstreamError,
)
from app.llm_client import LLMClient
from app.logging_config import new_request_id, request_id_ctx, setup_logging
from app.models import ErrorResponse, SummarizeRequest, SummarizeResponse
from app.settings import settings
from app.summarizer import (
    build_context,
    build_evidence_whitelist,
    extract_notebook_imports,
    filter_technologies,
    find_notebooks,
    select_key_files,
)
from app.url_parser import parse_github_url

logger = logging.getLogger("app.main")


# ── Shared state ───────────────────────────────────────────────
class _State:
    """Mutable container so lifespan and endpoints share instances."""
    github_client: GitHubClient | None = None
    llm_client: LLMClient | None = None
    cache: TTLCache | None = None


state = _State()


def _ensure_state() -> tuple[GitHubClient, LLMClient, TTLCache]:
    """Lazily initialise clients + cache for TestClient compatibility."""
    if state.github_client is None:
        state.github_client = GitHubClient()
    if state.llm_client is None:
        state.llm_client = LLMClient()
    if state.cache is None:
        state.cache = TTLCache(default_ttl=settings.cache_ttl_seconds)
    return state.github_client, state.llm_client, state.cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage client and cache lifetime."""
    setup_logging(settings.log_level)

    state.cache = TTLCache(default_ttl=settings.cache_ttl_seconds)
    state.github_client = GitHubClient()
    state.llm_client = LLMClient()

    logger.info(
        "Application started (github_token=%s, nebius_model=%s)",
        bool(settings.github_token),
        settings.nebius_model,
    )
    if not settings.nebius_api_key:
        logger.warning(
            "NEBIUS_API_KEY is not set — LLM calls will fail. "
            "Set the env var or add it to .env before sending requests."
        )
    yield

    if state.github_client:
        await state.github_client.aclose()
    if state.llm_client:
        await state.llm_client.aclose()
    logger.info("Application shutdown")


app = FastAPI(
    title="GitHub Repository Summarizer",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request_id + timing ────────────────────────────
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = new_request_id()
    request_id_ctx.set(rid)
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-Id"] = rid
    logger.info(
        "%s %s → %s (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


# ── Error helper ───────────────────────────────────────────────
def _error_response(status: int, message: str) -> JSONResponse:
    """Return error matching spec: {"status": "error", "message": "..."}"""
    body = ErrorResponse(message=message)
    return JSONResponse(status_code=status, content=body.model_dump())


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the UI at the root path."""
    import os
    ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    try:
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>GitHub Repo Summarizer API</h1><p>UI file not found at /frontend/index.html. Use POST /summarize to get started.</p>"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(body: SummarizeRequest):
    gh_client, llm_client, cache = _ensure_state()

    # 1. Parse URL
    try:
        owner, repo = parse_github_url(body.github_url)
    except ValueError as exc:
        return _error_response(400, str(exc))

    # 2. Check cache
    cached = await cache.resolve_latest(owner, repo)
    if cached is not None:
        logger.info("Cache HIT for %s/%s", owner, repo)
        return SummarizeResponse(**cached)

    # 3. Fetch from GitHub
    try:
        gh_data = await gh_client.fetch_all(owner, repo)
    except RepoNotFoundError as exc:
        return _error_response(404, str(exc))
    except RateLimitError as exc:
        return _error_response(429, str(exc))
    except UpstreamError as exc:
        return _error_response(502, str(exc))
    except Exception as exc:
        logger.exception("Unexpected error fetching GitHub data")
        return _error_response(502, f"Failed to fetch repository data: {exc}")

    # 4. Select key files and fetch their content
    key_files = select_key_files(gh_data["tree"])
    if key_files:
        try:
            for fpath in key_files:
                content = await gh_client.fetch_file_content(owner, repo, fpath)
                if content:
                    gh_data.setdefault("file_contents", {})[fpath] = content
        except Exception as exc:
            logger.warning("Failed to fetch some key files: %s", exc)

    # 5. Extract notebook imports (evidence for technologies)
    notebook_imports: set[str] = set()
    notebook_paths = find_notebooks(gh_data["tree"])
    if notebook_paths:
        for nb_path in notebook_paths:
            try:
                nb_content = await gh_client.fetch_file_content(owner, repo, nb_path)
                if nb_content:
                    notebook_imports |= extract_notebook_imports(nb_content)
            except Exception as exc:
                logger.warning("Failed to extract notebook imports from %s: %s", nb_path, exc)

    # 6. Build LLM context
    context = build_context(
        owner=owner,
        repo=repo,
        repo_data=gh_data["repo_data"],
        languages=gh_data["languages"],
        readme=gh_data["readme"],
        tree=gh_data["tree"],
        file_contents=gh_data.get("file_contents", {}),
        notebook_imports=notebook_imports or None,
    )

    # 7. Call LLM
    try:
        llm_result = await llm_client.generate_summary(context)
    except Exception as exc:
        logger.exception("LLM call failed")
        return _error_response(502, f"LLM service error: {exc}")

    # 8. Post-filter technologies against evidence
    evidence = build_evidence_whitelist(
        languages=gh_data["languages"],
        file_contents=gh_data.get("file_contents", {}),
        notebook_imports=notebook_imports,
    )
    filtered_tech = filter_technologies(
        technologies=llm_result["technologies"],
        evidence=evidence,
        languages=gh_data["languages"],
    )

    # 9. Build response
    resp = SummarizeResponse(
        summary=llm_result["summary"],
        technologies=filtered_tech,
        structure=llm_result["structure"],
    )

    # 10. Cache
    await cache.store_with_latest(
        owner, repo,
        gh_data.get("default_branch", "main"),
        resp.model_dump(),
    )
    logger.info("Cache MISS → stored %s/%s", owner, repo)

    return resp
