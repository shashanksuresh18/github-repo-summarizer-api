"""
API integration tests — all GitHub and LLM calls fully mocked.
No real HTTP traffic leaves this process.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
import httpx
import respx
from fastapi.testclient import TestClient

from app.main import app, state


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset shared state between tests so cache doesn't leak."""
    state.github_client = None
    state.llm_client = None
    state.cache = None
    yield
    state.github_client = None
    state.llm_client = None
    state.cache = None


# ── Mock Data ──────────────────────────────────────────────────
REPO_META = {
    "name": "requests",
    "full_name": "psf/requests",
    "default_branch": "main",
    "description": "A simple, yet elegant, HTTP library.",
    "topics": ["python", "http", "requests"],
}

LANGUAGES = {"Python": 120000, "Shell": 5000}

README_TEXT = (
    "# Requests\n\n"
    "Requests is a simple, yet elegant, HTTP library for Python.\n"
    "It allows you to send HTTP/1.1 requests extremely easily."
)

TREE = {
    "sha": "abc123",
    "tree": [
        {"path": "README.md", "type": "blob", "size": 2048},
        {"path": "requirements.txt", "type": "blob", "size": 100},
        {"path": "setup.py", "type": "blob", "size": 800},
        {"path": "src", "type": "tree"},
        {"path": "src/requests", "type": "tree"},
        {"path": "tests", "type": "tree"},
        {"path": "docs", "type": "tree"},
    ],
    "truncated": False,
}

LLM_RESPONSE = {
    "summary": "Requests is a popular Python HTTP library for making web requests easily.",
    "technologies": ["Python", "urllib3", "certifi", "charset-normalizer"],
    "structure": "Standard Python package layout with source in src/requests/, tests in tests/, docs in docs/.",
}


def _mock_github_happy_path():
    """Register respx mocks for a successful GitHub flow."""
    respx.get("https://api.github.com/repos/psf/requests").mock(
        return_value=httpx.Response(200, json=REPO_META)
    )
    respx.get("https://api.github.com/repos/psf/requests/languages").mock(
        return_value=httpx.Response(200, json=LANGUAGES)
    )
    respx.get("https://api.github.com/repos/psf/requests/readme").mock(
        return_value=httpx.Response(200, text=README_TEXT)
    )
    respx.get("https://api.github.com/repos/psf/requests/git/trees/main").mock(
        return_value=httpx.Response(200, json=TREE)
    )
    # Key file content fetches
    respx.get("https://api.github.com/repos/psf/requests/contents/requirements.txt").mock(
        return_value=httpx.Response(200, text="urllib3\ncertifi\ncharset-normalizer")
    )
    respx.get("https://api.github.com/repos/psf/requests/contents/setup.py").mock(
        return_value=httpx.Response(200, text="from setuptools import setup\nsetup(name='requests')")
    )


def _mock_llm():
    """Return a patch that mocks the LLM generate_summary call."""
    return patch(
        "app.main.LLMClient.generate_summary",
        new_callable=AsyncMock,
        return_value=LLM_RESPONSE,
    )


# ── Tests ──────────────────────────────────────────────────────
def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@respx.mock
def test_summarize_happy_path():
    _mock_github_happy_path()

    with _mock_llm():
        with TestClient(app) as client:
            resp = client.post(
                "/summarize",
                json={"github_url": "https://github.com/psf/requests"},
            )

    assert resp.status_code == 200
    data = resp.json()

    # Exact response shape per spec
    assert "summary" in data
    assert "technologies" in data
    assert "structure" in data
    assert isinstance(data["summary"], str)
    assert isinstance(data["technologies"], list)
    assert isinstance(data["structure"], str)
    assert len(data["summary"]) > 10
    assert len(data["technologies"]) > 0


def test_summarize_invalid_url():
    with TestClient(app) as client:
        resp = client.post(
            "/summarize",
            json={"github_url": "https://gitlab.com/foo/bar"},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "error"
    assert "message" in body


def test_summarize_empty_url():
    with TestClient(app) as client:
        resp = client.post(
            "/summarize",
            json={"github_url": ""},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "error"


@respx.mock
def test_summarize_repo_not_found():
    respx.get("https://api.github.com/repos/owner/nonexistent").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )

    with TestClient(app) as client:
        resp = client.post(
            "/summarize",
            json={"github_url": "https://github.com/owner/nonexistent"},
        )

    assert resp.status_code == 404
    data = resp.json()
    assert data["status"] == "error"
    assert "not found" in data["message"].lower() or "Not Found" in data["message"]


@respx.mock
def test_summarize_rate_limited():
    respx.get("https://api.github.com/repos/owner/repo").mock(
        return_value=httpx.Response(
            429,
            json={"message": "rate limit exceeded"},
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": "1700000000",
            },
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/summarize",
            json={"github_url": "https://github.com/owner/repo"},
        )

    assert resp.status_code == 429
    data = resp.json()
    assert data["status"] == "error"


@respx.mock
def test_summarize_cache_hit():
    """Second request for the same repo should be a cache hit (no second LLM call)."""
    _mock_github_happy_path()

    with _mock_llm() as mock_llm:
        with TestClient(app) as client:
            # First call
            resp1 = client.post(
                "/summarize",
                json={"github_url": "https://github.com/psf/requests"},
            )
            assert resp1.status_code == 200

            # Second call — should use cache
            resp2 = client.post(
                "/summarize",
                json={"github_url": "https://github.com/psf/requests"},
            )
            assert resp2.status_code == 200
            assert resp2.json() == resp1.json()

        # LLM should only be called ONCE
        assert mock_llm.call_count == 1


@respx.mock
def test_summarize_no_readme():
    """Missing README should still return a valid response."""
    respx.get("https://api.github.com/repos/owner/empty").mock(
        return_value=httpx.Response(200, json={
            "name": "empty", "full_name": "owner/empty",
            "default_branch": "main", "description": None, "topics": [],
        })
    )
    respx.get("https://api.github.com/repos/owner/empty/languages").mock(
        return_value=httpx.Response(200, json={"Python": 1000})
    )
    respx.get("https://api.github.com/repos/owner/empty/readme").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )
    respx.get("https://api.github.com/repos/owner/empty/git/trees/main").mock(
        return_value=httpx.Response(200, json={"sha": "x", "tree": [], "truncated": False})
    )

    with _mock_llm():
        with TestClient(app) as client:
            resp = client.post(
                "/summarize",
                json={"github_url": "https://github.com/owner/empty"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "technologies" in data
    assert "structure" in data
