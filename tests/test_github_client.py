"""Tests for app.github_client — all HTTP calls mocked via respx."""

import pytest
import httpx
import respx

from app.github_client import GitHubClient, RateLimitError, RepoNotFoundError


@pytest.fixture
def mock_client():
    """Create a GitHubClient with a controlled httpx.AsyncClient."""
    transport = httpx.MockTransport(lambda req: httpx.Response(200))
    client = httpx.AsyncClient(
        base_url="https://api.github.com",
        transport=transport,
    )
    gc = GitHubClient(client=client)
    return gc


# ── Repo metadata ──────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_fetch_repo_success():
    payload = {
        "name": "fastapi",
        "full_name": "fastapi/fastapi",
        "default_branch": "main",
        "description": "FastAPI framework",
        "topics": ["python", "api"],
    }
    respx.get("https://api.github.com/repos/fastapi/fastapi").mock(
        return_value=httpx.Response(200, json=payload)
    )

    async with httpx.AsyncClient(base_url="https://api.github.com") as hc:
        gc = GitHubClient(client=hc)
        data, elapsed = await gc.fetch_repo("fastapi", "fastapi")

    assert data["default_branch"] == "main"
    assert data["description"] == "FastAPI framework"
    assert elapsed >= 0


@pytest.mark.asyncio
@respx.mock
async def test_fetch_repo_not_found():
    respx.get("https://api.github.com/repos/owner/missing").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )

    async with httpx.AsyncClient(base_url="https://api.github.com") as hc:
        gc = GitHubClient(client=hc)
        with pytest.raises(RepoNotFoundError):
            await gc.fetch_repo("owner", "missing")


# ── Rate limit ─────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_fetch_repo_rate_limited():
    respx.get("https://api.github.com/repos/owner/repo").mock(
        return_value=httpx.Response(
            429,
            json={"message": "rate limit"},
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": "1700000000",
            },
        )
    )

    async with httpx.AsyncClient(base_url="https://api.github.com") as hc:
        gc = GitHubClient(client=hc)
        with pytest.raises(RateLimitError) as exc_info:
            await gc.fetch_repo("owner", "repo")
        assert exc_info.value.reset_timestamp == 1700000000


# ── Languages ──────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_fetch_languages():
    respx.get("https://api.github.com/repos/owner/repo/languages").mock(
        return_value=httpx.Response(200, json={"Python": 50000, "HTML": 3000})
    )

    async with httpx.AsyncClient(base_url="https://api.github.com") as hc:
        gc = GitHubClient(client=hc)
        data, elapsed = await gc.fetch_languages("owner", "repo")

    assert data == {"Python": 50000, "HTML": 3000}


# ── README ─────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_fetch_readme_success():
    respx.get("https://api.github.com/repos/owner/repo/readme").mock(
        return_value=httpx.Response(200, text="# My Project\nHello world")
    )

    async with httpx.AsyncClient(base_url="https://api.github.com") as hc:
        gc = GitHubClient(client=hc)
        content, elapsed = await gc.fetch_readme("owner", "repo")

    assert content is not None
    assert "My Project" in content


@pytest.mark.asyncio
@respx.mock
async def test_fetch_readme_not_found():
    respx.get("https://api.github.com/repos/owner/repo/readme").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )

    async with httpx.AsyncClient(base_url="https://api.github.com") as hc:
        gc = GitHubClient(client=hc)
        content, elapsed = await gc.fetch_readme("owner", "repo")

    assert content is None


# ── Tree ───────────────────────────────────────────────────────
@pytest.mark.asyncio
@respx.mock
async def test_fetch_tree_shallow():
    tree_payload = {
        "sha": "abc123",
        "tree": [
            {"path": "README.md", "type": "blob", "size": 1234},
            {"path": "src", "type": "tree"},
            {"path": "Dockerfile", "type": "blob", "size": 256},
        ],
        "truncated": False,
    }
    respx.get("https://api.github.com/repos/owner/repo/git/trees/main").mock(
        return_value=httpx.Response(200, json=tree_payload)
    )

    async with httpx.AsyncClient(base_url="https://api.github.com") as hc:
        gc = GitHubClient(client=hc)
        items, elapsed = await gc.fetch_tree("owner", "repo", "main")

    assert len(items) == 3
    assert items[0]["path"] == "README.md"
