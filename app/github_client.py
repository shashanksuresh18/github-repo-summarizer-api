"""
Async GitHub REST API client.

Features:
- httpx.AsyncClient with configurable timeouts.
- Retries with exponential backoff on 5xx / network errors.
- ETag / If-None-Match conditional requests for README and repo metadata.
- Proper 403 rate-limit handling (reads X-RateLimit-Reset header).
- Shallow tree fetch (non-recursive, capped entries).
- README capped at configurable max chars.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.settings import settings

logger = logging.getLogger("app.github_client")


# ── Custom exceptions ──────────────────────────────────────────
class GitHubError(Exception):
    """Base for GitHub-related errors."""


class RepoNotFoundError(GitHubError):
    """404 — repository doesn't exist or is private."""


class RateLimitError(GitHubError):
    """403/429 — rate limit exceeded."""

    def __init__(self, message: str, reset_timestamp: int | None = None):
        super().__init__(message)
        self.reset_timestamp = reset_timestamp


class UpstreamError(GitHubError):
    """5xx — GitHub itself is having issues."""


# ── Helpers ─────────────────────────────────────────────────────
def _rate_limit_reset(response: httpx.Response) -> int | None:
    """Extract X-RateLimit-Reset header (unix timestamp) if present."""
    val = response.headers.get("x-ratelimit-reset")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return None


def _check_rate_limit(response: httpx.Response) -> None:
    """Raise RateLimitError if response indicates rate limiting."""
    if response.status_code in (403, 429):
        remaining = response.headers.get("x-ratelimit-remaining")
        # GitHub returns 403 with remaining=0 when rate-limited
        if response.status_code == 429 or remaining == "0":
            reset_ts = _rate_limit_reset(response)
            hint = " Try again later."
            if reset_ts:
                import datetime as _dt
                reset_dt = _dt.datetime.fromtimestamp(reset_ts, tz=_dt.timezone.utc)
                ts_str = reset_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                hint = f" Try again after {ts_str}."

            token_hint = ""
            if not settings.github_token:
                token_hint = " or set GITHUB_TOKEN for higher limits."

            raise RateLimitError(
                f"GitHub rate limit hit.{hint}{token_hint}",
                reset_timestamp=reset_ts,
            )


# ── ETag store (in-memory, per-client lifetime) ────────────────
class _ETagStore:
    """Very small store that keeps ETag → body mappings."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, Any]] = {}

    def get_etag(self, url: str) -> str | None:
        entry = self._store.get(url)
        return entry[0] if entry else None

    def get_body(self, url: str) -> Any | None:
        entry = self._store.get(url)
        return entry[1] if entry else None

    def save(self, url: str, etag: str, body: Any) -> None:
        self._store[url] = (etag, body)


# ── Client ──────────────────────────────────────────────────────
class GitHubClient:
    """Async GitHub REST API wrapper."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "repo-summarizer/1.0",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"

        self._client = client or httpx.AsyncClient(
            base_url=settings.github_api_base,
            headers=headers,
            timeout=httpx.Timeout(
                connect=settings.http_connect_timeout,
                read=settings.http_read_timeout,
                write=settings.http_read_timeout,
                pool=settings.http_read_timeout,
            ),
        )
        self._etags = _ETagStore()

    # ── Low-level request with retries ─────────────────────────
    async def _request(
        self,
        method: str,
        path: str,
        *,
        use_etag: bool = False,
        accept: str | None = None,
    ) -> httpx.Response:
        """Fire an HTTP request with retry + backoff on transient failures."""
        extra_headers: dict[str, str] = {}
        url = str(self._client.base_url) + path

        if use_etag:
            etag = self._etags.get_etag(url)
            if etag:
                extra_headers["If-None-Match"] = etag

        if accept:
            extra_headers["Accept"] = accept

        last_exc: Exception | None = None
        for attempt in range(1, settings.http_max_retries + 1):
            try:
                resp = await self._client.request(
                    method, path, headers=extra_headers
                )

                # 304 Not Modified — return cached body via a sentinel
                if resp.status_code == 304:
                    return resp

                # Rate-limit check
                _check_rate_limit(resp)

                # 404
                if resp.status_code == 404:
                    raise RepoNotFoundError(
                        f"Repository not found or private: {path}"
                    )

                # 5xx — retry
                if resp.status_code >= 500:
                    last_exc = UpstreamError(
                        f"GitHub returned {resp.status_code} for {path}"
                    )
                    await self._backoff(attempt)
                    continue

                # Success
                resp.raise_for_status()

                # Save ETag
                if use_etag:
                    etag_val = resp.headers.get("etag")
                    if etag_val:
                        # README raw endpoint returns text, not JSON
                        content_type = resp.headers.get("content-type", "")
                        if "json" in content_type:
                            try:
                                body = resp.json()
                            except Exception:
                                body = resp.text
                        else:
                            body = resp.text
                        self._etags.save(url, etag_val, body)

                return resp

            except (RateLimitError, RepoNotFoundError):
                raise
            except httpx.HTTPStatusError:
                raise
            except Exception as exc:  # network errors
                last_exc = exc
                logger.warning(
                    "GitHub request failed (attempt %d/%d): %s",
                    attempt, settings.http_max_retries, exc,
                )
                await self._backoff(attempt)

        raise UpstreamError(
            f"GitHub request failed after {settings.http_max_retries} retries"
        ) from last_exc

    @staticmethod
    async def _backoff(attempt: int) -> None:
        import asyncio

        wait = settings.http_backoff_base * (2 ** (attempt - 1))
        await asyncio.sleep(wait)

    # ── High-level fetch methods ───────────────────────────────
    async def fetch_repo(self, owner: str, repo: str) -> tuple[dict, float]:
        """Fetch repo metadata. Returns (data, elapsed_ms)."""
        t0 = time.perf_counter()
        resp = await self._request("GET", f"/repos/{owner}/{repo}", use_etag=True)
        elapsed = (time.perf_counter() - t0) * 1000

        if resp.status_code == 304:
            url = str(self._client.base_url) + f"/repos/{owner}/{repo}"
            data = self._etags.get_body(url)
        else:
            data = resp.json()

        return data, elapsed

    async def fetch_languages(self, owner: str, repo: str) -> tuple[dict, float]:
        t0 = time.perf_counter()
        resp = await self._request("GET", f"/repos/{owner}/{repo}/languages")
        elapsed = (time.perf_counter() - t0) * 1000
        return resp.json(), elapsed

    async def fetch_readme(self, owner: str, repo: str) -> tuple[str | None, float]:
        """Fetch README content (plain text). Returns None if not found."""
        t0 = time.perf_counter()
        try:
            resp = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/readme",
                accept="application/vnd.github.raw",
                use_etag=True,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if resp.status_code == 304:
                url = str(self._client.base_url) + f"/repos/{owner}/{repo}/readme"
                content = self._etags.get_body(url)
                # stored body might be json (from a different accept); handle gracefully
                if isinstance(content, dict):
                    import base64 as _b64
                    content = _b64.b64decode(content.get("content", "")).decode(
                        errors="replace"
                    )
            else:
                content = resp.text

            if content and len(content) > settings.readme_max_chars:
                content = content[: settings.readme_max_chars]

            return content, elapsed

        except RepoNotFoundError:
            # README endpoint returns 404 when no README exists
            elapsed = (time.perf_counter() - t0) * 1000
            return None, elapsed

    async def fetch_tree(
        self, owner: str, repo: str, branch: str
    ) -> tuple[list[dict], float]:
        """Fetch the repo tree **non-recursively** (shallow, top-level only).

        Returns a list of tree node dicts (path, type, size) capped at
        ``settings.tree_max_entries``.
        """
        t0 = time.perf_counter()
        # recursive=0 → shallow top-level listing
        resp = await self._request(
            "GET", f"/repos/{owner}/{repo}/git/trees/{branch}"
        )
        elapsed = (time.perf_counter() - t0) * 1000

        data = resp.json()
        tree_items: list[dict] = data.get("tree", [])

        # Cap entries
        if len(tree_items) > settings.tree_max_entries:
            tree_items = tree_items[: settings.tree_max_entries]

        return tree_items, elapsed

    async def fetch_file_content(
        self, owner: str, repo: str, path: str
    ) -> str | None:
        """Fetch raw content of a single file. Returns None on 404."""
        try:
            resp = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{path}",
                accept="application/vnd.github.raw",
            )
            if resp.status_code == 304:
                return None
            content = resp.text
            if content and len(content) > settings.file_max_chars:
                content = content[: settings.file_max_chars]
            return content
        except RepoNotFoundError:
            return None
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", path, exc)
            return None

    async def fetch_all(
        self, owner: str, repo: str, key_files: list[str] | None = None
    ) -> dict[str, Any]:
        """Orchestrate all fetches and return aggregated data + timings."""
        # 1) Repo metadata (must be first — we need default_branch)
        repo_data, t_repo = await self.fetch_repo(owner, repo)
        default_branch = repo_data.get("default_branch", "main")

        # 2) Languages
        languages, t_lang = await self.fetch_languages(owner, repo)

        # 3) README
        readme, t_readme = await self.fetch_readme(owner, repo)

        # 4) Tree (shallow)
        tree, t_tree = await self.fetch_tree(owner, repo, default_branch)

        # 5) Key file contents
        file_contents: dict[str, str] = {}
        if key_files:
            for fpath in key_files:
                content = await self.fetch_file_content(owner, repo, fpath)
                if content:
                    file_contents[fpath] = content

        return {
            "repo_data": repo_data,
            "default_branch": default_branch,
            "languages": languages,
            "readme": readme,
            "tree": tree,
            "file_contents": file_contents,
            "timings": {
                "repo": round(t_repo, 1),
                "languages": round(t_lang, 1),
                "readme": round(t_readme, 1),
                "tree": round(t_tree, 1),
            },
        }

    async def aclose(self) -> None:
        await self._client.aclose()
