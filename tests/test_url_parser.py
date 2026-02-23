"""Tests for app.url_parser — GitHub URL parsing and validation."""

import pytest

from app.url_parser import parse_github_url


# ── Valid URLs ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/fastapi/fastapi", ("fastapi", "fastapi")),
        ("https://github.com/owner/repo/", ("owner", "repo")),
        ("https://github.com/owner/repo.git", ("owner", "repo")),
        ("https://github.com/owner/repo.git/", ("owner", "repo")),
        ("http://github.com/owner/repo", ("owner", "repo")),
        ("https://github.com/some-org/my_repo.py", ("some-org", "my_repo.py")),
        ("  https://github.com/owner/repo  ", ("owner", "repo")),
    ],
)
def test_parse_valid_url(url: str, expected: tuple[str, str]):
    assert parse_github_url(url) == expected


# ── Invalid URLs ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not-a-url",
        "https://gitlab.com/owner/repo",
        "https://github.com/",
        "https://github.com/owner",
        "https://github.com/owner/",
        "ftp://github.com/owner/repo",
        "https://github.com/owner/repo/extra/path",
    ],
)
def test_parse_invalid_url(url: str):
    with pytest.raises(ValueError):
        parse_github_url(url)
