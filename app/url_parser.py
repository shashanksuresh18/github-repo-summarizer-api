"""
Robust GitHub URL parsing and validation.

Accepted forms:
  https://github.com/owner/repo
  https://github.com/owner/repo/
  https://github.com/owner/repo.git
  http://github.com/owner/repo   (also accepted)

Anything else raises ValueError with a human-readable message.
"""

from __future__ import annotations

import re

_GITHUB_URL_RE = re.compile(
    r"^https?://github\.com/"
    r"(?P<owner>[A-Za-z0-9\-_.]+)/"
    r"(?P<repo>[A-Za-z0-9\-_.]+?)"
    r"(?:\.git)?/?\s*$"
)


def parse_github_url(url: str) -> tuple[str, str]:
    """Return (owner, repo) from a GitHub URL.

    Raises ``ValueError`` with a descriptive message when the URL is
    not a valid GitHub repository URL.
    """
    if not url or not url.strip():
        raise ValueError("Repository URL must not be empty.")

    url = url.strip()

    match = _GITHUB_URL_RE.match(url)
    if not match:
        raise ValueError(
            f"Invalid GitHub repository URL: '{url}'. "
            "Expected format: https://github.com/owner/repo"
        )

    owner = match.group("owner")
    repo = match.group("repo")
    return owner, repo
