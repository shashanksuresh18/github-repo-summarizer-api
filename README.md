# GitHub Repository Summarizer API

A FastAPI service that takes a GitHub repository URL, fetches its contents, and uses an LLM (via [Nebius Token Factory](https://tokenfactory.nebius.com/)) to generate a human-readable summary of the project.

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <this-repo>
cd github-repo-summarizer-api
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example `.env` file and add your Nebius API key:

```bash
cp .env.example .env
```

Edit `.env` and set your key:

```
NEBIUS_API_KEY=your-nebius-api-key-here
NEBIUS_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

**Note:** `.env` is optional for local development. In evaluation, the API key will be provided via the `NEBIUS_API_KEY` environment variable.


Alternatively, export environment variables directly:

```bash
export NEBIUS_API_KEY=your-nebius-api-key-here
```

**Optional:** Set `GITHUB_TOKEN` for higher GitHub API rate limits (5,000 requests/hour vs. 60 unauthenticated).

### 3. Start the server

```bash
uvicorn app.main:app --reload
```

The server starts on `http://localhost:8000`.

### 4. Test it

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

## Model Choice

**Qwen/Qwen3-30B-A3B-Instruct-2507** — selected for its strong instruction-following capability, which ensures reliable structured JSON output, combined with low inference cost ($0.10/M input tokens) that enables rapid iteration within the $1 trial budget.

## Repo Processing Strategy

The service builds a compact "context pack" for the LLM instead of sending the entire repository:

### What's included
- **Repository metadata**: description, topics, default branch
- **Languages breakdown**: from GitHub's language detection API
- **README content**: capped at 15,000 characters for context efficiency
- **Directory tree**: filtered and capped at 300 entries
- **Key configuration files** (up to 6): `requirements.txt`, `pyproject.toml`, `package.json`, `setup.py`, `Dockerfile`, `go.mod`, etc. — each capped at 10,000 characters
- **Notebook import extraction**: for `.ipynb` repos, actual `import` statements are parsed from notebook cells and listed as `NOTEBOOK_IMPORTS` evidence — this grounds the LLM's technology list in real data

### What's skipped
- **Binary files**: images, executables, archives, fonts
- **Lock files**: `package-lock.json`, `yarn.lock`, `poetry.lock`, etc.
- **Generated directories**: `node_modules/`, `dist/`, `build/`, `__pycache__/`, `.venv/`
- **Dataset/model files**: `.csv`, `.parquet`, `.h5`, `.npy`, `.pkl`, `.joblib`
- **Large/irrelevant files**: `.min.js`, `.map`, databases

### Anti-hallucination (technology post-filter)
After the LLM returns its JSON, the `technologies` list is filtered against an **evidence whitelist** built from:
1. GitHub languages API names
2. Dependency file packages (requirements.txt, package.json, etc.)
3. Notebook import statements

Any item not evidenced — or matching banned algorithm/concept patterns (e.g., "Logistic Regression", "Random Forest", "Train-test split") — is removed. This ensures only real languages, libraries, and frameworks appear in the output.

### Why this approach
The LLM doesn't need raw source code to understand a project. README + directory structure + dependency/config files provide enough signal for an accurate summary. This keeps the prompt compact and the API response fast.

### Context Management & Safety
- **Total context budget**: The system enforces a **hard cap of 40,000 characters** for the total LLM prompt. Sections are added in priority order (Metadata → Languages → Tree → README → Dependencies); lower-priority sections are automatically dropped if the budget is reached, preventing token overflows.
- **Deterministic JSON**: Uses `response_format: {"type": "json_object"}` to ensure the LLM output is always valid JSON.
- **Graceful Rate-Limiting**: Detailed error messages provide the exact UTC time when the GitHub rate limit resets and suggest setting `GITHUB_TOKEN` for higher limits.


## API Reference

### `POST /summarize`

**Request:**
```json
{
  "github_url": "https://github.com/psf/requests"
}
```

**Response (200):**
```json
{
  "summary": "Requests is a popular Python HTTP library...",
  "technologies": ["Python", "urllib3", "certifi"],
  "structure": "Standard Python package layout with source in src/requests/..."
}
```

**Error Response (400 — invalid URL):**
```json
{
  "status": "error",
  "message": "Not a valid GitHub repository URL"
}
```

**Error Response (429 — rate limited):**
```json
{
  "status": "error",
  "message": "GitHub rate limit hit. Try again after <ISO_UTC_TIMESTAMP>. Consider setting GITHUB_TOKEN for higher limits."
}
```

**Error Response (404 — repo not found):**
```json
{
  "status": "error",
  "message": "Repository not found or private: /repos/owner/nonexistent"
}
```

### `GET /health`

Returns `{"status": "ok"}` for health checks.

## Running Tests

```bash
pytest tests/ -v
```

All tests are fully mocked — no real GitHub or LLM API calls are made.

## Docker

```bash
docker build -t repo-summarizer .
docker run -p 8000:8000 -e NEBIUS_API_KEY=your-key repo-summarizer
```

## Project Structure

```
├── app/
│   ├── main.py            # FastAPI app, /summarize endpoint
│   ├── models.py           # Pydantic request/response models
│   ├── settings.py         # Environment-based configuration
│   ├── github_client.py    # Async GitHub API client with retries
│   ├── llm_client.py       # Nebius Token Factory LLM integration
│   ├── summarizer.py       # Repo content processing & context building
│   ├── url_parser.py       # GitHub URL validation
│   ├── cache.py            # In-memory TTL cache
│   └── logging_config.py   # Structured JSON logging
├── tests/
│   ├── test_api.py         # API integration tests
│   ├── test_summarizer.py  # Content processing & JSON parsing tests
│   ├── test_github_client.py
│   └── test_url_parser.py
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```
**Submission:** Do not include `.env` in the submitted ZIP. Include `.env.example` only.
