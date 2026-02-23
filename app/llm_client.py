"""
Nebius Token Factory LLM client.

Uses the OpenAI-compatible SDK to call Nebius models.
Returns structured JSON with summary, technologies, and structure.
"""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI

from app.settings import settings

logger = logging.getLogger("app.llm_client")

_SYSTEM_PROMPT = """\
You are a senior software engineer analysing a GitHub repository.
You will receive repository metadata, directory structure, README content, and key configuration files.

Return ONLY a valid JSON object with exactly these three keys:
{
  "summary": "A clear 3–5 sentence description of what this project does, its purpose, and notable features.",
  "technologies": ["list", "of", "main", "languages", "frameworks", "libraries", "tools"],
  "structure": "One paragraph (1–3 sentences) describing how the project is organised: main directories, where source code lives, tests, docs, config, etc."
}

STRICT RULES FOR technologies:
- The technologies list must ONLY include items that appear in LANGUAGES_BYTES, DEPENDENCY_SNIPPETS, or NOTEBOOK_IMPORTS sections.
- Include programming languages, frameworks, libraries, and build tools ONLY.
- Do NOT include ML algorithms or concepts (e.g., "Logistic Regression", "Random Forest", "Train-test split", "Decision Tree").
- Do NOT guess or infer technologies that are not explicitly evidenced in the provided context.
- List 5–12 items.

Rules for summary:
- Be specific about the project's purpose. Mention what problem it solves.
- 3–5 sentences.

Rules for structure:
- Describe the directory layout based on the tree provided.
- 1–3 sentences.

Output rules:
- Do NOT wrap the JSON in markdown code fences or add any text before/after.
- Do NOT include any explanation outside the JSON object.
"""

_RETRY_PROMPT = (
    "Your previous response was not valid JSON. "
    "Return ONLY a valid JSON object with keys: summary, technologies, structure. "
    "No code fences, no extra text."
)

# Strip ```json ... ``` wrappers
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


class LLMClient:
    """Async wrapper around the Nebius Token Factory chat completions API."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key,
        )
        self._model = settings.nebius_model

    async def generate_summary(self, context: str) -> dict:
        """Send repo context to the LLM and return parsed JSON response.

        Retries once with a correction prompt if JSON parsing fails.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        raw = await self._call(messages)
        result = self._parse_json(raw)

        if result is not None:
            return self._validate(result)

        # Retry once with correction prompt
        logger.warning("LLM returned invalid JSON, retrying with correction prompt")
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": _RETRY_PROMPT})

        raw_retry = await self._call(messages)
        result_retry = self._parse_json(raw_retry)

        if result_retry is not None:
            return self._validate(result_retry)

        # Final fallback: return raw text as summary
        logger.error("LLM retry also failed, using raw text as fallback")
        return {
            "summary": raw.strip()[:1000],
            "technologies": [],
            "structure": "Unable to parse project structure from LLM response.",
        }

    async def _call(self, messages: list[dict]) -> str:
        """Make the chat completion request and return raw text."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        logger.debug("LLM raw response (first 500 chars): %s", content[:500])
        return content

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Try to parse JSON from LLM output, stripping code fences if needed."""
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip code fences and retry
        match = _CODE_FENCE_RE.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _validate(data: dict) -> dict:
        """Ensure all required keys exist with correct types."""
        summary = str(data.get("summary", ""))
        technologies = data.get("technologies", [])
        structure = str(data.get("structure", ""))

        if not isinstance(technologies, list):
            technologies = [str(technologies)]
        technologies = [str(t) for t in technologies if t]

        return {
            "summary": summary,
            "technologies": technologies,
            "structure": structure,
        }

    async def aclose(self) -> None:
        await self._client.close()
