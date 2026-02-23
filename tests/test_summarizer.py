"""
Tests for the repo content processor (summarizer module)
and the LLM client JSON parsing logic.
"""

import json
import pytest

from app.llm_client import LLMClient
from app.summarizer import (
    build_context,
    build_evidence_whitelist,
    extract_notebook_imports,
    filter_technologies,
    filter_tree,
    find_notebooks,
    normalize_import,
    select_key_files,
)


# ── Tree filtering ─────────────────────────────────────────────
class TestFilterTree:
    def test_skips_node_modules(self):
        tree = [
            {"path": "src", "type": "tree"},
            {"path": "node_modules", "type": "tree"},
            {"path": "node_modules/lodash", "type": "tree"},
        ]
        result = filter_tree(tree)
        paths = [n["path"] for n in result]
        assert "src" in paths
        assert "node_modules" not in paths
        assert "node_modules/lodash" not in paths

    def test_skips_pycache(self):
        tree = [
            {"path": "app", "type": "tree"},
            {"path": "__pycache__", "type": "tree"},
            {"path": "app/__pycache__/main.cpython-311.pyc", "type": "blob"},
        ]
        result = filter_tree(tree)
        paths = [n["path"] for n in result]
        assert "app" in paths
        assert "__pycache__" not in paths

    def test_skips_lock_files(self):
        tree = [
            {"path": "package.json", "type": "blob"},
            {"path": "package-lock.json", "type": "blob"},
            {"path": "yarn.lock", "type": "blob"},
            {"path": "poetry.lock", "type": "blob"},
        ]
        result = filter_tree(tree)
        paths = [n["path"] for n in result]
        assert "package.json" in paths
        assert "package-lock.json" not in paths
        assert "yarn.lock" not in paths
        assert "poetry.lock" not in paths

    def test_skips_binary_files(self):
        tree = [
            {"path": "README.md", "type": "blob"},
            {"path": "logo.png", "type": "blob"},
            {"path": "favicon.ico", "type": "blob"},
            {"path": "app.exe", "type": "blob"},
        ]
        result = filter_tree(tree)
        paths = [n["path"] for n in result]
        assert "README.md" in paths
        assert "logo.png" not in paths
        assert "favicon.ico" not in paths
        assert "app.exe" not in paths

    def test_keeps_source_files(self):
        tree = [
            {"path": "main.py", "type": "blob"},
            {"path": "index.js", "type": "blob"},
            {"path": "Dockerfile", "type": "blob"},
        ]
        result = filter_tree(tree)
        assert len(result) == 3

    def test_skips_dataset_files(self):
        tree = [
            {"path": "main.py", "type": "blob"},
            {"path": "data.csv", "type": "blob"},
            {"path": "dataset.parquet", "type": "blob"},
            {"path": "model.h5", "type": "blob"},
            {"path": "features.npy", "type": "blob"},
            {"path": "results.pkl", "type": "blob"},
        ]
        result = filter_tree(tree)
        paths = [n["path"] for n in result]
        assert "main.py" in paths
        assert "data.csv" not in paths
        assert "dataset.parquet" not in paths
        assert "model.h5" not in paths
        assert "features.npy" not in paths
        assert "results.pkl" not in paths


# ── Key file selection ─────────────────────────────────────────
class TestSelectKeyFiles:
    def test_selects_priority_files(self):
        tree = [
            {"path": "requirements.txt", "type": "blob"},
            {"path": "setup.py", "type": "blob"},
            {"path": "Dockerfile", "type": "blob"},
            {"path": "main.py", "type": "blob"},
        ]
        selected = select_key_files(tree)
        assert "requirements.txt" in selected
        assert "setup.py" in selected
        assert "Dockerfile" in selected
        assert "main.py" not in selected  # not a priority file

    def test_respects_max_limit(self):
        tree = [
            {"path": "pyproject.toml", "type": "blob"},
            {"path": "requirements.txt", "type": "blob"},
            {"path": "setup.py", "type": "blob"},
            {"path": "setup.cfg", "type": "blob"},
            {"path": "package.json", "type": "blob"},
            {"path": "go.mod", "type": "blob"},
            {"path": "Cargo.toml", "type": "blob"},
            {"path": "pom.xml", "type": "blob"},
        ]
        selected = select_key_files(tree)
        assert len(selected) <= 6

    def test_empty_tree(self):
        selected = select_key_files([])
        assert selected == []

    def test_ignores_directories(self):
        tree = [
            {"path": "requirements.txt", "type": "tree"},  # dir, not blob
        ]
        selected = select_key_files(tree)
        assert selected == []


# ── Context building ───────────────────────────────────────────
class TestBuildContext:
    def test_includes_all_sections(self):
        context = build_context(
            owner="psf",
            repo="requests",
            repo_data={
                "description": "HTTP library",
                "topics": ["python", "http"],
            },
            languages={"Python": 10000},
            readme="# Requests\n\nA nice HTTP library.",
            tree=[
                {"path": "README.md", "type": "blob", "size": 500},
                {"path": "src", "type": "tree"},
            ],
            file_contents={"requirements.txt": "urllib3\ncertifi"},
        )
        assert "psf/requests" in context
        assert "HTTP library" in context
        assert "Python" in context
        assert "README.md" in context
        assert "requirements.txt" in context
        assert "urllib3" in context

    def test_missing_readme(self):
        context = build_context(
            owner="o", repo="r",
            repo_data={"description": None, "topics": []},
            languages={},
            readme=None,
            tree=[],
            file_contents={},
        )
        assert "No README found" in context

    def test_empty_languages(self):
        context = build_context(
            owner="o", repo="r",
            repo_data={"description": "Test", "topics": []},
            languages={},
            readme="hello",
            tree=[],
            file_contents={},
        )
        assert "LANGUAGES_BYTES" not in context

    def test_includes_notebook_imports_section(self):
        context = build_context(
            owner="o", repo="r",
            repo_data={"description": "ML project", "topics": []},
            languages={"Python": 5000, "Jupyter Notebook": 15000},
            readme="Stroke prediction notebook",
            tree=[],
            file_contents={},
            notebook_imports={"pandas", "numpy", "sklearn", "matplotlib"},
        )
        assert "NOTEBOOK_IMPORTS" in context
        assert "scikit-learn" in context  # normalized from sklearn
        assert "pandas" in context
        assert "numpy" in context
        assert "matplotlib" in context


# ── Notebook import extraction ─────────────────────────────────
class TestNotebookImports:
    def _make_notebook(self, code_cells: list[str]) -> str:
        """Build a minimal .ipynb JSON from code cell sources."""
        cells = []
        for code in code_cells:
            cells.append({
                "cell_type": "code",
                "source": code.splitlines(keepends=True),
            })
        return json.dumps({"cells": cells})

    def test_extracts_import_statements(self):
        nb = self._make_notebook([
            "import pandas as pd\nimport numpy as np\n",
            "from sklearn.model_selection import train_test_split\n",
            "import matplotlib.pyplot as plt\n",
        ])
        imports = extract_notebook_imports(nb)
        assert "pandas" in imports
        assert "numpy" in imports
        assert "sklearn" in imports
        assert "matplotlib" in imports

    def test_excludes_stdlib(self):
        nb = self._make_notebook([
            "import os\nimport sys\nimport json\nimport re\n",
            "import pandas as pd\n",
        ])
        imports = extract_notebook_imports(nb)
        assert "os" not in imports
        assert "sys" not in imports
        assert "json" not in imports
        assert "re" not in imports
        assert "pandas" in imports

    def test_handles_invalid_json(self):
        imports = extract_notebook_imports("not valid json")
        assert imports == set()

    def test_handles_empty_notebook(self):
        nb = json.dumps({"cells": []})
        imports = extract_notebook_imports(nb)
        assert imports == set()

    def test_ignores_markdown_cells(self):
        nb = json.dumps({"cells": [
            {"cell_type": "markdown", "source": ["import pandas"]},
            {"cell_type": "code", "source": ["import numpy"]},
        ]})
        imports = extract_notebook_imports(nb)
        assert "pandas" not in imports
        assert "numpy" in imports


class TestFindNotebooks:
    def test_finds_ipynb_files(self):
        tree = [
            {"path": "analysis.ipynb", "type": "blob"},
            {"path": "main.py", "type": "blob"},
            {"path": "notebooks/explore.ipynb", "type": "blob"},
        ]
        result = find_notebooks(tree)
        assert "analysis.ipynb" in result
        assert "notebooks/explore.ipynb" in result

    def test_caps_at_three(self):
        tree = [{"path": f"nb{i}.ipynb", "type": "blob"} for i in range(10)]
        result = find_notebooks(tree)
        assert len(result) == 3


class TestNormalizeImport:
    def test_known_mappings(self):
        assert normalize_import("sklearn") == "scikit-learn"
        assert normalize_import("cv2") == "opencv-python"
        assert normalize_import("PIL") == "pillow"
        assert normalize_import("bs4") == "beautifulsoup4"
        assert normalize_import("torch") == "pytorch"

    def test_unknown_passes_through(self):
        assert normalize_import("some_custom_lib") == "some_custom_lib"


# ── Evidence whitelist ─────────────────────────────────────────
class TestBuildEvidenceWhitelist:
    def test_includes_languages(self):
        evidence = build_evidence_whitelist(
            languages={"Python": 10000, "Jupyter Notebook": 5000},
            file_contents={},
            notebook_imports=set(),
        )
        assert "python" in evidence
        assert "jupyter notebook" in evidence

    def test_includes_notebook_imports(self):
        evidence = build_evidence_whitelist(
            languages={},
            file_contents={},
            notebook_imports={"pandas", "sklearn", "matplotlib"},
        )
        assert "pandas" in evidence
        assert "scikit-learn" in evidence
        assert "matplotlib" in evidence

    def test_includes_requirements_deps(self):
        evidence = build_evidence_whitelist(
            languages={},
            file_contents={"requirements.txt": "flask>=2.0\nSQLAlchemy\ngunicorn"},
            notebook_imports=set(),
        )
        assert "flask" in evidence
        assert "sqlalchemy" in evidence
        assert "gunicorn" in evidence

    def test_includes_docker_evidence(self):
        evidence = build_evidence_whitelist(
            languages={},
            file_contents={"Dockerfile": "FROM python:3.11\nRUN pip install flask"},
            notebook_imports=set(),
        )
        assert "docker" in evidence


# ── Technology post-filter ─────────────────────────────────────
class TestFilterTechnologies:
    def test_keeps_evidenced_tech(self):
        evidence = {"python", "pandas", "numpy", "scikit-learn"}
        languages = {"Python": 10000}
        result = filter_technologies(
            ["Python", "pandas", "numpy", "scikit-learn"],
            evidence, languages,
        )
        assert "Python" in result
        assert "pandas" in result

    def test_filters_out_algorithms(self):
        evidence = {"python", "pandas", "scikit-learn"}
        languages = {"Python": 10000}
        result = filter_technologies(
            ["Python", "pandas", "Logistic Regression", "Random Forest",
             "Train-test split", "Decision Tree", "scikit-learn"],
            evidence, languages,
        )
        assert "Logistic Regression" not in result
        assert "Random Forest" not in result
        assert "Train-test split" not in result
        assert "Decision Tree" not in result
        assert "Python" in result
        assert "scikit-learn" in result

    def test_filters_unknown_multiword(self):
        evidence = {"python"}
        languages = {"Python": 10000}
        result = filter_technologies(
            ["Python", "Data Preprocessing", "Feature Engineering"],
            evidence, languages,
        )
        assert "Data Preprocessing" not in result
        assert "Feature Engineering" not in result

    def test_falls_back_to_languages_if_empty(self):
        evidence = set()
        languages = {"Python": 10000, "Jupyter Notebook": 5000}
        result = filter_technologies(
            ["Logistic Regression", "Random Forest"],
            evidence, languages,
        )
        # Should fall back to language names
        assert len(result) > 0
        assert "Python" in result

    def test_caps_at_twelve(self):
        evidence = {f"lib{i}" for i in range(20)}
        languages = {}
        techs = [f"lib{i}" for i in range(20)]
        result = filter_technologies(techs, evidence, languages)
        assert len(result) <= 12

    def test_deduplicates(self):
        evidence = {"python", "pandas"}
        languages = {"Python": 10000}
        result = filter_technologies(
            ["Python", "python", "PYTHON", "pandas", "pandas"],
            evidence, languages,
        )
        low_result = [t.lower() for t in result]
        assert low_result.count("python") == 1
        assert low_result.count("pandas") == 1

    def test_notebook_repo_end_to_end(self):
        """Simulates a notebook-only repo: only evidenced libs should survive."""
        notebook_imports = {"pandas", "numpy", "sklearn", "matplotlib"}
        languages = {"Python": 2000, "Jupyter Notebook": 18000}

        evidence = build_evidence_whitelist(
            languages=languages,
            file_contents={},
            notebook_imports=notebook_imports,
        )

        # Simulate LLM returning hallucinated algorithms
        llm_techs = [
            "Python", "Jupyter Notebook", "pandas", "numpy",
            "scikit-learn", "matplotlib",
            "Logistic Regression", "Random Forest",
            "Train-test split", "XGBoost", "Decision Tree",
        ]

        result = filter_technologies(llm_techs, evidence, languages)

        # Must include evidenced tech
        assert "Python" in result
        assert "pandas" in result
        assert "scikit-learn" in result
        assert "matplotlib" in result

        # Must NOT include algorithms/concepts
        assert "Logistic Regression" not in result
        assert "Random Forest" not in result
        assert "Train-test split" not in result
        assert "Decision Tree" not in result

        # XGBoost is a real library but NOT in notebook_imports → must be filtered
        assert "XGBoost" not in result


# ── LLM JSON parsing ──────────────────────────────────────────
class TestLLMJsonParsing:
    def test_parse_clean_json(self):
        raw = '{"summary": "Test", "technologies": ["Python"], "structure": "flat"}'
        result = LLMClient._parse_json(raw)
        assert result is not None
        assert result["summary"] == "Test"

    def test_parse_code_fenced_json(self):
        raw = '```json\n{"summary": "Test", "technologies": ["Python"], "structure": "flat"}\n```'
        result = LLMClient._parse_json(raw)
        assert result is not None
        assert result["summary"] == "Test"

    def test_parse_json_with_extra_text(self):
        raw = 'Here is the result:\n{"summary": "Test", "technologies": [], "structure": "flat"}\nDone!'
        result = LLMClient._parse_json(raw)
        assert result is not None
        assert result["summary"] == "Test"

    def test_parse_invalid_returns_none(self):
        result = LLMClient._parse_json("this is not json at all")
        assert result is None

    def test_validate_ensures_types(self):
        data = {"summary": 123, "technologies": "not-a-list", "structure": None}
        result = LLMClient._validate(data)
        assert isinstance(result["summary"], str)
        assert isinstance(result["technologies"], list)
        assert isinstance(result["structure"], str)
