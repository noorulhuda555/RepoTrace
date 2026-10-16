import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

from .ast_parser import CodeStructuralParser
from .code_graph import DependencyGraphMiner
from .git_miner import GitRepositoryMiner
from .ir_engine import BM25InformationRetrievalEngine
from .models import CodeMatch, CodeUnit
from .semantic_engine import SemanticVectorEngine
from .stacktrace_miner import StackTraceMiner
from .tokenizer import CodeTokenizer

_SOURCE_EXTENSIONS = {
    ".java", ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".cs",
    ".kt", ".kts", ".cpp", ".cc", ".c", ".h", ".hpp", ".php", ".rb",
    ".scala", ".swift", ".html", ".xhtml", ".jsp", ".vue", ".xml",
    ".properties", ".yaml", ".yml", ".sql",
}
_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".gradle", ".mvn", ".venv",
    "venv", "node_modules", "vendor", "build", "dist", "target", "coverage",
    "__pycache__", ".next", ".tox",
}
_SPLIT = re.compile(r"[^A-Za-z0-9_$]+")
_LOGGER = logging.getLogger(__name__)
# Prefer finer-grained units when overall scores tie (method > class > file).
_KIND_RANK = {"method": 0, "class": 1, "file": 2, "view": 2}


class HybridCodeIntelligenceEngine:
    """Mine a full source repository and rank likely issue locations across architectural layers."""

    def __init__(self, repository_path=None, max_file_bytes=None):
        self.repository_path = Path(repository_path).resolve() if repository_path else None
        self.max_file_bytes = max_file_bytes
        self.parser = CodeStructuralParser()
        self.tokenizer = CodeTokenizer()
        self.stack_traces = StackTraceMiner()
        self.units: List[CodeUnit] = []
        self._indexed_mtimes = {}
        self._bm25 = BM25InformationRetrievalEngine()
        self._semantic = SemanticVectorEngine()
        self._graph = DependencyGraphMiner()

    def index_repository(self, repository_path=None):
        if repository_path is not None:
            self.repository_path = Path(repository_path).resolve()
        if self.repository_path is None or not self.repository_path.is_dir():
            raise ValueError("A valid repository directory is required.")

        units: List[CodeUnit] = []
        mtimes = {}
        for root, directories, files in os.walk(
            self.repository_path, followlinks=False, onerror=self._log_walk_error
        ):
            directories[:] = sorted(
                name for name in directories
                if name.casefold() not in _SKIP_DIRS
            )
            for filename in sorted(files):
                path = Path(root, filename)
                if path.suffix.lower() not in _SOURCE_EXTENSIONS:
                    continue
                try:
                    stat = path.stat()
                    if self.max_file_bytes is not None and stat.st_size > self.max_file_bytes:
                        continue
                    relative = path.relative_to(self.repository_path).as_posix()
                    source = path.read_text(encoding="utf-8", errors="replace")
                except (OSError, ValueError) as error:
                    _LOGGER.warning("Skipping unreadable source file %s: %s", path, error)
                    continue
                mtimes[relative] = stat.st_mtime_ns
                units.extend(self.parser.parse_file(Path(relative), source))

        self._replace_index(units)
        self._indexed_mtimes = mtimes
        return {"repository": str(self.repository_path), "files": len(mtimes), "units": len(units)}

    def update_repository_index(self):
        if self.repository_path is None or not self.repository_path.is_dir():
            raise ValueError("A valid repository directory is required.")
        current = {}
        for root, directories, files in os.walk(
            self.repository_path, followlinks=False, onerror=self._log_walk_error
        ):
            directories[:] = [
                name for name in directories
                if name.casefold() not in _SKIP_DIRS
            ]
            for filename in files:
                path = Path(root, filename)
                if path.suffix.lower() not in _SOURCE_EXTENSIONS:
                    continue
                try:
                    stat = path.stat()
                    if self.max_file_bytes is None or stat.st_size <= self.max_file_bytes:
                        current[path.relative_to(self.repository_path).as_posix()] = stat.st_mtime_ns
                except (OSError, ValueError) as error:
                    _LOGGER.warning("Could not inspect source file %s: %s", path, error)
                    continue

        changed = {
            path for path, mtime in current.items()
            if self._indexed_mtimes.get(path) != mtime
        }
        removed = set(self._indexed_mtimes) - set(current)
        if not changed and not removed:
            return {"updated_files": 0, "removed_files": 0, "units": len(self.units)}

        units = [
            unit for unit in self.units
            if unit.path not in changed and unit.path not in removed
        ]
        for relative in sorted(changed):
            try:
                source = (self.repository_path / relative).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError as error:
                _LOGGER.warning("Could not read changed source file %s: %s", relative, error)
                continue
            units.extend(self.parser.parse_file(Path(relative), source))

        self._replace_index(units)
        self._indexed_mtimes = current
        return {
            "updated_files": len(changed),
            "removed_files": len(removed),
            "units": len(self.units),
        }

    def _replace_index(self, units):
        self.units = units
        documents = [unit.search_text for unit in units]
        self._bm25.fit(documents)
        self._semantic.fit(documents)
        self._graph.fit(units)

    def analyze_issue(self, issue, skills=None, top_k=10):
        if not self.units:
            raise ValueError("Index a repository before analyzing an issue.")
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer.")

        if isinstance(issue, str):
            issue_text = issue
        else:
            issue_text = "{}\n{}".format(
                issue.get("title", ""), issue.get("body", issue.get("description", ""))
            )
        issue_text = issue_text.strip()
        if not issue_text:
            raise ValueError("Issue title or description must not be empty.")

        query = "{}\n{}".format(issue_text, self._skill_text(skills)).strip()
        has_test_query = any(w in query.casefold() for w in ("test", "tests", "assert", "mock"))

        # Detect requested architectural roles (e.g., 'controller', 'service', 'view')
        requested_roles = self.tokenizer.extract_role_intent(query)

        # Retrieval runs
        candidate_pool_size = max(80, top_k * 10)
        bm25 = dict(self._bm25.search(query, candidate_pool_size))
        semantic = dict(self._semantic.search(query, candidate_pool_size))
        frames = self.stack_traces.extract_frames(issue_text)
        graph = self._graph.scores(self.units, query)
        history = GitRepositoryMiner(self.repository_path).recent_file_scores(query)

        bm25_max = max(bm25.values(), default=0)
        semantic_max = max(semantic.values(), default=0)

        # Dynamic Adaptive Weights: Stack trace presence shifts balance
        if frames:
            w_stack = 0.35
            w_bm25 = 0.25
            w_semantic = 0.20
            w_struct = 0.12
            w_git = 0.04
            w_skills = 0.04
        else:
            w_stack = 0.00
            w_bm25 = 0.38
            w_semantic = 0.32
            w_struct = 0.20
            w_git = 0.05
            w_skills = 0.05

        results: List[CodeMatch] = []

        for index, unit in enumerate(self.units):
            bm25_sig = bm25.get(index, 0) / bm25_max if bm25_max else 0
            semantic_sig = semantic.get(index, 0) / semantic_max if semantic_max else 0
            stack_sig = self.stack_traces.score(unit, frames) if frames else 0
            struct_sig = graph.get(index, 0)
            git_sig = history.get(unit.path, 0)
            skills_sig = self._skill_match(unit, skills)

            signals = {
                "text": bm25_sig,
                "semantic": semantic_sig,
                "stack_trace": stack_sig,
                "structural": struct_sig,
                "git_history": git_sig,
                "skills": skills_sig,
            }

            base_score = (
                w_bm25 * signals["text"]
                + w_semantic * signals["semantic"]
                + w_stack * signals["stack_trace"]
                + w_struct * signals["structural"]
                + w_git * signals["git_history"]
                + w_skills * signals["skills"]
            )

            if base_score <= 0:
                continue

            # Role Intent Match: boost if unit fulfills a specifically requested role
            role_boost = 1.0
            if unit.role and unit.role in requested_roles:
                role_boost = 1.25

            # Production Code Prioritization vs Test Code:
            # If the unit is a test file and query didn't explicitly ask for tests, demote test cases
            test_factor = 0.35 if (unit.is_test and not has_test_query) else 1.0

            # When a stack frame is present, prefer method/class hits over the whole file
            stack_specificity = 1.0
            if frames and stack_sig > 0:
                if unit.kind == "method":
                    stack_specificity = 1.2
                elif unit.kind == "file":
                    stack_specificity = 0.75

            final_score = base_score * role_boost * test_factor * stack_specificity

            snippet = " ".join(unit.content.strip().split())
            if len(snippet) > 360:
                snippet = snippet[:357] + "..."

            results.append(CodeMatch(
                path=unit.path,
                kind=unit.kind,
                name=unit.name,
                start_line=unit.start_line,
                end_line=unit.end_line,
                score=final_score,
                reason=self._explain(signals, unit, requested_roles),
                snippet=snippet,
                signals=signals,
                role=unit.role,
                is_test=unit.is_test,
            ))

        # Rank descending; break ties toward more specific code units
        results.sort(
            key=lambda item: (
                -item.score,
                _KIND_RANK.get(item.kind, 3),
                item.path,
                item.start_line,
            )
        )

        # Categorize architectural layer recommendations
        layers = self._categorize_layers(results)

        return {
            "query": issue_text,
            "repository": str(self.repository_path),
            "indexed_units": len(self.units),
            "layers": layers,
            "results": [match.to_dict() for match in results[:max(0, top_k)]],
        }

    def _categorize_layers(self, matches: List[CodeMatch]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Identifies top production candidates for Controller, Service, Repository, View, Model."""
        layers = {
            "controller": None,
            "service": None,
            "repository": None,
            "view": None,
            "model": None,
            "tests": []
        }

        for match in matches:
            if match.is_test:
                if len(layers["tests"]) < 3:
                    layers["tests"].append(match.to_dict())
                continue

            role = match.role
            if role == "controller" and layers["controller"] is None:
                layers["controller"] = match.to_dict()
            elif role == "repository" and layers["repository"] is None:
                layers["repository"] = match.to_dict()
            elif role == "service" and layers["service"] is None:
                layers["service"] = match.to_dict()
            elif role == "view" and layers["view"] is None:
                layers["view"] = match.to_dict()
            elif role == "model" and layers["model"] is None:
                layers["model"] = match.to_dict()

        return layers

    @staticmethod
    def _log_walk_error(error):
        _LOGGER.warning("Could not scan repository directory %s: %s", error.filename, error)

    @staticmethod
    def _skill_text(skills):
        if not skills:
            return ""
        if isinstance(skills, str):
            return skills
        if isinstance(skills, dict):
            return " ".join(str(name) for name, value in skills.items() if value)
        return " ".join(str(skill) for skill in skills if skill)

    def _skill_match(self, unit, skills):
        terms = {
            part.casefold()
            for part in _SPLIT.split(self._skill_text(skills)) if len(part) > 1
        }
        if not terms:
            return 0.0
        unit_terms = set(self.tokenizer.tokenize(unit.search_text))
        return len(terms & unit_terms) / len(terms)

    @staticmethod
    def _explain(signals, unit: CodeUnit, requested_roles: Set[str]):
        labels = {
            "text": "issue terms match",
            "semantic": "similar code language",
            "stack_trace": "stack trace location",
            "structural": "matching symbols or dependencies",
            "git_history": "related Git history",
            "skills": "predicted skill match",
        }
        reasons = []
        if unit.role and unit.role in requested_roles:
            reasons.append(f"targeted {unit.role} match")

        ordered = sorted(signals, key=signals.get, reverse=True)
        for key in ordered:
            if signals[key] > 0.1:
                reasons.append(labels[key])

        if unit.is_test:
            reasons.append("test suite unit")

        return "; ".join(reasons) or "relevance match"
