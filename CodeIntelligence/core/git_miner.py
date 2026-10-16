import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


class GitRepositoryMiner:
    """Mine recent commit subjects and changed files for localization evidence with safe fallback."""

    def __init__(self, repository_path, timeout=20):
        self.repository_path = str(repository_path)
        self.timeout = timeout
        git_installed = shutil.which("git") is not None
        is_git_dir = (Path(self.repository_path) / ".git").is_dir()
        self.is_git_repo = git_installed and is_git_dir

    def _run(self, *args):
        if not self.is_git_repo:
            return ""
        try:
            result = subprocess.run(
                ["git", "-C", self.repository_path, *args],
                capture_output=True, check=False, text=True,
                encoding="utf-8", errors="replace", timeout=self.timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as error:
            _LOGGER.debug("Git command failed for %s: %s", self.repository_path, error)
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout

    def recent_file_scores(self, query, limit=300):
        if not self.is_git_repo:
            return self._filesystem_scores(query)

        output = self._run(
            "log", "-n", str(max(1, limit)), "--format=%ct%x09%s", "--name-only"
        )
        if not output:
            return self._filesystem_scores(query)

        terms = set(query.casefold().split())
        scores = {}
        timestamp = int(time.time())
        subject = ""
        for line in output.splitlines():
            if "\t" in line:
                raw_timestamp, subject = line.split("\t", 1)
                try:
                    timestamp = int(raw_timestamp)
                except ValueError:
                    timestamp = int(time.time())
            elif line.strip():
                path = line.strip().replace("\\", "/")
                overlap = sum(term in path.casefold() or term in subject.casefold() for term in terms)
                if overlap:
                    age_years = max(0.0, (time.time() - timestamp) / (365.25 * 24 * 3600))
                    relevance = overlap * (1 + 0.15 / (1 + age_years))
                    scores[path] = max(scores.get(path, 0.0), relevance)
        maximum = max(scores.values(), default=0)
        return {path: score / maximum for path, score in scores.items()} if maximum else {}

    def _filesystem_scores(self, query):
        """Fallback when .git is not present: uses file modification times and path matching."""
        terms = set(query.casefold().split())
        scores = {}
        now = time.time()
        for root, _, files in os.walk(self.repository_path):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.relpath(full, self.repository_path).replace("\\", "/")
                overlap = sum(term in rel.casefold() for term in terms)
                if overlap:
                    try:
                        mtime = os.path.getmtime(full)
                        age_days = max(0.0, (now - mtime) / 86400.0)
                        recency = 1.0 / (1.0 + age_days * 0.05)
                        scores[rel] = overlap * recency
                    except OSError:
                        scores[rel] = overlap
        maximum = max(scores.values(), default=0)
        return {path: score / maximum for path, score in scores.items()} if maximum else {}

    def mine_repository_history(self, max_commits=300):
        if not self.is_git_repo:
            return {}
        output = self._run(
            "log", "-n", str(max(1, max_commits)), "--format=%ct%x09%s", "--name-only"
        )
        history = {}
        timestamp = 0
        subject = ""
        for line in output.splitlines():
            if "\t" in line:
                value, subject = line.split("\t", 1)
                try:
                    timestamp = int(value)
                except ValueError:
                    timestamp = 0
            elif line.strip():
                path = line.strip().replace("\\", "/")
                info = history.setdefault(path, {
                    "commit_count": 0, "bug_fix_count": 0, "latest_commit": timestamp,
                })
                info["commit_count"] += 1
                info["latest_commit"] = max(info["latest_commit"], timestamp)
                if any(word in subject.casefold() for word in ("fix", "bug", "issue", "patch", "error")):
                    info["bug_fix_count"] += 1
        return history
