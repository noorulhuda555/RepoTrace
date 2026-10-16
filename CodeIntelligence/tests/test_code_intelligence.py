import tempfile
import unittest
from pathlib import Path

from core import (
    BM25InformationRetrievalEngine,
    CodeStructuralParser,
    HybridCodeIntelligenceEngine,
    StackTraceMiner,
)


JAVA_SOURCE = """package example;
public class TokenManager {
    public void refreshToken() {
        throw new IllegalStateException("expired token");
    }
}
"""


class CodeIntelligenceTests(unittest.TestCase):
    def test_java_parser_extracts_file_class_and_method(self):
        units = CodeStructuralParser().parse_file(Path("src/TokenManager.java"), JAVA_SOURCE)
        self.assertEqual([unit.kind for unit in units], ["file", "class", "method"])
        self.assertEqual(units[1].name, "TokenManager")
        self.assertEqual(units[2].name, "refreshToken")
        self.assertEqual(units[2].parent, "TokenManager")

    def test_stack_trace_matches_the_exact_method_and_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "TokenManager.java")
            source.write_text(JAVA_SOURCE, encoding="utf-8")
            engine = HybridCodeIntelligenceEngine(directory)
            engine.index_repository()
            result = engine.analyze_issue(
                "Token refresh failed\n"
                "at example.TokenManager.refreshToken(TokenManager.java:4)",
                top_k=5,
            )
        self.assertTrue(result["results"])
        self.assertEqual(result["results"][0]["name"], "refreshToken")
        self.assertEqual(result["results"][0]["start_line"], 3)
        self.assertGreater(result["results"][0]["signals"]["stack_trace"], 0)

    def test_incremental_index_picks_up_new_and_removed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "First.java")
            first.write_text("class First {\n    void first() {}\n}\n", encoding="utf-8")
            engine = HybridCodeIntelligenceEngine(directory)
            initial = engine.index_repository()
            second = Path(directory, "Second.java")
            second.write_text("class Second {\n    void second() {}\n}\n", encoding="utf-8")
            first.unlink()
            update = engine.update_repository_index()
        self.assertEqual(initial["files"], 1)
        self.assertEqual(update["updated_files"], 1)
        self.assertEqual(update["removed_files"], 1)
        self.assertEqual(update["units"], 3)

    def test_bm25_returns_matching_document(self):
        engine = BM25InformationRetrievalEngine().fit([
            "TokenManager refreshToken expired authentication",
            "Unrelated rendering component",
        ])
        results = engine.search("expired token refresh", top_k=1)
        self.assertEqual(results[0][0], 0)

    def test_stack_trace_parses_python_frames(self):
        frames = StackTraceMiner().extract_frames(
            'File "src/auth.py", line 42, in refresh_token'
        )
        self.assertEqual(frames[0]["path"], "src/auth.py")
        self.assertEqual(frames[0]["line"], 42)
        self.assertEqual(frames[0]["symbol"], "refresh_token")


if __name__ == "__main__":
    unittest.main()
