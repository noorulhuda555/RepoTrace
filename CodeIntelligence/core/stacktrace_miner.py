import re


_JAVA_FRAME = re.compile(r"^\s*at\s+([\w.$]+)\(([^():]+):(\d+)\)\s*$", re.MULTILINE)
_PYTHON_FRAME = re.compile(
    r'^\s*File\s+["\'](.+?)["\'],\s+line\s+(\d+)(?:,\s+in\s+(.+))?', re.MULTILINE
)
_JS_FRAME = re.compile(
    r"^\s*at\s+(?:(.*?)\s+\()?(.+?\.(?:js|jsx|ts|tsx)):(\d+):\d+\)?\s*$", re.MULTILINE
)


class StackTraceMiner:
    def extract_frames(self, text):
        frames = []
        for match in _JAVA_FRAME.finditer(text or ""):
            frames.append({
                "symbol": match.group(1),
                "path": match.group(2).replace("\\", "/"),
                "line": int(match.group(3)),
                "weight": max(0.2, 1.0 - 0.15 * len(frames)),
            })
        python_matches = list(_PYTHON_FRAME.finditer(text or ""))
        python_matches.reverse()
        for match in python_matches:
            frames.append({
                "symbol": match.group(3) or "",
                "path": match.group(1).replace("\\", "/"),
                "line": int(match.group(2)),
                "weight": max(0.2, 1.0 - 0.15 * len(frames)),
            })
        if not frames:
            for match in _JS_FRAME.finditer(text or ""):
                frames.append({
                    "symbol": match.group(1) or "",
                    "path": match.group(2).replace("\\", "/"),
                    "line": int(match.group(3)),
                    "weight": max(0.2, 1.0 - 0.15 * len(frames)),
                })
        return frames

    def parse(self, text):
        return self.extract_frames(text)

    def score(self, unit, frames):
        best = 0.0
        normalized_path = unit.path.replace("\\", "/").casefold()
        for frame in frames:
            trace_path = frame["path"].casefold()
            path_match = normalized_path.endswith(trace_path) or trace_path.endswith(normalized_path)
            symbol = frame["symbol"].rsplit(".", 1)[-1].casefold()
            # Prefer an exact method/class name match. File units always contain
            # nested symbols in their text, so do not credit that weak match.
            if symbol and unit.name.casefold() == symbol:
                symbol_score = 0.30
            elif (
                symbol
                and unit.kind != "file"
                and symbol in unit.content.casefold()
            ):
                symbol_score = 0.10
            else:
                symbol_score = 0.0
            line_match = unit.start_line <= frame["line"] <= unit.end_line
            weight = frame.get("weight", 1.0)
            score = weight * (
                (0.55 if path_match else 0)
                + symbol_score
                + (0.15 if line_match else 0)
            )
            best = max(best, score)
        return best
