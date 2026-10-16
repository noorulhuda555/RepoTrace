import re
from pathlib import Path
from typing import List, Tuple, Set

from .models import CodeUnit
from .tokenizer import CodeTokenizer

_LANGUAGES = {
    ".java": "java", ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".go": "go", ".rs": "rust", ".cs": "c_sharp",
    ".kt": "kotlin", ".kts": "kotlin", ".cpp": "cpp", ".cc": "cpp", ".c": "c",
    ".h": "c", ".hpp": "cpp", ".php": "php", ".rb": "ruby", ".scala": "scala",
    ".swift": "swift", ".html": "html", ".xhtml": "html", ".jsp": "jsp",
    ".xml": "xml", ".yaml": "yaml", ".yml": "yaml", ".properties": "properties",
    ".sql": "sql"
}

_JAVA_TYPE = re.compile(r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)")
_JAVA_METHOD = re.compile(
    r"(?m)^[ \t]*(?:(?:public|protected|private|static|final|abstract|synchronized|"
    r"native|default|strictfp)\s+)*(?:<[^;{}]+>\s*)?"
    r"(?:[\w.$<>\[\], ?]+\s+)?([A-Za-z_$][\w$]*)\s*\([^;{}]*\)"
    r"\s*(?:throws\s+[\w.$, ]+\s*)?\{"
)
_METHOD_EXCLUSIONS = {"if", "for", "while", "switch", "catch", "synchronized", "try"}
_VIEW_RETURN_PATTERN = re.compile(r'return\s+["\']([a-zA-Z0-9_\-/]+)["\']|ModelAndView\(["\']([a-zA-Z0-9_\-/]+)["\']')


def _detect_file_role(path_text: str, content: str) -> Tuple[str, bool]:
    """Detects architectural role and whether the unit is a test file."""
    path_low = path_text.casefold()
    is_test = ("/test/" in path_low or "/tests/" in path_low or
               path_low.endswith("test.java") or path_low.endswith("tests.java") or
               path_low.endswith("_test.py") or path_low.endswith("test.py"))

    if path_low.endswith((".html", ".xhtml", ".jsp", ".vue")):
        return "view", is_test

    if path_low.endswith((".properties", ".yaml", ".yml", ".xml")):
        return "configuration", is_test

    if "@restcontroller" in content.casefold() or "@controller" in content.casefold() or "controller" in path_low:
        return "controller", is_test

    if "@repository" in content.casefold() or "repository" in path_low or "dao" in path_low:
        return "repository", is_test

    if "@service" in content.casefold() or "service" in path_low:
        return "service", is_test

    if "@entity" in content.casefold() or "/model/" in path_low or "/domain/" in path_low or "/entity/" in path_low:
        return "model", is_test

    if is_test:
        return "test", is_test

    return "general", is_test


class CodeStructuralParser:
    """Extract file, class, method, and view units with architectural role inference."""

    def __init__(self):
        self.tokenizer = CodeTokenizer()
        try:
            from tree_sitter_language_pack import get_parser
        except ImportError:
            self._get_parser = None
        else:
            self._get_parser = get_parser

    def parse_file(self, path, source=None):
        path = Path(path)
        if source is None:
            source = path.read_text(encoding="utf-8-sig", errors="replace")
        else:
            source = source.removeprefix("\ufeff")

        language = _LANGUAGES.get(path.suffix.lower(), "")
        path_text = path.as_posix()
        lines = source.splitlines()
        file_role, is_test = _detect_file_role(path_text, source)

        # File-level unit
        file_unit = CodeUnit(
            path=path_text,
            kind="view" if file_role == "view" else "file",
            name=path.name,
            start_line=1,
            end_line=max(1, len(lines)),
            content=source if file_role == "view" else source[:1500],
            language=language,
            symbols=[path.name],
            role=file_role,
            is_test=is_test,
        )

        units = [file_unit]
        if not source.strip():
            return units

        # HTML / View parser: extract template expressions and structural elements
        if file_role == "view" or language == "html":
            view_symbols = re.findall(r'th:[a-zA-Z]+="([^"]+)"|\${([^}]+)}|id="([^"]+)"|class="([^"]+)"', source)
            flattened = [item for sub in view_symbols for item in sub if item]
            file_unit.symbols.extend(flattened)
            return units

        # Tree-sitter for Java (download/network failures fall back to regex)
        if language == "java" and self._get_parser:
            try:
                parsed = self._tree_sitter_units(path_text, source, language, file_role, is_test)
            except Exception:
                parsed = []
            if parsed:
                return units + parsed

        # Regex fallback for Java
        if language == "java":
            units.extend(self._java_units(path_text, source, file_role, is_test))

        return units

    def _tree_sitter_units(self, path, source, language, file_role, is_test):
        parser = self._get_parser(language)
        source_bytes = source.encode("utf-8")
        root = parser.parse(source_bytes).root_node
        units = []
        class_types = {
            "class_declaration", "interface_declaration", "enum_declaration",
            "record_declaration", "annotation_type_declaration",
        }
        method_types = {"method_declaration", "constructor_declaration"}

        def visit(node, parent="", parent_role=file_role):
            child_parent = parent
            current_role = parent_role

            if node.type in class_types:
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
                    content = source_bytes[
                        node.start_byte:min(node.end_byte, node.start_byte + 1500)
                    ].decode("utf-8", "replace")

                    # Refine class role
                    class_role = file_role
                    if name.endswith("Controller") or "@Controller" in content:
                        class_role = "controller"
                    elif name.endswith("Repository") or name.endswith("Dao") or "@Repository" in content:
                        class_role = "repository"
                    elif name.endswith("Service") or "@Service" in content:
                        class_role = "service"

                    units.append(CodeUnit(
                        path=path, kind="class", name=name,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content=content, parent=parent, language=language,
                        symbols=[name], role=class_role, is_test=is_test,
                    ))
                    child_parent = ".".join(part for part in (parent, name) if part)
                    current_role = class_role

            elif node.type in method_types:
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
                    content = source_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")
                    symbols = self.tokenizer.identifiers(content)

                    # Extract view names if method is inside controller
                    for v_match in _VIEW_RETURN_PATTERN.finditer(content):
                        v_name = v_match.group(1) or v_match.group(2)
                        if v_name:
                            symbols.append(v_name)

                    units.append(CodeUnit(
                        path=path, kind="method", name=name,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content=content, parent=parent, language=language,
                        symbols=symbols, role=current_role, is_test=is_test,
                    ))

            for child in node.children:
                visit(child, child_parent, current_role)

        visit(root)
        return units

    def _java_units(self, path, source, file_role, is_test):
        units = []
        classes = []

        for match in _JAVA_TYPE.finditer(source):
            opening = source.find("{", match.end())
            if opening < 0:
                continue
            closing = self._matching_brace(source, opening)
            if closing < 0:
                continue
            name = match.group(1)

            class_role = file_role
            if name.endswith("Controller"):
                class_role = "controller"
            elif name.endswith("Repository") or name.endswith("Dao"):
                class_role = "repository"
            elif name.endswith("Service"):
                class_role = "service"

            classes.append((opening, closing, name, class_role))
            content = source[match.start():min(closing + 1, match.start() + 1500)]
            units.append(CodeUnit(
                path=path, kind="class", name=name,
                start_line=source.count("\n", 0, match.start()) + 1,
                end_line=source.count("\n", 0, closing + 1) + 1,
                content=content, parent="", language="java",
                symbols=[name], role=class_role, is_test=is_test,
            ))

        for match in _JAVA_METHOD.finditer(source):
            name = match.group(1)
            if name in _METHOD_EXCLUSIONS:
                continue
            opening = source.find("{", match.start(), match.end())
            closing = self._matching_brace(source, opening) if opening >= 0 else -1
            if closing < 0:
                continue
            enclosing = [item for item in classes if item[0] < opening < item[1]]
            parent = min(enclosing, key=lambda item: item[1] - item[0])[2] if enclosing else ""
            method_role = min(enclosing, key=lambda item: item[1] - item[0])[3] if enclosing else file_role
            content = source[match.start():closing + 1]

            symbols = self.tokenizer.identifiers(content)
            for v_match in _VIEW_RETURN_PATTERN.finditer(content):
                v_name = v_match.group(1) or v_match.group(2)
                if v_name:
                    symbols.append(v_name)

            units.append(CodeUnit(
                path=path, kind="method", name=name,
                start_line=source.count("\n", 0, match.start()) + 1,
                end_line=source.count("\n", 0, closing + 1) + 1,
                content=content, parent=parent, language="java",
                symbols=symbols, role=method_role, is_test=is_test,
            ))

        return units

    @staticmethod
    def _matching_brace(source, opening):
        depth = 0
        quote = ""
        escaped = False
        line_comment = False
        block_comment = False
        index = opening
        while index < len(source):
            char = source[index]
            following = source[index + 1] if index + 1 < len(source) else ""
            if line_comment:
                if char == "\n":
                    line_comment = False
            elif block_comment:
                if char == "*" and following == "/":
                    block_comment = False
                    index += 1
            elif quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = ""
            elif char == "/" and following == "/":
                line_comment = True
                index += 1
            elif char == "/" and following == "*":
                block_comment = True
                index += 1
            elif char in ("'", '"'):
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
            index += 1
        return -1
