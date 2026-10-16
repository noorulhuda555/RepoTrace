import re
from typing import List, Dict, Set, Tuple

_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|[0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_COMMENTS = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/|#[^\n]*")
_STRINGS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`[^`]*`', re.DOTALL)

# Domain Synonyms & Morphological Root Expansions
SYNONYM_MAP = {
    "veterinarian": ["vet", "vets", "veterinary"],
    "veterinarians": ["vet", "vets", "veterinarian", "veterinary"],
    "veterinary": ["vet", "vets", "veterinarian"],
    "vet": ["veterinarian", "veterinarians", "vets"],
    "vets": ["veterinarian", "veterinarians", "vet"],
    "controller": ["ctrl", "endpoint", "handler", "restcontroller"],
    "controllers": ["controller", "ctrl", "endpoint", "handler"],
    "repository": ["repo", "dao", "service", "store"],
    "repositories": ["repository", "repo", "dao", "store"],
    "repo": ["repository", "dao", "store"],
    "service": ["svc", "repository", "manager", "logic"],
    "services": ["service", "svc", "repository"],
    "view": ["template", "html", "jsp", "ui", "page", "display"],
    "views": ["view", "template", "html", "ui", "page"],
    "display": ["show", "view", "render", "list", "html"],
    "displays": ["display", "show", "view", "render", "list"],
    "displayed": ["display", "show", "view", "render", "list"],
    "load": ["find", "fetch", "get", "query", "retrieve"],
    "loads": ["load", "find", "fetch", "get", "retrieve"],
    "loading": ["load", "find", "fetch", "get"],
    "auth": ["authenticate", "authentication", "login", "jwt", "token"],
    "authentication": ["auth", "login", "jwt", "token"],
    "database": ["db", "sql", "repository", "jpa", "hibernate"],
    "config": ["configuration", "properties", "settings"],
    "configuration": ["config", "properties", "settings"],
}

# Generic query instructional noise words that should not overpower domain nouns
META_INSTRUCTION_WORDS = {
    "find", "code", "that", "the", "a", "an", "and", "or", "to", "for",
    "with", "in", "on", "at", "by", "from", "of", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "identify", "related", "which", "where", "how", "what", "why",
    "this", "that", "these", "those", "all", "any", "some", "can", "could",
    "should", "would", "must", "please", "help", "need", "want", "issue",
    "problem", "bug", "error", "not", "no"
}

# Architecture role keywords
ARCHITECTURE_ROLE_WORDS = {
    "controller": "controller",
    "controllers": "controller",
    "service": "service",
    "services": "service",
    "repository": "repository",
    "repositories": "repository",
    "repo": "repository",
    "view": "view",
    "views": "view",
    "template": "view",
    "templates": "view",
    "html": "view",
    "model": "model",
    "models": "model",
    "entity": "model",
    "entities": "model",
}


def _stem_word(word: str) -> str:
    """Lightweight rule-based English stemmer for common plurals and suffixes."""
    w = word.casefold()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return w


class CodeTokenizer:
    """Tokenize issue text and source identifiers with domain expansion and stemming."""

    def tokenize(self, text, include_comments=True, include_strings=True, expand_synonyms=False):
        if not text:
            return []
        source = text
        if not include_comments:
            source = _COMMENTS.sub(" ", source)
        if not include_strings:
            source = _STRINGS.sub(" ", source)
        result = []
        for token in _TOKEN.findall(source):
            pieces = _CAMEL_BOUNDARY.sub(" ", token).replace("_", " ").replace("$", " ").split()
            for piece in pieces:
                word = piece.casefold()
                if len(word) > 1 or word.isdigit():
                    result.append(word)
                    stemmed = _stem_word(word)
                    if stemmed != word and len(stemmed) > 1:
                        result.append(stemmed)
                    if expand_synonyms and word in SYNONYM_MAP:
                        result.extend(SYNONYM_MAP[word])
        return result

    def tokenize_query(self, query_text: str) -> List[str]:
        """
        Specialized tokenizer for issue queries:
        Expands domain synonyms (e.g., 'veterinarian' -> 'vet', 'vets')
        and weights domain entities higher than instructional stopwords.
        """
        if not query_text:
            return []
        tokens = self.tokenize(query_text, expand_synonyms=True)
        weighted_tokens = []
        for token in tokens:
            t = token.casefold()
            if t in META_INSTRUCTION_WORDS:
                # Add only once (low weight)
                weighted_tokens.append(t)
            elif t in ARCHITECTURE_ROLE_WORDS:
                # Add twice (moderate importance)
                weighted_tokens.extend([t, ARCHITECTURE_ROLE_WORDS[t]])
            else:
                # Domain entity (e.g. 'veterinarian', 'vet', 'owner', 'pet'): Boost!
                weighted_tokens.extend([t, t, t])
        return weighted_tokens

    def extract_role_intent(self, query_text: str) -> Set[str]:
        """Detects if user is asking for specific architecture layers (controller, service, view)."""
        roles = set()
        words = set(query_text.casefold().split())
        for word in words:
            clean = re.sub(r"[^\w]", "", word)
            if clean in ARCHITECTURE_ROLE_WORDS:
                roles.add(ARCHITECTURE_ROLE_WORDS[clean])
        return roles

    def identifiers(self, text):
        return list(dict.fromkeys(re.findall(
            r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*", text or ""
        )))
