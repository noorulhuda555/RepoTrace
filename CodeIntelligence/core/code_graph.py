from collections import defaultdict
from typing import List, Dict, Set

from .models import CodeUnit
from .tokenizer import CodeTokenizer


class DependencyGraphMiner:
    """Index declared, referenced, and architectural dependencies across repository code units."""

    def __init__(self):
        self.tokenizer = CodeTokenizer()
        self.symbol_to_units = defaultdict(set)
        self.unit_symbols = []
        self.adjacency = defaultdict(set)

    def fit(self, units: List[CodeUnit]):
        self.symbol_to_units.clear()
        self.unit_symbols = []
        self.adjacency.clear()

        # Step 1: Index declared symbols and names
        for index, unit in enumerate(units):
            symbols = set(unit.symbols)
            symbols.add(unit.name)
            if unit.parent:
                symbols.add(unit.parent)

            normalized = {
                symbol.rsplit(".", 1)[-1].casefold()
                for symbol in symbols if len(symbol) > 1
            }
            # Also add path components
            path_parts = unit.path.replace("\\", "/").split("/")
            for part in path_parts:
                clean_part = part.split(".")[0].casefold()
                if len(clean_part) > 2:
                    normalized.add(clean_part)

            self.unit_symbols.append(normalized)
            for symbol in normalized:
                self.symbol_to_units[symbol].add(index)

        # Step 2: Build cross-layer architectural links (Controller -> View, Controller -> Repository)
        for i, unit in enumerate(units):
            # Check for view template links
            for sym in unit.symbols:
                s_low = sym.casefold().strip("/")
                for j, other in enumerate(units):
                    if i != j:
                        other_path_low = other.path.replace("\\", "/").casefold()
                        if (s_low in other_path_low or
                            (s_low.endswith("list") and other_path_low.endswith("list.html"))):
                            self.adjacency[i].add(j)
                            self.adjacency[j].add(i)

        return self

    def scores(self, units: List[CodeUnit], query: str) -> Dict[int, float]:
        query_terms = self.tokenizer.tokenize_query(query)
        query_symbols = set(query_terms)
        for ident in self.tokenizer.identifiers(query):
            query_symbols.add(ident.rsplit(".", 1)[-1].casefold())

        raw_scores = defaultdict(float)

        for symbol in query_symbols:
            for index in self.symbol_to_units.get(symbol, ()):
                unit = units[index]
                # High priority match if exact name matches
                if unit.name.casefold() == symbol:
                    raw_scores[index] += 3.0
                else:
                    raw_scores[index] += 1.0

        # Propagate structural spreading activation
        propagated = defaultdict(float)
        for idx, base_score in raw_scores.items():
            propagated[idx] += base_score
            if base_score > 1.5:
                # Share activation with connected architectural units
                for neighbor in self.adjacency.get(idx, ()):
                    propagated[neighbor] += base_score * 0.4

        maximum = max(propagated.values(), default=0)
        return {index: value / maximum for index, value in propagated.items()} if maximum else {}
