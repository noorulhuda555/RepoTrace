import argparse
from pathlib import Path
import sys

from core import HybridCodeIntelligenceEngine


def _read_issue_body():
    print("Issue details (press Enter on an empty line to finish):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Find the files, classes, and methods most relevant to an issue."
    )
    parser.add_argument(
        "--repository",
        help="Path to the source-code repository to index (prompted if omitted).",
    )
    parser.add_argument("--title", help="Issue title (prompted if omitted).")
    parser.add_argument(
        "--body",
        help="Issue description. Use quotes around text containing spaces "
             "(prompted if omitted).",
    )
    parser.add_argument(
        "--skills",
        nargs="*",
        help="Optional predicted skill names, for example --skills Java authentication.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Maximum number of results (default: 10)."
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be a positive integer.")
    return args


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = _parse_arguments()
    repository = args.repository or input(
        "Path to the repository you want to scan: "
    ).strip()
    repository_path = Path(repository).expanduser()
    if not repository_path.is_dir():
        raise SystemExit("Repository folder does not exist: {}".format(repository_path))

    title = args.title if args.title is not None else input("Issue title: ").strip()
    body = args.body if args.body is not None else _read_issue_body()
    if not title and not body:
        raise SystemExit("Enter an issue title or description.")
    skills = args.skills
    if skills is None:
        raw_skills = input("Predicted skills (comma-separated, optional): ").strip()
        skills = [skill.strip() for skill in raw_skills.split(",") if skill.strip()]

    engine = HybridCodeIntelligenceEngine(repository_path)
    index_summary = engine.index_repository()
    print(
        "Indexed {} source files and {} code units.".format(
            index_summary["files"], index_summary["units"]
        )
    )
    result = engine.analyze_issue(
        {"title": title, "body": body}, skills=skills, top_k=args.top_k
    )
    if not result["results"]:
        print("No relevant code locations found for this issue.")
        return

    # Architectural Layer Identification Summary (FYP Core Feature)
    layers = result.get("layers", {})
    has_layers = any(layers.get(k) for k in ("controller", "service", "repository", "view", "model"))
    if has_layers:
        print("\n" + "=" * 80)
        print("   CODE INTELLIGENCE: ARCHITECTURAL LAYER MAPPING")
        print("=" * 80)
        if layers.get("controller"):
            c = layers["controller"]
            print(f"   [Controller]        {c['path']}:{c['start_line']} [{c['kind']}] {c['name']} (score: {c['score']:.3f})")
        if layers.get("view"):
            v = layers["view"]
            print(f"   [View / Template]   {v['path']}:{v['start_line']} [{v['kind']}] {v['name']} (score: {v['score']:.3f})")
        if layers.get("service"):
            s = layers["service"]
            print(f"   [Service]           {s['path']}:{s['start_line']} [{s['kind']}] {s['name']} (score: {s['score']:.3f})")
        if layers.get("repository"):
            r = layers["repository"]
            print(f"   [Repository / DAO]  {r['path']}:{r['start_line']} [{r['kind']}] {r['name']} (score: {r['score']:.3f})")
        if layers.get("model"):
            m = layers["model"]
            print(f"   [Domain Model]      {m['path']}:{m['start_line']} [{m['kind']}] {m['name']} (score: {m['score']:.3f})")
        print("=" * 80 + "\n")

    print("RANKED SUSPICIOUS CODE LOCATIONS:")
    for index, match in enumerate(result["results"], start=1):
        role_label = f"[{match.get('role', match['kind'])}]" if match.get('role') and match.get('role') != 'general' else f"[{match['kind']}]"
        print(
            "\n{index}. {path}:{line}  {role} {name}  (score: {score:.3f})".format(
                index=index,
                path=match["path"],
                line=match["start_line"],
                role=role_label,
                name=match["name"],
                score=match["score"],
            )
        )
        print("   Why: {}".format(match["reason"]))
        if match["snippet"]:
            print("   {}".format(match["snippet"]))

    # Optional verification tests section
    tests = layers.get("tests", [])
    if tests:
        print("\n" + "-" * 80)
        print("RELATED VERIFICATION TESTS:")
        for t in tests[:3]:
            print(f"   * {t['path']}:{t['start_line']} [{t['name']}] (score: {t['score']:.3f})")
        print("-" * 80)


if __name__ == "__main__":
    main()
