import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core import HybridCodeIntelligenceEngine

def test_engine():
    petclinic_path = Path(r"C:\Users\Labaik Laptop\Desktop\FYP\spring-petclinic-main")
    if not petclinic_path.is_dir():
        print(f"Repo path {petclinic_path} not found for automated test.")
        return

    print("Initializing HybridCodeIntelligenceEngine...")
    engine = HybridCodeIntelligenceEngine(petclinic_path)
    summary = engine.index_repository()
    print(f"Indexed {summary['files']} files and {summary['units']} units.")

    issue = {
        "title": "Veterinarian list is not displayed",
        "body": "Find the code that loads and displays the veterinarians. Identify the related controller, service, and view."
    }

    result = engine.analyze_issue(issue, skills=["Java", "Spring"], top_k=5)
    
    print("\n--- ARCHITECTURAL LAYERS IDENTIFIED ---")
    layers = result.get("layers", {})
    for layer_name in ("controller", "view", "service", "repository", "model"):
        val = layers.get(layer_name)
        if val:
            print(f"[{layer_name.upper()}] {val['path']}:{val['start_line']} {val['name']} (score: {val['score']:.3f})")

    print("\n--- TOP RANKED MATCHES ---")
    for i, match in enumerate(result["results"], 1):
        print(f"{i}. {match['path']}:{match['start_line']} [{match.get('role', match['kind'])}] {match['name']} (score: {match['score']:.3f})")
        print(f"   Why: {match['reason']}")

if __name__ == "__main__":
    test_engine()
