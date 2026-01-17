from __future__ import annotations

import argparse
from pathlib import Path

from .graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LangGraph visualization")
    parser.add_argument("--out", default="docs/langgraph.mmd", help="Output Mermaid file path")
    args = parser.parse_args()

    graph = build_graph().compile()
    mermaid = graph.get_graph().draw_mermaid()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mermaid, encoding="utf-8")

    print(f"Mermaid graph written to: {out_path}")


if __name__ == "__main__":
    main()
