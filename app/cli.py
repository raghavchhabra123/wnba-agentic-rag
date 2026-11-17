"""Interactive command-line interface.

    python -m app.cli
    python -m app.cli "Who led the Fever in assists against the Sky on June 7, 2025?"
"""

import sys

from wnba_rag.agent import ask, build_agent


def show(result: dict) -> None:
    print("\n" + result["answer"])
    tools = [f"{t['name']}({t['args']})" for t in result["trace"] if t.get("step") == "tool"]
    if tools:
        print("\n[tools] " + " -> ".join(tools))
    print()


def main() -> None:
    graph = build_agent()
    if len(sys.argv) > 1:
        show(ask(" ".join(sys.argv[1:]), graph=graph))
        return
    print("WNBA agentic RAG (2025 season). Name the teams and date in your question. 'exit' to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            show(ask(q, graph=graph))


if __name__ == "__main__":
    main()
