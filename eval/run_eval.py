"""Run systems over a benchmark, grade them, and write a summary.

Results are appended to eval/results_<bench>_<split>/<system>.jsonl (v1 keeps its
original eval/results_<split>/ folders), so an interrupted run resumes where it stopped.

Usage:
    python -m eval.run_eval --bench v1 --split test
    python -m eval.run_eval --bench v2 --split test
    python -m eval.run_eval --bench v2 --split dev --systems agentic --limit 20
    python -m eval.run_eval --bench v2 --split test --summary-only
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from eval.grade import grade
from wnba_rag.paths import EVAL_DIR

ORDER = ["llm_only", "naive_rag", "scoped_rag", "agentic"]


def paths(bench: str, split: str, tag: str = ""):
    suffix = f"_{tag}" if tag else ""
    if bench == "v1":
        return EVAL_DIR / f"benchmark_{split}.json", EVAL_DIR / f"results_{split}{suffix}"
    return EVAL_DIR / f"benchmark_{bench}_{split}.json", EVAL_DIR / f"results_{bench}_{split}{suffix}"


def run_system(name: str, items: list[dict], workers: int, results_dir) -> None:
    from wnba_rag.baselines import SYSTEMS

    fn = SYSTEMS[name]
    path = results_dir / f"{name}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["qid"] for line in path.read_text().splitlines() if line.strip()}
    todo = [it for it in items if it["qid"] not in done]
    print(f"{name}: {len(todo)} to run ({len(done)} already done)", flush=True)

    def one(item: dict) -> dict:
        start = time.time()
        try:
            out = fn(item["question"])
        except Exception as exc:
            out = {"answer": f"ERROR: {exc}", "trace": []}
        return {"qid": item["qid"], "type": item["type"], "question": item["question"],
                "answer": out.get("answer", ""), "resolved_game_id": out.get("game_id"),
                "mode": out.get("mode"), "trace": out.get("trace", []),
                "usage": out.get("usage", {}), "latency_s": round(time.time() - start, 2)}

    with path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, it) for it in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            f.write(json.dumps(fut.result(), default=str) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {name}: {i}/{len(todo)}", flush=True)


def _pct_table(df: pd.DataFrame, index: str, systems: list[str], title: str) -> list[str]:
    piv = df.pivot_table(index=index, columns="system", values="pass", aggfunc="mean")
    cnt = df[df["system"] == systems[-1]].groupby(index).size()
    lines = ["", f"## {title}", "", f"| {index} | n | " + " | ".join(systems) + " |",
             "|---|---:|" + "---:|" * len(systems)]
    for key, row in piv.iterrows():
        lines.append(f"| {key} | {cnt.get(key, 0)} | "
                     + " | ".join(f"{row[s]:.0%}" if s in row and pd.notna(row[s]) else "-" for s in systems) + " |")
    return lines


def summarize(items: list[dict], results_dir, title: str) -> str:
    by_qid = {it["qid"]: it for it in items}
    rows = []
    for name in ORDER:
        path = results_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["qid"] not in by_qid:
                continue
            item = by_qid[r["qid"]]
            g = grade(item, r["answer"])
            tools = [t["name"] for t in r["trace"] if t.get("step") == "tool"]
            gid = item.get("game_id")
            rows.append({
                "system": name, "qid": r["qid"], "type": r["type"],
                "category": item.get("category", "single_game"), "pass": g["pass"],
                "game_ok": (r.get("resolved_game_id") == gid)
                if (gid is not None and r.get("resolved_game_id") is not None) else None,
                "n_tools": len(tools), "latency_s": r["latency_s"],
                "cost_usd": (r.get("usage") or {}).get("cost_usd"),
                "tokens": ((r.get("usage") or {}).get("input_tokens", 0)
                           + (r.get("usage") or {}).get("output_tokens", 0)) or None,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return "No results yet."
    df.to_csv(results_dir / "graded.csv", index=False)

    systems = [s for s in ORDER if s in set(df["system"])]
    overall = df.groupby("system")["pass"].agg(["mean", "sum", "count"]).loc[systems]
    lat = df.groupby("system")["latency_s"].median().loc[systems]
    cost = df.groupby("system")["cost_usd"].mean()

    lines = [f"# Evaluation results: {title}", "",
             f"{df['qid'].nunique()} questions; deterministic grading (see eval/grade.py). "
             "Model: gpt-4o-mini for every system.", "",
             "| System | Accuracy | Correct | Median latency (s) | Avg cost / question |",
             "|---|---:|---:|---:|---:|"]
    for s in systems:
        c = cost.get(s)
        cost_s = f"${c:.5f}" if pd.notna(c) else "-"
        lines.append(f"| {s} | {overall.loc[s, 'mean']:.1%} | {int(overall.loc[s, 'sum'])}/"
                     f"{int(overall.loc[s, 'count'])} | {lat.loc[s]:.1f} | {cost_s} |")
    if df["category"].nunique() > 1:
        lines += _pct_table(df, "category", systems, "Accuracy by category")
    lines += _pct_table(df, "type", systems, "Accuracy by question type")
    resolved = df[df["game_ok"].notna()]
    if not resolved.empty:
        lines += ["", "## Game resolution (single-game questions)", ""]
        for s, grp in resolved.groupby("system"):
            n_all = ((df["system"] == s) & df["qid"].map(lambda q: by_qid[q].get("game_id") is not None)).sum()
            lines.append(f"- {s}: resolved the correct game for {int(grp['game_ok'].sum())}/{n_all} questions")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["v1", "v2"], default="v1")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--systems", nargs="+", default=ORDER)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--tag", default="", help="separate results folder, e.g. run2 for repeat runs")
    args = ap.parse_args()

    bench_path, results_dir = paths(args.bench, args.split, args.tag)
    items = json.loads(bench_path.read_text())
    if args.limit:
        items = items[: args.limit]
    results_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for name in args.systems:
            run_system(name, items, args.workers, results_dir)
    report = summarize(items, results_dir, f"{args.bench} {args.split} split")
    (results_dir / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
