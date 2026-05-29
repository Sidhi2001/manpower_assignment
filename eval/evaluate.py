"""
Basic evaluation script — run against any PDF to check answer quality.

Usage (single model):
    python eval/evaluate.py --pdf path/to/doc.pdf --questions eval/sample_questions.json

Usage (compare multiple models):
    python eval/evaluate.py --pdf path/to/doc.pdf --questions eval/sample_questions.json \
        --models llama-3.1-8b-instant llama-3.3-70b-versatile

Questions JSON can be a list of strings or a list of objects with keys:
    question, expected (optional), type (optional)

Score: 2 = fully correct, 1 = partially correct, 0 = wrong/hallucinated
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.pdf_processor import process_pdf
from modules.vector_store import VectorStore

load_dotenv()

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "of", "to",
    "in", "it", "that", "this", "with", "for", "on", "at", "by", "from",
    "as", "be", "been", "has", "have", "had", "its", "which", "also", "not",
    "can", "more", "than", "their", "they", "how", "what", "why", "does",
    "using", "used", "use", "because", "between", "during", "after", "being",
}

_SUFFIXES = ("ations", "ation", "ings", "ing", "ied", "ies", "ness", "ment", "ers", "ed", "ly", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _keywords(text: str) -> set:
    raw = re.findall(r"\d+(?:\.\d+)?|[a-z]+", text.lower())
    result = set()
    for t in raw:
        if re.match(r"\d", t):
            result.add(t)
        elif t not in STOPWORDS and len(t) > 2:
            result.add(_stem(t))
    return result


def _keyword_score(expected: str, actual: str) -> int:
    exp_kw = _keywords(expected)
    if not exp_kw:
        return 0
    ratio = len(exp_kw & _keywords(actual)) / len(exp_kw)
    if ratio >= 0.50:
        return 2
    if ratio >= 0.22:
        return 1
    return 0


def _normalise(item) -> dict:
    if isinstance(item, str):
        return {"question": item, "expected": None, "type": "—"}
    return {
        "question": item["question"],
        "expected": item.get("expected"),
        "type": item.get("type", "—"),
    }


def _fresh_store(chunks: list) -> VectorStore:
    """Build a clean VectorStore in a throwaway temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="eval_chroma_")
    vs = VectorStore(persist_dir=tmpdir)
    vs.add_chunks(chunks)
    return vs


def run_eval(chunks: list, raw_questions: list, model: str = None) -> list[dict]:
    import importlib
    import modules.rag_pipeline as rp
    if model:
        os.environ["LLM_MODEL"] = model
    importlib.reload(rp)
    from modules.rag_pipeline import RAGPipeline as _RAG

    print(f"  model: {model or os.getenv('LLM_MODEL', 'default')}")
    vs = _fresh_store(chunks)
    rag = _RAG(vs)

    results = []
    history = []
    for item in raw_questions:
        entry = _normalise(item)
        q = entry["question"]
        print(f"  Q: {q}")
        answer, citations = rag.query(q, history)
        answer = answer.replace("\n", " ").strip()
        pages = sorted({c["page"] for c in citations})
        score = _keyword_score(entry["expected"], answer) if entry["expected"] else "?"
        results.append({
            "question": q,
            "expected": entry["expected"] or "—",
            "type": entry["type"],
            "answer": answer,
            "cited_pages": pages,
            "score": score,
        })
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        time.sleep(2)  # avoid Groq free-tier rate limit

    return results


def _avg(results: list[dict]) -> float:
    scored = [r["score"] for r in results if isinstance(r["score"], int)]
    return sum(scored) / len(scored) if scored else 0.0


def print_markdown_table(results: list[dict]) -> None:
    avg = _avg(results)
    pct = round(avg / 2 * 100)
    print(f"\n## Evaluation Results  —  avg score {avg:.2f}/2  ({pct}% correct)\n")
    print("| # | Type | Question | Cited Pages | Score (0-2) |")
    print("|---|------|----------|-------------|-------------|")
    for i, r in enumerate(results, 1):
        pages = ", ".join(str(p) for p in r["cited_pages"]) or "—"
        q = r["question"][:55] + ("…" if len(r["question"]) > 55 else "")
        print(f"| {i} | {r['type']} | {q} | {pages} | {r['score']} |")
    print("\n**Score key:** 0 = wrong/hallucinated, 1 = partially correct, 2 = fully correct")
    print("\n---\n### Full Answers\n")
    for i, r in enumerate(results, 1):
        print(f"**Q{i} [{r['type']}]:** {r['question']}")
        print(f"**Expected:** {r['expected']}")
        print(f"**Got:** {r['answer'][:400]}")
        print(f"**Score:** {r['score']}/2  |  Pages: {', '.join(str(p) for p in r['cited_pages']) or '—'}")
        print()


def print_comparison_table(model_results: dict[str, list[dict]]) -> None:
    """Side-by-side score comparison across multiple models."""
    models = list(model_results.keys())
    questions = model_results[models[0]]

    print("\n## Model Comparison\n")

    # summary row
    print("| Model | Avg Score | % Correct | Score 2s | Score 1s | Score 0s |")
    print("|-------|-----------|-----------|----------|----------|----------|")
    for model, results in model_results.items():
        scored = [r["score"] for r in results if isinstance(r["score"], int)]
        avg = sum(scored) / len(scored) if scored else 0
        print(
            f"| {model} "
            f"| {avg:.2f}/2 "
            f"| {round(avg/2*100)}% "
            f"| {scored.count(2)} "
            f"| {scored.count(1)} "
            f"| {scored.count(0)} |"
        )

    # per-question breakdown
    header = "| # | Question | " + " | ".join(f"{m[:20]}" for m in models) + " |"
    sep    = "|---|----------|" + "|".join(["------"] * len(models)) + "|"
    print(f"\n### Per-question scores\n\n{header}\n{sep}")
    for i, entry in enumerate(questions, 1):
        q = entry["question"][:45] + ("…" if len(entry["question"]) > 45 else "")
        scores = " | ".join(str(model_results[m][i - 1]["score"]) for m in models)
        print(f"| {i} | {q} | {scores} |")


def save_csv(results: list[dict], path: str, model: str = "") -> None:
    fields = ["#", "model", "type", "question", "expected", "answer", "cited_pages", "score"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({
                "#": i,
                "model": model,
                "type": r["type"],
                "question": r["question"],
                "expected": r["expected"],
                "answer": r["answer"],
                "cited_pages": ", ".join(str(p) for p in r["cited_pages"]),
                "score": r["score"],
            })
    print(f"Results appended to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--questions", default=None, help="JSON file with question list")
    parser.add_argument("--output", default=None, help="CSV file to save results")
    parser.add_argument("--models", nargs="+", default=None,
                        help="One or more model names to compare (e.g. llama-3.1-8b-instant llama-3.3-70b-versatile)")
    args = parser.parse_args()

    if args.questions:
        with open(args.questions) as f:
            raw = json.load(f)
    else:
        raw = [
            "What is the main topic of this document?",
            "Who are the authors or contributors mentioned?",
            "What are the key findings or conclusions?",
            "What methodology or approach is described?",
            "What problem does this document address?",
        ]

    out_path = args.output or os.path.splitext(args.questions or "eval/results")[0] + "_results.csv"

    print(f"Processing PDF: {args.pdf}…")
    chunks = process_pdf(args.pdf)

    if args.models and len(args.models) > 1:
        model_results = {}
        for model in args.models:
            print(f"\n{'='*60}\nRunning: {model}\n{'='*60}")
            results = run_eval(chunks, raw, model=model)
            model_results[model] = results
            save_csv(results, out_path, model=model)

        print_comparison_table(model_results)
    else:
        model = (args.models or [None])[0]
        results = run_eval(chunks, raw, model=model)
        print_markdown_table(results)
        save_csv(results, out_path, model=model or os.getenv("LLM_MODEL", ""))


if __name__ == "__main__":
    main()
