"""
RAG pipeline: retrieve → generate → suggest follow-ups.

LLM     : llama-3.1-8b-instant via Groq (OpenAI-compatible)
Retrieve: top-5 cosine similarity from ChromaDB, deduplicated by page
History : last 6 messages injected before the context-augmented turn
"""

import os
from typing import List, Dict, Tuple

import openai

from .vector_store import VectorStore


LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE = "https://api.groq.com/openai/v1"
N_CHUNKS = 5
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """\
You are a precise assistant that answers questions exclusively from the \
provided PDF document excerpts.

Rules:
1. Base every claim on the supplied context. If the answer is not there, \
say "I could not find this in the document."
2. Write a clear, well-structured answer — prose or concise bullets as appropriate.
3. At the end of your answer add ONE line: "Sources: Page X, Page Y" \
listing only the page numbers you drew from. Do not put page refs mid-sentence.
4. For follow-up questions, use the conversation history to resolve \
pronouns and references.\
"""

SUGGEST_PROMPT = """\
You generate short follow-up questions about a document.
Output EXACTLY 3 questions, one per line, no numbering, no bullets, no extra text.\
"""


def _dedupe_citations(chunks: List[Dict]) -> List[Dict]:
    """Keep one entry per page (highest relevance score), sorted by page number."""
    best: Dict[int, Dict] = {}
    for c in chunks:
        pg = c["page"]
        if pg not in best or c["score"] > best[pg]["score"]:
            best[pg] = c
    return sorted(best.values(), key=lambda x: x["page"])


class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self._vs = vector_store
        self._oai = openai.OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=GROQ_BASE,
        )

    def query(self, question: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
        chunks = self._vs.query(question, n_results=N_CHUNKS)
        context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[-MAX_HISTORY_TURNS:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({
            "role": "user",
            "content": f"--- Document excerpts ---\n{context}\n\n--- Question ---\n{question}",
        })

        response = self._oai.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content

        # Deduplicate: one card per page, best-scoring chunk wins
        citations = [
            {
                "page": c["page"],
                "snippet": c["text"][:300].replace("\n", " ") + "…",
                "score": c["score"],
            }
            for c in _dedupe_citations(chunks)
        ]
        return answer, citations

    def suggest_questions(self, last_answer: str, history: List[Dict]) -> List[str]:
        """Return 3 follow-up question strings (empty list on failure)."""
        try:
            recent = "\n".join(
                f"{m['role'].upper()}: {m['content'][:200]}"
                for m in history[-4:]
                if m["role"] in ("user", "assistant")
            )
            response = self._oai.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SUGGEST_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Conversation so far:\n{recent}\n\n"
                            f"Last answer: {last_answer[:500]}\n\n"
                            "Generate 3 follow-up questions:"
                        ),
                    },
                ],
                temperature=0.8,
                max_tokens=120,
            )
            raw = response.choices[0].message.content.strip()
            return [q.strip() for q in raw.split("\n") if q.strip()][:3]
        except Exception:
            return []
