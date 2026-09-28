"""Avaliação da busca e das respostas (módulo 03).

1. Recuperação. Antes de avaliar respostas, avaliamos a BUSCA: se o trecho certo
   não chega ao modelo, nenhum prompt salva a resposta. O conjunto
   `data/eval/questions.json` traz, para cada pergunta, as fontes esperadas.
   Métricas por pergunta (nos `k` primeiros trechos):
   - hit: 1 se pelo menos uma fonte esperada aparece;
   - rr (reciprocal rank): 1 / posição da primeira fonte esperada (0 se não aparece);
   - coverage: fração das fontes esperadas que aparecem (importa nas multi_document).
   Médias: hit rate, MRR (média dos rr) e cobertura média. Perguntas `no_answer`
   ficam de fora: elas avaliam a GERAÇÃO (admitir que não sabe), não a busca.

2. Respostas, com um LLM como juiz:
   - grounded: tudo o que a resposta afirma está no contexto? (alucinação)
   - correct: concorda com a referência? (nas no_answer, correto é admitir que não sabe)
   - complete: cobre o essencial da referência?
   O juiz também é um modelo e erra: use-o para comparar versões no mesmo
   conjunto e revise à mão uma amostra dos julgamentos.

Uso:
    from src.evaluation import evaluate_retrieval, judge_answer
    table, summary = evaluate_retrieval(lambda q: retriever.search(q, mode="vector"))
    judge_answer(question, answer, context, reference)
"""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.config import settings
from src.models import get_llm, token_usage
from src.prompts import JUDGE_PROMPT

# ---------------------------------------------------------------------------
# 1. Recuperação
# ---------------------------------------------------------------------------


def load_questions(path: str | Path | None = None) -> list[dict]:
    return json.loads(Path(path or settings.eval_questions_path).read_text(encoding="utf-8"))


def retrieval_metrics(retrieved: list[str], expected: list[str]) -> dict:
    expected_set = set(expected)
    positions = [i for i, source in enumerate(retrieved, start=1) if source in expected_set]
    return {
        "hit": int(bool(positions)),
        "rr": 1 / positions[0] if positions else 0.0,
        "coverage": len(expected_set & set(retrieved)) / len(expected_set) if expected_set else 0.0,
    }


def evaluate_retrieval(
    search: Callable[[str], list[Document]],
    questions: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Roda a busca em cada pergunta com fonte esperada e calcula as métricas."""
    questions = [q for q in (questions or load_questions()) if q["expected_sources"]]
    rows = []
    for item in questions:
        sources = [c.metadata.get("source") for c in search(item["question"])]
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                **retrieval_metrics(sources, item["expected_sources"]),
                "retrieved": sources,
                "expected": item["expected_sources"],
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "questions": len(table),
        "hit_rate": float(round(table["hit"].mean(), 3)),
        "mrr": float(round(table["rr"].mean(), 3)),
        "coverage": float(round(table["coverage"].mean(), 3)),
    }
    return table, summary


# ---------------------------------------------------------------------------
# 2. Respostas (LLM como juiz)
# ---------------------------------------------------------------------------


class Judgment(BaseModel):
    reasoning: str = Field(description="Duas frases explicando os julgamentos.")
    grounded: bool = Field(description="Todas as afirmações estão apoiadas no contexto.")
    correct: bool = Field(description="Concorda com a resposta de referência.")
    complete: bool = Field(description="Cobre o essencial da referência.")


def judge_answer(question: str, answer: str, context: str, reference: str) -> dict:
    """Julga uma resposta. Devolve os campos do `Judgment` mais o consumo de tokens."""
    judge = JUDGE_PROMPT | get_llm().with_structured_output(Judgment, method="function_calling", include_raw=True)
    output = judge.invoke({"question": question, "answer": answer, "context": context, "reference": reference})
    judgment: Judgment | None = output["parsed"]
    if judgment is None:
        empty = {"reasoning": "Julgamento fora do formato.", "grounded": None, "correct": None, "complete": None}
        return {**empty, "usage": token_usage(output["raw"])}
    return {**judgment.model_dump(), "usage": token_usage(output["raw"])}
