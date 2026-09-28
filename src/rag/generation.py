"""Geração de respostas fundamentadas, com citações (módulo 02).

O modelo recebe os trechos recuperados numerados ([1], [2]...) e devolve uma
resposta estruturada (`RAGAnswer`): o texto, os números dos trechos citados e
se encontrou ou não a resposta. A aplicação traduz os números em fontes
(arquivo, título, página), que o front-end mostra ao cliente.

Uso:
    from src.rag.generation import generate_answer
    result = generate_answer("Qual o prazo de troca?", chunks)
    result["answer"], result["sources"]
"""

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from src.config import get_logger
from src.models import add_usage, get_llm, token_usage
from src.prompts import NO_ANSWER_MESSAGE, RAG_PROMPT

logger = get_logger(__name__)


class RAGAnswer(BaseModel):
    """Resposta do assistente com as citações usadas."""

    found_answer: bool = Field(
        description=(
            "True se o contexto permite responder, INCLUSIVE com uma resposta negativa apoiada nele "
            "(ex.: 'esse cupom não existe'). False só quando o contexto não trata do assunto."
        )
    )
    answer: str = Field(description="Resposta ao cliente, com citações entre colchetes, ex.: [1].")
    citations: list[int] = Field(default_factory=list, description="Números dos trechos citados na resposta.")


def format_context(chunks: list[Document]) -> str:
    """Trechos numerados, com a origem de cada um, prontos para o prompt."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        md = chunk.metadata
        origin = md.get("source", "?") + (f", página {md['page']}" if md.get("page") else "")
        blocks.append(f"[{number}] (fonte: {origin})\n{chunk.page_content}")
    return "\n\n".join(blocks)


def describe_source(chunk: Document) -> dict[str, Any]:
    """Informações de uma fonte para exibir ao cliente."""
    md = chunk.metadata
    return {
        "source": md.get("source"),
        "title": md.get("title"),
        "page": md.get("page"),
        "section": md.get("section"),
        "text": chunk.page_content,
    }


def generate_answer(
    question: str,
    chunks: list[Document],
    history: list[BaseMessage] | None = None,
) -> dict[str, Any]:
    """Gera a resposta a partir dos trechos. Sem trechos, não chama o modelo.

    Devolve `answer`, `found`, `sources` (fontes citadas) e `usage` (tokens).
    """
    if not chunks:
        return {"answer": NO_ANSWER_MESSAGE, "found": False, "sources": [], "usage": {}}

    model = get_llm().with_structured_output(RAGAnswer, method="function_calling", include_raw=True)
    inputs = {"question": question, "context": format_context(chunks), "history": history or []}
    usage: dict[str, int] = {}
    output: dict = {}
    for _ in range(2):  # uma nova tentativa se o modelo sair do formato
        output = (RAG_PROMPT | model).invoke(inputs)
        usage = add_usage(usage, token_usage(output["raw"]))
        if output["parsed"] is not None:
            break

    result: RAGAnswer | None = output["parsed"]
    if result is None:
        # Plano B: o modelo respondeu em texto livre. Aproveitamos o texto e lemos as citações [n] nele.
        text = str(getattr(output["raw"], "content", "") or "").strip()
        logger.warning("Resposta fora do formato; usando o texto livre | erro=%s", output["parsing_error"])
        if not text:
            return {"answer": NO_ANSWER_MESSAGE, "found": False, "sources": [], "usage": usage}
        result = RAGAnswer(found_answer=True, answer=text, citations=[int(n) for n in re.findall(r"\[(\d+)\]", text)])

    if not result.found_answer:
        # Sem resposta, nenhuma fonte é exibida: citar documentos ao lado de "não encontrei" confunde o cliente.
        return {"answer": NO_ANSWER_MESSAGE, "found": False, "sources": [], "usage": usage}
    valid = [n for n in dict.fromkeys(result.citations) if 1 <= n <= len(chunks)]
    return {
        "answer": result.answer,
        "found": True,
        "sources": [{"number": n, **describe_source(chunks[n - 1])} for n in valid],
        "usage": usage,
    }
