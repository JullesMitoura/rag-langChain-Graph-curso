"""Recuperação: encontrar os trechos que ajudam a responder (módulo 02).

Três formas de buscar, que se complementam:
- **Vetorial (FAISS)**: encontra trechos com SIGNIFICADO parecido, mesmo com
  palavras diferentes ("devolver" x "reembolso").
- **Lexical (BM25)**: encontra trechos com as MESMAS palavras, ótima para
  códigos e nomes exatos ("NB-001", "Nova Pontos", "IPX7").
- **Híbrida**: junta as duas listas com a fusão RRF (Reciprocal Rank Fusion),
  que soma 1 / (60 + posição) de cada trecho em cada lista.

Filtros por metadados garantem que só entrem documentos em vigor e do público
certo: a política de frete de 2024 está arquivada e o procedimento de
reembolso dos atendentes é interno.

`HybridRetriever` é um `BaseRetriever` do LangChain: funciona com `.invoke()`
e encaixa em chains LCEL como qualquer retriever.

Uso:
    from src.rag.retrieval import HybridRetriever
    retriever = HybridRetriever(index=load_index())
    chunks = retriever.invoke("Qual o prazo para trocar um fone?")
    chunks = retriever.search("fone IPX7", k=6, mode="bm25")   # sobrescreve a configuração
"""

import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, PrivateAttr
from rank_bm25 import BM25Okapi

from src.config import get_logger, settings
from src.models import get_llm
from src.prompts import RERANK_PROMPT
from src.rag.vectorstore import FAISS, stored_chunks

logger = get_logger(__name__)

MetadataFilter = Callable[[dict], bool]
SearchMode = Literal["hybrid", "vector", "bm25"]
RRF_CONSTANT = 60

# ---------------------------------------------------------------------------
# Filtros por metadados
# ---------------------------------------------------------------------------


def make_filter(audiences: Iterable[str] = ("customers",), include_archived: bool = False) -> MetadataFilter:
    """Filtro de metadados: público permitido e (por padrão) só documentos em vigor."""
    allowed = set(audiences)

    def metadata_filter(metadata: dict) -> bool:
        if not include_archived and metadata.get("status", "active") != "active":
            return False
        return metadata.get("audience", "customers") in allowed

    return metadata_filter


CUSTOMERS_ONLY = make_filter()

# ---------------------------------------------------------------------------
# Buscas
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acentos, só letras e números: 'Garantia' e 'garantía' viram 'garantia'."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+(?:-[0-9]+)?", plain)


class BM25Search:
    """Busca lexical BM25 sobre os mesmos trechos do índice vetorial."""

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def search(self, query: str, k: int, metadata_filter: MetadataFilter | None = None) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        results = []
        for i in sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True):
            if scores[i] <= 0 or len(results) == k:
                break
            if metadata_filter is None or metadata_filter(self.chunks[i].metadata):
                results.append(self.chunks[i])
        return results


def vector_search(index: FAISS, query: str, k: int, metadata_filter: MetadataFilter | None = None) -> list[Document]:
    """Busca por similaridade no FAISS. `fetch_k` alto para sobrar candidatos depois do filtro."""
    return index.similarity_search(query, k=k, filter=metadata_filter, fetch_k=max(50, k * 5))


def rrf_fusion(*rankings: list[Document], k: int | None = None) -> list[Document]:
    """Reciprocal Rank Fusion: combina rankings sem precisar comparar as notas de cada busca."""
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for ranking in rankings:
        for position, chunk in enumerate(ranking, start=1):
            key = chunk.metadata.get("id") or chunk.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_CONSTANT + position)
            by_key[key] = chunk
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [by_key[key] for key in ordered[: k or len(ordered)]]


# ---------------------------------------------------------------------------
# Reordenação com o LLM
# ---------------------------------------------------------------------------


class ChunkScore(BaseModel):
    number: int = Field(description="Número do trecho, como aparece na lista.")
    score: int = Field(ge=0, le=3, description="0 = não ajuda; 3 = responde diretamente.")


class ChunkScores(BaseModel):
    scores: list[ChunkScore] = Field(description="Uma nota para cada trecho da lista.")


def rerank_with_llm(query: str, chunks: list[Document], k: int) -> list[Document]:
    """Reordenação (rerank) com o LLM: uma chamada dá nota a todos os candidatos."""
    if not chunks:
        return []
    context = "\n\n".join(f"[{i}] {c.page_content[:700]}" for i, c in enumerate(chunks, start=1))
    grader = RERANK_PROMPT | get_llm().with_structured_output(ChunkScores, method="function_calling")
    scores = {s.number: s.score for s in grader.invoke({"question": query, "context": context}).scores}
    ordered = sorted(range(1, len(chunks) + 1), key=lambda i: scores.get(i, 0), reverse=True)
    return [chunks[i - 1] for i in ordered if scores.get(i, 0) > 0][:k]


# ---------------------------------------------------------------------------
# Retriever do produto
# ---------------------------------------------------------------------------


class HybridRetriever(BaseRetriever):
    """Busca vetorial, BM25 ou híbrida (RRF), com filtro de metadados e reordenação opcional."""

    index: FAISS
    k: int = Field(default_factory=lambda: settings.retrieval_k)
    fetch_k: int = Field(default_factory=lambda: settings.fetch_k)
    mode: SearchMode = "hybrid"
    rerank: bool = False
    metadata_filter: MetadataFilter | None = CUSTOMERS_ONLY

    _bm25: BM25Search = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._bm25 = BM25Search(list(stored_chunks(self.index).values()))

    def search(
        self,
        query: str,
        k: int | None = None,
        mode: SearchMode | None = None,
        rerank: bool | None = None,
    ) -> list[Document]:
        """Os `k` trechos mais úteis. Os argumentos sobrescrevem a configuração só nesta chamada."""
        k = k or self.k
        mode = mode or self.mode
        if mode == "vector":
            results = vector_search(self.index, query, self.fetch_k, self.metadata_filter)
        elif mode == "bm25":
            results = self._bm25.search(query, self.fetch_k, self.metadata_filter)
        else:
            results = rrf_fusion(
                vector_search(self.index, query, self.fetch_k, self.metadata_filter),
                self._bm25.search(query, self.fetch_k, self.metadata_filter),
            )
        if self.rerank if rerank is None else rerank:
            return rerank_with_llm(query, results, k)
        return results[:k]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.search(query)
