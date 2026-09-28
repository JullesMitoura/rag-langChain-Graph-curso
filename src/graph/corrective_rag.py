"""RAG corretivo como grafo LangGraph (módulo 03).

Um RAG simples busca uma vez e responde com o que veio, mesmo que a busca tenha
falhado. O RAG corretivo confere a própria busca:

    START -> condense -> retrieve -> grade --(há trechos relevantes)--> generate -> END
                            ^          |
                            |          +--(nenhum relevante, ainda há tentativas)--> rewrite
                            +---------------------------------------------------------+
                                       |
                                       +--(nenhum relevante, acabaram as tentativas)--> no_answer -> END

- condense: transforma "e para o Nova Mini?" em uma pergunta completa, usando o histórico;
- grade: o LLM marca quais trechos ajudam de fato (saída estruturada);
- rewrite: nova consulta com outras palavras, no máximo `MAX_REWRITES` vezes.

Uso:
    from src.graph.corrective_rag import build_rag_graph
    graph = build_rag_graph(checkpointer=InMemorySaver())
    graph.invoke({"messages": [("user", "Qual o prazo de troca?")]}, {"configurable": {"thread_id": "1"}})
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.config import get_logger, settings
from src.models import add_usage, get_llm, token_usage
from src.prompts import CONDENSE_PROMPT, GRADE_PROMPT, NO_ANSWER_MESSAGE, REWRITE_PROMPT
from src.rag.generation import format_context, generate_answer
from src.rag.retrieval import HybridRetriever
from src.rag.vectorstore import load_index

logger = get_logger(__name__)

HISTORY_MESSAGES = 6  # recorte da conversa usado para condensar e responder

# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------


def accumulate_usage(current: dict | None, new: dict | None) -> dict:
    """Reducer: soma o consumo de tokens de cada nó ao total da conversa."""
    return add_usage(current, new)


class RAGState(TypedDict, total=False):
    """Contrato entre os nós. Com um checkpointer, é salvo a cada passo.

    Chaves da conversa inteira: `messages` e `usage` (acumuladas por reducers).
    Chaves do turno: sobrescritas pelo nó `condense` no início de cada pergunta.
    """

    messages: Annotated[list[AnyMessage], add_messages]  # perguntas e respostas finais
    usage: Annotated[dict, accumulate_usage]  # tokens acumulados na conversa

    question: str  # pergunta completa e independente (depois de condensar)
    queries: list[str]  # consultas de busca tentadas; a última é a atual
    chunks: list[Document]  # recuperados na última busca
    relevant: list[Document]  # aprovados pela avaliação de relevância
    rewrites: int  # quantas vezes a consulta foi reescrita
    sources: list[dict]  # fontes citadas na resposta
    found: bool


class RelevanceGrade(BaseModel):
    """Trechos que ajudam a responder a pergunta."""

    reasoning: str = Field(description="Frase curta explicando a decisão.")
    relevant: list[int] = Field(default_factory=list, description="Números dos trechos úteis. Vazio se nenhum.")


# ---------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------


def build_rag_graph(retriever: HybridRetriever | None = None, checkpointer=None) -> CompiledStateGraph:
    """Monta e compila o grafo de RAG corretivo."""
    retriever = retriever or HybridRetriever(index=load_index())

    def condense(state: RAGState) -> dict:
        messages = state["messages"]
        question, usage = messages[-1].content, {}
        if len(messages) > 1:
            response = get_llm().invoke(CONDENSE_PROMPT.invoke({"history": messages[-HISTORY_MESSAGES:]}))
            usage = token_usage(response)
            question = StrOutputParser().invoke(response).strip() or question
        logger.info("Pergunta do turno | pergunta=%s", question)
        # Zera o trabalho do turno anterior: o estado é persistido entre perguntas.
        return {
            "question": question,
            "queries": [question],
            "chunks": [],
            "relevant": [],
            "rewrites": 0,
            "sources": [],
            "found": False,
            "usage": usage,
        }

    def retrieve(state: RAGState) -> dict:
        return {"chunks": retriever.search(state["queries"][-1], k=settings.retrieval_k + 2)}

    def grade(state: RAGState) -> dict:
        chunks = state["chunks"]
        if not chunks:
            return {"relevant": []}
        grader = GRADE_PROMPT | get_llm().with_structured_output(
            RelevanceGrade, method="function_calling", include_raw=True
        )
        output = grader.invoke({"question": state["question"], "context": format_context(chunks)})
        numbers = output["parsed"].relevant if output["parsed"] else []
        relevant = [chunks[n - 1] for n in dict.fromkeys(numbers) if 1 <= n <= len(chunks)]
        logger.info("Avaliação | recuperados=%s | relevantes=%s", len(chunks), len(relevant))
        return {"relevant": relevant[: settings.retrieval_k], "usage": token_usage(output["raw"])}

    def route(state: RAGState) -> Literal["generate", "rewrite", "no_answer"]:
        if state.get("relevant"):
            return "generate"
        if state.get("rewrites", 0) < settings.max_rewrites:
            return "rewrite"
        return "no_answer"

    def rewrite(state: RAGState) -> dict:
        response = get_llm().invoke(
            REWRITE_PROMPT.invoke({"question": state["question"], "tried": "; ".join(state["queries"])})
        )
        query = StrOutputParser().invoke(response).strip() or state["question"]
        logger.info("Consulta reescrita | tentativa=%s | consulta=%s", state.get("rewrites", 0) + 1, query)
        return {
            "queries": [*state["queries"], query],
            "rewrites": state.get("rewrites", 0) + 1,
            "usage": token_usage(response),
        }

    def generate(state: RAGState) -> dict:
        history = [m for m in state["messages"][:-1] if isinstance(m, (HumanMessage, AIMessage))]
        result = generate_answer(state["question"], state["relevant"], history[-HISTORY_MESSAGES:])
        return {
            "messages": [AIMessage(result["answer"])],
            "sources": result["sources"],
            "found": result["found"],
            "usage": result["usage"],
        }

    def no_answer(state: RAGState) -> dict:
        return {"messages": [AIMessage(NO_ANSWER_MESSAGE)], "sources": [], "found": False}

    builder = StateGraph(RAGState)
    builder.add_node("condense", condense)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)
    builder.add_node("no_answer", no_answer)

    builder.add_edge(START, "condense")
    builder.add_edge("condense", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route, ["generate", "rewrite", "no_answer"])
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("no_answer", END)
    return builder.compile(checkpointer=checkpointer, name="corrective_rag")
