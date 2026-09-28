"""Agente de suporte com RAG agêntico (módulo 03).

No RAG corretivo, o fluxo é fixo: toda pergunta passa pela busca. Aqui a busca é
uma FERRAMENTA: o agente decide se busca na base, se consulta um pedido, se faz
as duas coisas ou se responde direto (uma saudação, por exemplo).

Ferramentas:
- `search_knowledge_base`: busca híbrida na base de conhecimento (criada por
  `make_search_tool`, porque depende do índice carregado);
- `get_order`: dados de um pedido no banco da loja (somente leitura).

É o mesmo `create_agent` do curso de LangChain, agora com um checkpointer (curso de
LangGraph) para manter a conversa.

Uso:
    from src.graph.agent import build_support_agent
    agent = build_support_agent(checkpointer=InMemorySaver())
    agent.invoke({"messages": [("user", "Meu PED-1001 está na garantia?")]}, {"configurable": {"thread_id": "1"}})
"""

import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolErrorMiddleware
from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import BaseModel, Field

from src.config import get_logger
from src.database import query
from src.models import get_llm
from src.prompts import AGENT_SYSTEM
from src.rag.retrieval import HybridRetriever
from src.rag.vectorstore import load_index

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Ferramentas
# ---------------------------------------------------------------------------


class GetOrderArgs(BaseModel):
    order_id: str = Field(
        pattern=r"^PED-\d{4,5}$",
        description="Identificador do pedido: PED- seguido de 4 ou 5 dígitos (ex.: PED-1001).",
    )


@tool(args_schema=GetOrderArgs)
def get_order(order_id: str) -> str:
    """Consulta status, datas, transportadora e itens de um pedido pelo identificador."""
    orders = query(
        "SELECT id, status, purchase_date, estimated_delivery, carrier, tracking_code, total FROM orders WHERE id = ?",
        (order_id,),
    )
    if not orders:
        raise ToolException(f"Pedido {order_id} não encontrado.")
    order = orders[0]
    order["items"] = query(
        "SELECT p.sku, p.name, i.quantity FROM order_items i JOIN products p ON p.sku = i.sku WHERE i.order_id = ?",
        (order_id,),
    )
    return json.dumps(order, ensure_ascii=False, default=str)


class SearchArgs(BaseModel):
    query: str = Field(description="O que procurar na base (ex.: 'prazo de troca', 'bateria do Nova Buds').")


def make_search_tool(retriever: HybridRetriever) -> BaseTool:
    """Cria a ferramenta de busca ligada a um índice já carregado."""

    @tool(args_schema=SearchArgs)
    def search_knowledge_base(query: str) -> str:
        """Busca nas políticas, manuais de produtos, central de ajuda e perguntas frequentes da TechNova."""
        chunks = retriever.invoke(query)
        if not chunks:
            return "Nenhum trecho encontrado na base para essa consulta."
        blocks = []
        for chunk in chunks:
            md = chunk.metadata
            origin = md.get("source", "?") + (f", p. {md['page']}" if md.get("page") else "")
            blocks.append(f"[{origin}]\n{chunk.page_content}")
        return "\n\n".join(blocks)

    return search_knowledge_base


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------


def _tool_error_message(error: Exception, request: Any) -> str:
    """Erros de negócio voltam ao modelo como mensagem; falhas internas viram mensagem genérica."""
    if isinstance(error, ToolException):
        return f"Erro: {error}"
    logger.error("Falha inesperada na ferramenta | nome=%s | erro=%r", request.tool_call["name"], error)
    return "Erro: falha interna ao executar a ferramenta."


def build_support_agent(retriever: HybridRetriever | None = None, checkpointer=None, max_model_calls: int = 6):
    """Agente com a base de conhecimento e a consulta de pedidos."""
    retriever = retriever or HybridRetriever(index=load_index())
    return create_agent(
        get_llm(),
        tools=[make_search_tool(retriever), get_order],
        system_prompt=AGENT_SYSTEM,
        middleware=[
            ToolErrorMiddleware(_tool_error_message),
            ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
        ],
        checkpointer=checkpointer,
        name="support_agent",
    )
