"""Assistente de Conhecimento TechNova: front-end de chat em Streamlit (projeto final).

Junta as peças do curso:
- o índice FAISS com a base de conhecimento (módulo 01);
- a busca híbrida e a geração com citações (módulo 02);
- o grafo de RAG corretivo, com as conversas persistidas em SQLite (módulo 03);
- o consumo de tokens por pergunta (módulo 03).

Antes de rodar, crie o índice (uma única vez):
    python -m src.rag.ingestion

Execute a partir da raiz do projeto:
    streamlit run app.py
"""

import sqlite3
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import settings
from src.graph.corrective_rag import build_rag_graph
from src.models import add_usage, describe_llm
from src.rag.vectorstore import index_exists, index_info

st.set_page_config(page_title="Assistente de Conhecimento TechNova", page_icon="📚", layout="centered")

EXAMPLE_QUESTIONS = [
    "Em quantos dias posso trocar um produto?",
    "Como faço para acionar a garantia?",
    "Quanto tempo dura a bateria do Fone Nova Buds?",
    "O que é o programa Nova Pontos?",
]

STEP_LABELS = {
    "condense": lambda v: f"Pergunta entendida como: *{v['question']}*",
    "retrieve": lambda v: f"Busca híbrida: {len(v['chunks'])} trechos recuperados",
    "grade": lambda v: f"Avaliação de relevância: {len(v['relevant'])} trechos aprovados",
    "rewrite": lambda v: f"Nenhum trecho útil; nova consulta: *{v['queries'][-1]}*",
    "generate": lambda v: f"Resposta gerada com {len(v['sources'])} fonte(s) citada(s)",
    "no_answer": lambda v: "Nenhum trecho útil após as tentativas: resposta de segurança",
}


@st.cache_resource
def get_graph():
    """Grafo e checkpointer criados uma única vez. As conversas ficam em data/checkpoints.db."""
    connection = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    return build_rag_graph(checkpointer=SqliteSaver(connection))


def render_details(details: dict) -> None:
    for source in details.get("sources", []):
        page = f", página {source['page']}" if source.get("page") else ""
        with st.expander(f"[{source['number']}] {source.get('title') or source['source']}{page}"):
            st.caption(source["source"])
            st.markdown(source["text"][:1200])
    usage = details.get("usage", {})
    with st.expander(f"Como a resposta foi montada · {usage.get('total_tokens', 0)} tokens"):
        for step in details.get("steps", []):
            st.markdown(f"- {step}")


def answer(question: str) -> None:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    details: dict = {"steps": [], "sources": [], "usage": add_usage()}
    reply = ""
    with st.chat_message("assistant"):
        with st.status("Consultando a base de conhecimento...") as status:
            try:
                for update in get_graph().stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
                    for node, values in update.items():
                        values = values or {}
                        if node in STEP_LABELS:
                            step = STEP_LABELS[node](values)
                            details["steps"].append(step)
                            status.write(step)
                        details["usage"] = add_usage(details["usage"], values.get("usage"))
                        if values.get("messages"):
                            reply = values["messages"][-1].content
                            details["sources"] = values.get("sources", [])
                status.update(label="Pronto", state="complete")
            except Exception as error:  # noqa: BLE001 - o chat nunca deve quebrar para o usuário
                status.update(label="Falha", state="error")
                reply = "Desculpe, não consegui responder agora. Tente novamente em instantes."
                st.error(f"Detalhe técnico: {type(error).__name__}")
        st.markdown(reply)
        render_details(details)

    st.session_state.history.append({"role": "assistant", "content": reply, "details": details})


def main() -> None:
    st.session_state.setdefault("thread_id", uuid.uuid4().hex)
    st.session_state.setdefault("history", [])
    st.title("📚 Assistente de Conhecimento TechNova")
    st.caption("Respostas com base nas políticas, manuais, central de ajuda e perguntas frequentes da loja.")

    if not index_exists():
        st.error("O índice ainda não foi criado. Na raiz do projeto, rode: `python -m src.rag.ingestion`")
        st.stop()

    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("details"):
                render_details(item["details"])

    question = st.chat_input("Pergunte sobre políticas, produtos ou procedimentos")
    question = question or st.session_state.pop("example_question", None)
    if question:
        answer(question)

    # A barra lateral vem por último para refletir o consumo da resposta que acabou de ser gerada.
    with st.sidebar:
        st.header("Assistente de Conhecimento")
        info = index_info()
        st.caption(f"Modelo: {describe_llm()}")
        st.caption(f"Embeddings: {info.get('embeddings', '?')} · {info.get('chunks', '?')} trechos no índice")
        state = get_graph().get_state({"configurable": {"thread_id": st.session_state.thread_id}})
        usage = (state.values or {}).get("usage") or add_usage()
        st.metric("Tokens na conversa", usage.get("total_tokens", 0))
        if st.button("Nova conversa", width="stretch"):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.history = []
            st.rerun()
        st.divider()
        st.caption("Perguntas para testar")
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, width="stretch"):
                st.session_state.example_question = example
                st.rerun()


main()
