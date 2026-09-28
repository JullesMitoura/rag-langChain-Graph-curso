"""Índice vetorial FAISS: criar, salvar e carregar (módulo 01).

FAISS (Facebook AI Similarity Search) é uma biblioteca de busca vetorial que
roda na própria máquina: o índice fica em uma pasta (`data/faiss_index`) com
os vetores (`index.faiss`), os documentos com metadados (`index.pkl`) e um
`index_info.json` que registra o modelo de embeddings usado.

Sobre a dependência: a integração do FAISS com o LangChain vive no pacote
`langchain-community`, que está sendo descontinuado em favor de pacotes
dedicados. Este é o ÚNICO arquivo do projeto que importa dele: se o FAISS
ganhar um pacote próprio, a migração acontece aqui e o resto do projeto não muda.

Uso:
    from src.rag.vectorstore import build_index, load_index, save_index
    index = build_index(chunks)
    save_index(index)
    index = load_index()
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import get_logger, settings
from src.models import describe_embeddings, get_embeddings

with warnings.catch_warnings():
    # O aviso de descontinuação do langchain-community é conhecido e está documentado acima.
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS

logger = get_logger(__name__)

INFO_FILE = "index_info.json"


def build_index(chunks: list[Document], embeddings: Embeddings | None = None) -> FAISS:
    """Calcula os embeddings dos trechos e monta um índice FAISS em memória.

    Trechos com `metadata["id"]` (gerado na ingestão) usam esse id no índice.
    """
    if not chunks:
        raise ValueError("Não há trechos para indexar.")
    ids = [c.metadata["id"] for c in chunks] if all("id" in c.metadata for c in chunks) else None
    logger.info("Criando índice FAISS | trechos=%s | embeddings=%s", len(chunks), describe_embeddings())
    return FAISS.from_documents(chunks, embeddings or get_embeddings(), ids=ids)


def save_index(index: FAISS, folder: str | Path | None = None) -> Path:
    """Grava o índice em disco, junto com o registro do modelo de embeddings usado."""
    target = Path(folder or settings.index_dir)
    index.save_local(str(target))
    info = {
        "embeddings": describe_embeddings(),
        "chunks": index.index.ntotal,
        "dimensions": index.index.d,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (target / INFO_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Índice salvo | pasta=%s | trechos=%s", target, info["chunks"])
    return target


def index_exists(folder: str | Path | None = None) -> bool:
    return (Path(folder or settings.index_dir) / "index.faiss").exists()


def index_info(folder: str | Path | None = None) -> dict:
    """Registro do índice salvo (modelo de embeddings, número de trechos, data)."""
    path = Path(folder or settings.index_dir) / INFO_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_index(folder: str | Path | None = None, embeddings: Embeddings | None = None) -> FAISS:
    """Carrega um índice salvo.

    Os documentos do índice são gravados com pickle, que pode executar código ao
    ser lido; por isso o LangChain exige `allow_dangerous_deserialization=True`.
    Carregue SOMENTE índices criados por você, nunca arquivos de terceiros.
    """
    source = Path(folder or settings.index_dir)
    if not index_exists(source):
        raise FileNotFoundError(f"Índice não encontrado em '{source}'. Rode a ingestão: python -m src.rag.ingestion")
    info = index_info(source)
    if info and info.get("embeddings") != describe_embeddings():
        logger.warning(
            "O índice foi criado com '%s', mas os embeddings atuais são '%s'. Reindexe os documentos.",
            info.get("embeddings"),
            describe_embeddings(),
        )
    return FAISS.load_local(str(source), embeddings or get_embeddings(), allow_dangerous_deserialization=True)


def stored_chunks(index: FAISS) -> dict[str, Document]:
    """Todos os trechos do índice, por id (o FAISS do LangChain não expõe uma listagem pública)."""
    return dict(index.docstore._dict)  # noqa: SLF001
