"""Configuração do projeto: variáveis de ambiente e logs.

Todo módulo que precisar de uma configuração importa `settings` daqui, em vez de
chamar `os.getenv`: um único lugar com nomes, tipos e valores padrão. O nome de
cada campo corresponde à variável de ambiente em maiúsculas (`chunk_size` ->
`CHUNK_SIZE`).

Os caminhos são relativos à raiz do projeto: execute o front-end, a ingestão e
os notebooks a partir dela (a primeira célula de cada notebook muda para a raiz).

Uso:
    from src.config import get_logger, settings
    settings.chunk_size
    logger = get_logger(__name__)
"""

import logging
import sys
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=True)  # lê o .env e exporta para o ambiente, sobrescrevendo variáveis existentes


class Settings(BaseSettings):
    """Variáveis de ambiente aceitas pela aplicação."""

    # Variável vazia no .env (ex.: LLM_TEMPERATURE=) vira None, em vez de erro de validação.
    model_config = SettingsConfigDict(env_parse_none_str="")

    # --- LLM de chat -------------------------------------------------------
    # Padrão: Azure OpenAI, o mesmo provedor dos embeddings.
    llm_provider: Literal["azure", "groq"] = "azure"
    llm_max_retries: int = 6
    llm_timeout: float = 60.0

    # --- Azure OpenAI (chat e embeddings) --------------------------------
    azure_openai_endpoint: str | None = None
    azure_openai_key: SecretStr | None = None
    azure_openai_llm: str = "gpt-5.4-mini"  # deployment de chat
    azure_openai_api_version: str = "2024-12-01-preview"
    # 0 deixa as respostas do RAG mais estáveis. Se o seu deployment recusar (alguns modelos de
    # raciocínio só aceitam a temperatura padrão), deixe LLM_TEMPERATURE vazio no .env.
    llm_temperature: float | None = 0.0

    # --- Groq (alternativa para o chat) -----------------------------------
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"

    # --- Embeddings -------------------------------------------------------
    embeddings_provider: Literal["azure", "openai"] = "azure"
    azure_openai_embeddings: str = "text-embedding-3-large"  # deployment de embeddings
    openai_api_key: SecretStr | None = None
    openai_embeddings_model: str = "text-embedding-3-large"
    # Vazio = dimensão nativa do modelo (3072 no text-embedding-3-large).
    embeddings_dimensions: int | None = None

    # --- Dados ------------------------------------------------------------
    documents_dir: str = "data/documents"
    index_dir: str = "data/faiss_index"
    store_db_path: str = "data/store.db"
    checkpoint_db_path: str = "data/checkpoints.db"
    eval_questions_path: str = "data/eval/questions.json"

    # --- Ingestão ---------------------------------------------------------
    chunk_size: int = 1000  # caracteres por trecho
    chunk_overlap: int = 150

    # --- Recuperação e geração -------------------------------------------
    retrieval_k: int = 4  # trechos entregues ao modelo
    fetch_k: int = 12  # candidatos de cada busca antes da fusão e da reordenação
    max_rewrites: int = 2  # tentativas de reescrever a consulta no RAG corretivo

    # --- Logs ------------------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Instância única (cacheada) das configurações."""
    return Settings()


settings = get_settings()


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
NOISY_LIBRARIES = ("httpx", "httpcore", "groq", "openai", "urllib3", "watchdog", "langchain", "langgraph", "faiss")


@lru_cache
def _configure_logging() -> None:
    """Configura o logger `src` uma única vez: formato, nível e saída."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger("src")
    root.setLevel(settings.log_level.upper())
    root.addHandler(handler)
    root.propagate = False
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Logger filho de `src`, já configurado. Use `get_logger(__name__)`."""
    _configure_logging()
    return logging.getLogger(name if name.startswith("src") else f"src.{name}")
