"""Acesso centralizado aos modelos: chat (LLM) e embeddings.

Nenhum outro módulo instancia um cliente de modelo diretamente: todos chamam
`get_llm()` ou `get_embeddings()`. Assim as configurações (modelo, temperatura,
retentativas) são as mesmas em toda a aplicação e trocar de provedor é mudar
uma variável de ambiente.

Chat (`LLM_PROVIDER`):
    azure -> (padrão) AzureChatOpenAI, deployment em `AZURE_OPENAI_LLM`
    groq  -> ChatGroq, modelo em `GROQ_MODEL` (plano gratuito com limite diário de tokens)

Embeddings (`EMBEDDINGS_PROVIDER`):
    azure  -> (padrão) AzureOpenAIEmbeddings, deployment em `AZURE_OPENAI_EMBEDDINGS`
    openai -> OpenAIEmbeddings, modelo em `OPENAI_EMBEDDINGS_MODEL`

IMPORTANTE: um índice só pode ser consultado com o MESMO modelo de embeddings
que o criou. Trocar de modelo exige reindexar os documentos.

Uso:
    from src.models import get_embeddings, get_llm, token_usage
    resposta = get_llm().invoke("Olá!")
    token_usage(resposta)  # {"input_tokens": 78, "output_tokens": 29, "total_tokens": 107, "calls": 1}
    vetor = get_embeddings().embed_query("Qual o prazo de troca?")
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from src.config import get_logger, settings

logger = get_logger(__name__)


def _require(**values: object) -> None:
    """Falha cedo, com mensagem clara, se faltar alguma configuração do provedor."""
    missing = [name.upper() for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Configuração ausente no .env: {', '.join(missing)}.")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def get_llm(model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    """Cria (uma única vez por combinação de parâmetros) o cliente de chat.

    Args:
        model: deployment (Azure) ou modelo (Groq). Padrão: o definido no .env.
        temperature: padrão `LLM_TEMPERATURE`; None usa a temperatura padrão do modelo.
    """
    temperature = settings.llm_temperature if temperature is None else temperature
    if settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        deployment = model or settings.azure_openai_llm
        _require(azure_openai_endpoint=settings.azure_openai_endpoint, azure_openai_key=settings.azure_openai_key)
        logger.info("Inicializando LLM | provedor=azure | deployment=%s", deployment)
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            azure_deployment=deployment,
            api_version=settings.azure_openai_api_version,
            temperature=temperature,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
        )
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        name = model or settings.groq_model
        _require(groq_api_key=settings.groq_api_key)
        logger.info("Inicializando LLM | provedor=groq | modelo=%s", name)
        # Nos modelos de raciocínio (gpt-oss), `reasoning_format="hidden"` devolve só a resposta.
        extras = {"reasoning_format": "hidden", "reasoning_effort": "low"} if "gpt-oss" in name else {}
        return ChatGroq(
            model=name,
            temperature=temperature or 0.0,
            api_key=settings.groq_api_key,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
            **extras,
        )
    raise ValueError(f"LLM_PROVIDER desconhecido: '{settings.llm_provider}'. Use 'azure' ou 'groq'.")


def describe_llm() -> str:
    """Provedor e modelo de chat em uso, para logs e para o front-end."""
    if settings.llm_provider == "azure":
        return f"azure · {settings.azure_openai_llm}"
    return f"groq · {settings.groq_model}"


def token_usage(message: BaseMessage) -> dict[str, int]:
    """Consumo de tokens de uma resposta do modelo, pronto para ser somado com `add_usage`."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "calls": 1 if isinstance(message, AIMessage) else 0,
    }


def add_usage(*usages: dict[str, int] | None) -> dict[str, int]:
    """Soma vários consumos produzidos por `token_usage` (None é ignorado)."""
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}
    for usage in usages:
        for key, value in (usage or {}).items():
            total[key] = total.get(key, 0) + value
    return total


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Cria (uma única vez) o cliente de embeddings definido em `EMBEDDINGS_PROVIDER`."""
    if settings.embeddings_provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        _require(azure_openai_endpoint=settings.azure_openai_endpoint, azure_openai_key=settings.azure_openai_key)
        logger.info("Inicializando embeddings | provedor=azure | deployment=%s", settings.azure_openai_embeddings)
        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            azure_deployment=settings.azure_openai_embeddings,
            api_version=settings.azure_openai_api_version,
            dimensions=settings.embeddings_dimensions,
            max_retries=settings.llm_max_retries,
        )
    if settings.embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        _require(openai_api_key=settings.openai_api_key)
        logger.info("Inicializando embeddings | provedor=openai | modelo=%s", settings.openai_embeddings_model)
        return OpenAIEmbeddings(
            model=settings.openai_embeddings_model,
            api_key=settings.openai_api_key,
            dimensions=settings.embeddings_dimensions,
            max_retries=settings.llm_max_retries,
        )
    raise ValueError(f"EMBEDDINGS_PROVIDER desconhecido: '{settings.embeddings_provider}'. Use 'azure' ou 'openai'.")


def describe_embeddings() -> str:
    """Provedor e modelo de embeddings em uso, para logs, índice e front-end."""
    if settings.embeddings_provider == "azure":
        return f"azure · {settings.azure_openai_embeddings}"
    return f"openai · {settings.openai_embeddings_model}"
