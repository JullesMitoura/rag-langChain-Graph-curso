"""Ingestão: carregar, dividir e indexar os documentos (módulo 01).

    arquivos -> load_folder() -> split_documents() -> embeddings -> FAISS -> disco

1. Carregar. Cada formato tem um carregador escrito sobre bibliotecas comuns
   (pypdf, BeautifulSoup, csv), para que fique claro o que acontece em cada etapa.
   O resultado é uma lista de `Document`: `page_content` (o texto) e `metadata`
   (fonte, página, título, status, público...). Os metadados não viram vetor,
   mas permitem filtrar a busca e citar as fontes.
       .md    políticas e procedimentos, com front matter (metadados no topo)
       .pdf   manuais de produtos, um Document por página
       .html  artigos da central de ajuda, só o conteúdo de <article>
       .csv   perguntas frequentes, um Document por linha

2. Dividir (chunking). Markdown é dividido primeiro pelos títulos (## e ###),
   que viram metadados (`section`, `subsection`). Depois tudo passa pelo
   `RecursiveCharacterTextSplitter` (parágrafos, frases, palavras) até caber em
   `CHUNK_SIZE` caracteres, com `CHUNK_OVERLAP` de sobreposição. Cada trecho
   recebe um cabeçalho de contexto (título > seção > página) e um id estável.

3. Indexar. Os ids estáveis permitem atualizar o índice sem recriá-lo: adicionar
   só trechos novos, remover uma fonte, reindexar um arquivo que mudou.

Uso (a partir da raiz do projeto):
    python -m src.rag.ingestion                               # recria o índice de data/documents
    python -m src.rag.ingestion data/documents --incremental   # depois de mover os arquivos para a subpasta certa
"""

import argparse
import csv
import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.messages.utils import count_tokens_approximately
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import get_logger, settings
from src.rag.vectorstore import FAISS, build_index, index_exists, load_index, save_index, stored_chunks

logger = get_logger(__name__)

# Valores assumidos quando o documento não informa: conteúdo em vigor e para clientes.
DEFAULT_METADATA = {"status": "active", "audience": "customers"}

# ---------------------------------------------------------------------------
# 1. Carregar
# ---------------------------------------------------------------------------


def _source(path: Path, base: Path) -> str:
    """Caminho relativo à pasta base, com barras normais (ex.: policies/exchange.md)."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def _clean(text: str) -> str:
    """Remove espaços repetidos e linhas em branco em excesso."""
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Separa o front matter (bloco `---` no topo com `chave: valor`) do corpo do Markdown."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    metadata = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, parts[2].lstrip()


def load_markdown(path: Path, base: Path) -> list[Document]:
    metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=body.strip(),
            metadata={
                **DEFAULT_METADATA,
                **metadata,
                "source": _source(path, base),
                "type": "markdown",
                "title": metadata.get("title", path.stem),
            },
        )
    ]


def load_pdf(path: Path, base: Path) -> list[Document]:
    """Um Document por página: a página vira metadado e pode ser citada na resposta."""
    reader = PdfReader(str(path))
    info = reader.metadata or {}
    documents = []
    for number, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if not text:  # página sem texto extraível (ex.: digitalizada)
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    **DEFAULT_METADATA,
                    "source": _source(path, base),
                    "type": "pdf",
                    "category": "manual",
                    "title": str(info.get("/Title") or path.stem),
                    "product": str(info.get("/Subject") or ""),
                    "page": number,
                    "total_pages": len(reader.pages),
                },
            )
        )
    return documents


def load_html(path: Path, base: Path) -> list[Document]:
    """Extrai só o conteúdo do artigo: menus, rodapés e scripts ficam de fora."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    description = soup.find("meta", attrs={"name": "description"})
    updated_at = soup.find("meta", attrs={"name": "updated_at"})
    main = soup.find("article") or soup.body or soup
    for element in main.find_all(["script", "style", "nav", "footer"]):
        element.decompose()
    return [
        Document(
            page_content=_clean(main.get_text("\n")),
            metadata={
                **DEFAULT_METADATA,
                "source": _source(path, base),
                "type": "html",
                "category": "help_center",
                "title": soup.title.get_text(strip=True) if soup.title else path.stem,
                "description": description.get("content", "") if description else "",
                "updated_at": updated_at.get("content", "") if updated_at else "",
            },
        )
    ]


def load_csv(path: Path, base: Path) -> list[Document]:
    """Uma pergunta frequente por Document: cada linha já é uma unidade de sentido."""
    with path.open(encoding="utf-8", newline="") as file:
        return [
            Document(
                page_content=f"Pergunta: {row['question']}\nResposta: {row['answer']}",
                metadata={
                    **DEFAULT_METADATA,
                    "source": _source(path, base),
                    "type": "faq",
                    "category": row.get("category", "faq"),
                    "title": row["question"],
                    "row": number,
                },
            )
            for number, row in enumerate(csv.DictReader(file), start=2)  # a linha 1 é o cabeçalho
        ]


LOADERS = {".md": load_markdown, ".pdf": load_pdf, ".html": load_html, ".csv": load_csv}


def load_file(path: str | Path, base: str | Path | None = None) -> list[Document]:
    """Carrega um arquivo com o carregador do seu formato."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Formato não suportado: {path.suffix}. Use {', '.join(sorted(LOADERS))}.")
    return loader(path, Path(base) if base else path.parent)


def load_folder(folder: str | Path | None = None, base: str | Path | None = None) -> list[Document]:
    """Carrega todos os arquivos suportados de uma pasta (e subpastas), em ordem alfabética.

    O `source` de cada documento é relativo a `base` (padrão: a própria pasta). Para carregar
    uma subpasta com os mesmos nomes do índice principal, use `base=settings.documents_dir`.
    """
    root = Path(folder or settings.documents_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in LOADERS)
    documents: list[Document] = []
    for file in files:
        try:
            documents.extend(load_file(file, base or root))
        except Exception as error:  # noqa: BLE001 - um arquivo com problema não deve interromper a ingestão
            logger.warning("Falha ao carregar | arquivo=%s | erro=%s: %s", file.name, type(error).__name__, error)
    logger.info("Documentos carregados | arquivos=%s | documentos=%s", len(files), len(documents))
    return documents


# ---------------------------------------------------------------------------
# 2. Dividir
# ---------------------------------------------------------------------------

MARKDOWN_HEADERS = [("##", "section"), ("###", "subsection")]


def make_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _split_markdown_by_headers(document: Document) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS, strip_headers=False)
    return [
        Document(page_content=part.page_content, metadata={**document.metadata, **part.metadata})
        for part in splitter.split_text(document.page_content)
    ]


def chunk_id(chunk: Document) -> str:
    """Id estável: o mesmo trecho gera sempre o mesmo id; qualquer mudança gera outro."""
    md = chunk.metadata
    key = f"{md.get('source')}|{md.get('page', '')}|{md.get('row', '')}|{chunk.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def context_header(metadata: dict) -> str:
    """Contexto que acompanha cada trecho: título do documento, seção e página."""
    parts = [metadata.get("title"), metadata.get("section"), metadata.get("subsection")]
    if metadata.get("page"):
        parts.append(f"página {metadata['page']}")
    return " > ".join(p for p in parts if p)


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    add_header: bool = True,
) -> list[Document]:
    """Divide os documentos em trechos prontos para indexar (com `metadata["id"]`)."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    for document in documents:
        blocks = _split_markdown_by_headers(document) if document.metadata.get("type") == "markdown" else [document]
        for position, chunk in enumerate(splitter.split_documents(blocks), start=1):
            chunk.metadata["chunk"] = position
            if add_header:
                chunk.page_content = f"[{context_header(chunk.metadata)}]\n{chunk.page_content}"
            chunk.metadata["id"] = chunk_id(chunk)
            chunks.append(chunk)
    logger.info("Documentos divididos | documentos=%s | trechos=%s", len(documents), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# 3. Indexar
# ---------------------------------------------------------------------------


def estimate_tokens(chunks: list[Document]) -> int:
    """Estimativa dos tokens enviados ao modelo de embeddings (custo da indexação)."""
    return sum(count_tokens_approximately([c.page_content]) for c in chunks)


def index_folder(folder: str | Path | None = None, target: str | Path | None = None) -> dict:
    """Recria o índice do zero a partir de uma pasta de documentos e o salva em disco."""
    documents = load_folder(folder)
    chunks = split_documents(documents)
    save_index(build_index(chunks), target)
    return {
        "files": len({d.metadata["source"] for d in documents}),
        "documents": len(documents),
        "chunks": len(chunks),
        "estimated_tokens": estimate_tokens(chunks),
    }


def add_chunks(index: FAISS, chunks: list[Document]) -> int:
    """Adiciona apenas os trechos cujo id ainda não está no índice. Devolve quantos entraram."""
    existing = stored_chunks(index)
    new = [c for c in chunks if c.metadata["id"] not in existing]
    if new:
        index.add_documents(new, ids=[c.metadata["id"] for c in new])
    logger.info("Trechos adicionados | novos=%s | ignorados=%s", len(new), len(chunks) - len(new))
    return len(new)


def remove_source(index: FAISS, source: str) -> int:
    """Remove todos os trechos de uma fonte. Devolve quantos saíram."""
    ids = [key for key, chunk in stored_chunks(index).items() if chunk.metadata.get("source") == source]
    if ids:
        index.delete(ids)
    logger.info("Trechos removidos | fonte=%s | removidos=%s", source, len(ids))
    return len(ids)


def reindex_file(index: FAISS, path: str | Path, base: str | Path | None = None) -> dict:
    """Atualiza um arquivo que mudou: remove a versão antiga e adiciona a nova."""
    chunks = split_documents(load_file(path, base or settings.documents_dir))
    source = chunks[0].metadata["source"] if chunks else Path(path).name
    return {"source": source, "removed": remove_source(index, source), "added": add_chunks(index, chunks)}


def index_incremental(folder: str | Path, target: str | Path | None = None) -> dict:
    """Adiciona ao índice existente os documentos de uma pasta, sem recriar o resto.

    Só ADICIONA: se um arquivo já indexado mudou, a versão antiga continua no índice.
    Para atualizar um arquivo editado, use `reindex_file`.
    """
    index = load_index(target)
    chunks = split_documents(load_folder(folder))
    added = add_chunks(index, chunks)
    save_index(index, target)
    return {"chunks_read": len(chunks), "added": added, "total_in_index": index.index.ntotal}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestão dos documentos da TechNova no índice FAISS.")
    parser.add_argument("folder", nargs="?", default=None, help="pasta de documentos (padrão: DOCUMENTS_DIR)")
    parser.add_argument("--incremental", action="store_true", help="adiciona ao índice existente")
    args = parser.parse_args()
    if args.incremental:
        if not index_exists():
            parser.error("Não há índice para atualizar. Rode sem --incremental primeiro.")
        print(index_incremental(args.folder or settings.documents_dir))
    else:
        print(index_folder(args.folder))


if __name__ == "__main__":
    main()
