"""Acesso somente leitura ao banco da loja (SQLite).

A TechNova é a loja online fictícia de eletrônicos dos cursos de LangChain e
LangGraph. Neste curso o banco `data/store.db` (clientes, produtos, pedidos e
itens) é FORNECIDO com o material e aparece na ferramenta de pedidos do agente
(módulo 03). Este módulo apenas lê o banco.

Uso:
    from src.database import query
    query("SELECT * FROM orders WHERE id = ?", ("PED-1001",))
"""

import sqlite3
from pathlib import Path
from typing import Any

from src.config import settings


def query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Executa um SELECT e devolve as linhas como lista de dicionários.

    O modo `ro` abre o banco somente para leitura e impede que o SQLite crie um
    arquivo vazio por engano quando o caminho está errado.
    """
    if not Path(settings.store_db_path).exists():
        raise FileNotFoundError(
            f"Banco da loja não encontrado em '{settings.store_db_path}'. Execute a partir da raiz do projeto."
        )
    connection = sqlite3.connect(f"file:{settings.store_db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]
    finally:
        connection.close()
