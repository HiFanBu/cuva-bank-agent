from __future__ import annotations

import sqlite3

import pytest

from app.data.seed import seed_accounts
from app.db import get_connection, init_db
from app.rag.ingest import seed_documents


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = get_connection(":memory:")
    init_db(c)
    seed_accounts(c)
    seed_documents(c)
    yield c
    c.close()
