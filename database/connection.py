"""PostgreSQL connection pool backed by DATABASE_URL in .env."""

import os
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

_pool: pool.ThreadedConnectionPool | None = None


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        database_url = os.environ["DATABASE_URL"]
        _pool = pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=database_url)
    return _pool


@contextmanager
def get_conn():
    """Yield a psycopg2 connection from the pool; return it on exit."""
    connection_pool = _get_pool()
    conn = connection_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        connection_pool.putconn(conn)


def apply_schema() -> None:
    """Run schema.sql against the database (idempotent — uses IF NOT EXISTS)."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        sql = f.read()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("Schema applied.")


if __name__ == "__main__":
    apply_schema()
