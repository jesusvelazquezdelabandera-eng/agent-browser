"""Conexión a la base de datos y utilidades de sesión."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.db.models import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker[Session]] = None


def crear_engine(url: str) -> Engine:
    """Crea el engine de SQLAlchemy con ajustes adecuados para SQLite."""
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
        else:
            # Crea la carpeta del archivo si no existe.
            ruta = url.split("sqlite:///", 1)[-1]
            Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(conexion, _registro) -> None:  # type: ignore[no-untyped-def]
            cursor = conexion.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_db(url: Optional[str] = None) -> sessionmaker[Session]:
    """Inicializa el engine global y crea las tablas que falten."""
    global _engine, _SessionLocal
    _engine = crear_engine(url or get_settings().database_url)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _SessionLocal


def get_sessionmaker() -> sessionmaker[Session]:
    """Devuelve la fábrica de sesiones, inicializándola si hace falta."""
    if _SessionLocal is None:
        return init_db()
    return _SessionLocal


@contextmanager
def sesion() -> Iterator[Session]:
    """Sesión con commit automático al salir y rollback si hay error."""
    s = get_sessionmaker()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def como_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """SQLite devuelve fechas sin zona horaria: las marcamos como UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
