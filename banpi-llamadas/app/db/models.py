"""Modelos de base de datos (SQLite + SQLAlchemy 2).

Tablas:
- `contactos`: personas a las que se puede llamar, con su consentimiento.
- `lista_interna`: números que han pedido no ser llamados (lista interna de no llamar).
- `campanas` y `campana_contactos`: campañas y el estado de cada contacto en ellas.
- `llamadas`: cada intento de llamada, con su resultado.
- `transcripciones`: cada turno hablado en una llamada.
- `citas`: visitas o llamadas con comercial agendadas.
- `auditoria`: registro inmutable de decisiones y eventos para auditoría.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def ahora_utc() -> datetime:
    """Fecha y hora actual en UTC (con zona horaria)."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarativa común."""


class Contacto(Base):
    __tablename__ = "contactos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nombre: Mapped[str] = mapped_column(String(200), default="")
    telefono: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(200), default="")
    # Consentimiento expreso para recibir llamadas comerciales automatizadas.
    consentimiento: Mapped[bool] = mapped_column(Boolean, default=False)
    consentimiento_fecha: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    consentimiento_origen: Mapped[str] = mapped_column(String(300), default="")
    # Base legal registrada: "consentimiento", "interes_legitimo" o vacío.
    base_legal: Mapped[str] = mapped_column(String(50), default="")
    no_llamar: Mapped[bool] = mapped_column(Boolean, default=False)
    # pendiente | contactado | cita | no_interesa | no_llamar | cola_humana | excluido
    estado: Mapped[str] = mapped_column(String(30), default="pendiente")
    notas: Mapped[str] = mapped_column(Text, default="")
    creado: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora_utc)

    llamadas: Mapped[list["Llamada"]] = relationship(back_populates="contacto")


class ListaInterna(Base):
    """Lista interna de no llamar (independiente de la lista Robinson)."""

    __tablename__ = "lista_interna"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telefono: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    motivo: Mapped[str] = mapped_column(String(300), default="")
    fecha: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora_utc)


class Campana(Base):
    __tablename__ = "campanas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nombre: Mapped[str] = mapped_column(String(200))
    # borrador | activa | pausada | terminada
    estado: Mapped[str] = mapped_column(String(20), default="borrador")
    creada: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora_utc)

    contactos: Mapped[list["CampanaContacto"]] = relationship(back_populates="campana")


class CampanaContacto(Base):
    """Estado de un contacto dentro de una campaña."""

    __tablename__ = "campana_contactos"
    __table_args__ = (UniqueConstraint("campana_id", "contacto_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    campana_id: Mapped[int] = mapped_column(ForeignKey("campanas.id"), index=True)
    contacto_id: Mapped[int] = mapped_column(ForeignKey("contactos.id"), index=True)
    # pendiente | en_curso | reintentar | completado | excluido | cola_humana
    estado: Mapped[str] = mapped_column(String(20), default="pendiente")
    proximo_intento: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    motivo: Mapped[str] = mapped_column(String(300), default="")

    campana: Mapped[Campana] = relationship(back_populates="contactos")
    contacto: Mapped[Contacto] = relationship()


class Llamada(Base):
    __tablename__ = "llamadas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contacto_id: Mapped[int] = mapped_column(ForeignKey("contactos.id"), index=True)
    campana_id: Mapped[Optional[int]] = mapped_column(ForeignKey("campanas.id"))
    inicio: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora_utc)
    fin: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    # no_contesta | ocupado | fallo | cita | reprogramar | no_interesa | no_llamar | sin_cita | colgada
    resultado: Mapped[str] = mapped_column(String(30), default="en_curso")
    estado_final: Mapped[str] = mapped_column(String(30), default="")
    grabacion_permitida: Mapped[bool] = mapped_column(Boolean, default=False)
    grabacion_ruta: Mapped[str] = mapped_column(String(300), default="")
    latencia_media_ms: Mapped[Optional[int]] = mapped_column(Integer)
    rellamar_texto: Mapped[str] = mapped_column(String(200), default="")
    rellamar_en: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    datos: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")

    contacto: Mapped[Contacto] = relationship(back_populates="llamadas")
    transcripciones: Mapped[list["Transcripcion"]] = relationship(
        back_populates="llamada", order_by="Transcripcion.id"
    )


class Transcripcion(Base):
    __tablename__ = "transcripciones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    llamada_id: Mapped[int] = mapped_column(ForeignKey("llamadas.id"), index=True)
    rol: Mapped[str] = mapped_column(String(10))  # "agente" o "cliente"
    texto: Mapped[str] = mapped_column(Text)
    estado: Mapped[str] = mapped_column(String(30), default="")
    momento: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora_utc)
    latencia_ms: Mapped[Optional[int]] = mapped_column(Integer)

    llamada: Mapped[Llamada] = relationship(back_populates="transcripciones")


class Cita(Base):
    __tablename__ = "citas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contacto_id: Mapped[int] = mapped_column(ForeignKey("contactos.id"), index=True)
    llamada_id: Mapped[Optional[int]] = mapped_column(ForeignKey("llamadas.id"))
    tipo: Mapped[str] = mapped_column(String(20))  # "visita" o "llamada"
    cuando_texto: Mapped[str] = mapped_column(String(200), default="")
    cuando: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resumen: Mapped[str] = mapped_column(Text, default="")
    creada: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora_utc)

    contacto: Mapped[Contacto] = relationship()


class Auditoria(Base):
    """Registro de auditoría. Solo se insertan filas, nunca se modifican."""

    __tablename__ = "auditoria"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    momento: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=ahora_utc, index=True)
    evento: Mapped[str] = mapped_column(String(50), index=True)
    contacto_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    llamada_id: Mapped[Optional[int]] = mapped_column(Integer)
    telefono: Mapped[str] = mapped_column(String(20), default="")
    detalle: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
