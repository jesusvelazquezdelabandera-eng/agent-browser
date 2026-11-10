"""Pruebas del panel web."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.models import Auditoria, Campana, CampanaContacto, Contacto, ListaInterna
from app.db.session import get_sessionmaker
from app.main import app

CSV = (
    "nombre;telefono;email;consentimiento;consentimiento_fecha;consentimiento_origen;base_legal\n"
    "María López;600 11 22 33;maria@example.com;sí;2026-10-01;formulario web;\n"
    "Juan Pérez;954123456;;no;;cliente anterior;interes_legitimo\n"
    "Mal;123;;sí;;;\n"
)


def test_flujo_panel() -> None:
    with TestClient(app) as cliente:
        r = cliente.get("/")
        assert r.status_code == 200 and "Banpi" in r.text

        r = cliente.post("/contactos/importar", files={"archivo": ("c.csv", CSV.encode(), "text/csv")})
        assert r.status_code == 200
        assert "Importados 2 nuevos, 0 actualizados, 1 con teléfono no válido" in r.text

        with get_sessionmaker()() as s:
            maria = s.scalar(select(Contacto).where(Contacto.telefono == "+34600112233"))
            assert maria.consentimiento and maria.base_legal == "consentimiento"
            assert maria.consentimiento_origen == "formulario web" and maria.consentimiento_fecha is not None
            juan = s.scalar(select(Contacto).where(Contacto.telefono == "+34954123456"))
            assert not juan.consentimiento and juan.base_legal == "interes_legitimo"

        r = cliente.post("/campanas", data={"nombre": "Otoño"})
        assert "creada con 2 contactos" in r.text
        with get_sessionmaker()() as s:
            campana = s.scalar(select(Campana))
            assert len(s.scalars(select(CampanaContacto)).all()) == 2
        r = cliente.post(f"/campanas/{campana.id}/iniciar")
        assert "activa" in r.text

        r = cliente.post(f"/contactos/{juan.id}/no-llamar")
        assert "no contactar" in r.text
        with get_sessionmaker()() as s:
            assert s.scalar(select(ListaInterna)).telefono == "+34954123456"
            assert s.scalar(select(Auditoria).where(Auditoria.evento == "no_llamar")) is not None

        assert cliente.get("/api/estado").json()["contactos"] == 2
        r = cliente.get("/exportar/auditoria.csv")
        assert r.status_code == 200 and "contacto_importado" in r.text
        assert cliente.get("/exportar/citas.csv").status_code == 200
