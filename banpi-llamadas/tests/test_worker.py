"""Pruebas del trabajador de campañas (cumplimiento, reintentos y resultados)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from app.agent.dialog import Accion, Dialogo, Resultado
from app.campaigns.worker import EjecutorLlamadas, OyenteBD, TrabajadorCampanas, aplicar_resultado
from app.config import Settings
from app.db.models import Auditoria, Campana, CampanaContacto, Cita, Contacto, ListaInterna, Llamada
from app.db.session import init_db
from app.telephony.base import EstadoMarcacion, ResultadoMarcacion

MADRID = ZoneInfo("Europe/Madrid")
LABORABLE = datetime(2026, 11, 10, 11, 0, tzinfo=MADRID).astimezone(timezone.utc)
SABADO = datetime(2026, 11, 14, 11, 0, tzinfo=MADRID).astimezone(timezone.utc)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        robinson_file=tmp_path / "robinson.txt",
        festivos_extra_file=tmp_path / "festivos.txt",
        llamadas_simultaneas=5,
        grabar_llamadas=False,
    )


@pytest.fixture
def fabrica():  # type: ignore[no-untyped-def]
    return init_db("sqlite://")


def crear(fabrica, contactos: list[dict]) -> int:  # type: ignore[no-untyped-def]
    with fabrica.begin() as s:
        campana = Campana(nombre="Prueba", estado="activa")
        s.add(campana)
        s.flush()
        for datos in contactos:
            c = Contacto(**{"consentimiento": True, "base_legal": "consentimiento", **datos})
            s.add(c)
            s.flush()
            s.add(CampanaContacto(campana_id=campana.id, contacto_id=c.id))
        return campana.id


class EjecutorFalso:
    def __init__(self) -> None:
        self.llamados: list[int] = []

    async def __call__(self, cc_id: int) -> None:
        self.llamados.append(cc_id)


def estados(fabrica) -> dict[str, tuple[str, str]]:  # type: ignore[no-untyped-def]
    with fabrica() as s:
        return {cc.contacto.telefono: (cc.estado, cc.motivo) for cc in s.scalars(select(CampanaContacto))}


def test_solo_llama_a_quien_se_permite(fabrica, settings: Settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "robinson.txt").write_text("600000003\n", encoding="utf-8")
    crear(
        fabrica,
        [
            {"telefono": "600000001"},
            {"telefono": "600000002", "consentimiento": False, "base_legal": ""},
            {"telefono": "600000003"},
            {"telefono": "600000004", "consentimiento": False, "base_legal": "interes_legitimo"},
            {"telefono": "600000005", "no_llamar": True},
        ],
    )
    ejecutor = EjecutorFalso()
    t = TrabajadorCampanas(fabrica, ejecutor, settings, reloj=lambda: LABORABLE)

    async def correr() -> int:
        n = await t.ciclo()
        await t.esperar_llamadas()
        return n

    assert asyncio.run(correr()) == 1
    e = estados(fabrica)
    assert e["600000001"][0] == "en_curso"
    assert e["600000002"] == ("excluido", "sin_consentimiento")
    assert e["600000003"] == ("excluido", "lista_robinson")
    assert e["600000004"] == ("cola_humana", "requiere_humano")
    assert e["600000005"] == ("excluido", "no_llamar")
    with fabrica() as s:
        assert s.scalar(select(Contacto).where(Contacto.telefono == "600000004")).estado == "cola_humana"
        assert len(s.scalars(select(Auditoria).where(Auditoria.evento == "decision_cumplimiento")).all()) == 5


def test_fuera_de_horario_programa_lunes(fabrica, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    crear(fabrica, [{"telefono": "600000001"}])
    ejecutor = EjecutorFalso()
    t = TrabajadorCampanas(fabrica, ejecutor, settings, reloj=lambda: SABADO)
    assert asyncio.run(t.ciclo()) == 0
    with fabrica() as s:
        cc = s.scalar(select(CampanaContacto))
        assert cc.estado == "reintentar" and cc.motivo == "fuera_de_horario"
        lunes = datetime(2026, 11, 16, 9, 0, tzinfo=MADRID).astimezone(timezone.utc)
        assert cc.proximo_intento.replace(tzinfo=timezone.utc) == lunes


def test_respeta_tres_intentos_en_siete_dias(fabrica, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    crear(fabrica, [{"telefono": "600000001"}])
    with fabrica.begin() as s:
        c = s.scalar(select(Contacto))
        for dias in (1, 2, 3):
            s.add(Llamada(contacto_id=c.id, inicio=LABORABLE - timedelta(days=dias), resultado="no_contesta"))
    ejecutor = EjecutorFalso()
    t = TrabajadorCampanas(fabrica, ejecutor, settings, reloj=lambda: LABORABLE)
    assert asyncio.run(t.ciclo()) == 0
    assert estados(fabrica)["600000001"] == ("reintentar", "max_intentos")


def test_no_lanza_mas_de_las_simultaneas(fabrica, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    crear(fabrica, [{"telefono": f"60000000{i}"} for i in range(1, 6)])
    t = TrabajadorCampanas(fabrica, EjecutorFalso(), replace(settings, llamadas_simultaneas=2), reloj=lambda: LABORABLE)
    assert asyncio.run(t.ciclo()) == 2


def _cc_y_llamada(fabrica, s):  # type: ignore[no-untyped-def]
    cc = s.scalar(select(CampanaContacto))
    llamada = Llamada(contacto_id=cc.contacto_id, campana_id=cc.campana_id, inicio=LABORABLE)
    s.add(llamada)
    s.flush()
    return cc, llamada


@pytest.mark.parametrize(
    "resultado,estado_cc,estado_contacto",
    [
        ("cita", "completado", "cita"),
        ("no_interesa", "completado", "no_interesa"),
        ("no_llamar", "completado", "no_llamar"),
        ("no_contesta", "reintentar", "pendiente"),
        ("sin_cita", "completado", "contactado"),
    ],
)
def test_aplicar_resultado(fabrica, settings: Settings, resultado: str, estado_cc: str, estado_contacto: str) -> None:  # type: ignore[no-untyped-def]
    crear(fabrica, [{"telefono": "600000001"}])
    with fabrica.begin() as s:
        cc, llamada = _cc_y_llamada(fabrica, s)
        aplicar_resultado(s, cc, llamada, resultado, None, settings, LABORABLE)
    with fabrica() as s:
        cc = s.scalar(select(CampanaContacto))
        assert cc.estado == estado_cc and cc.contacto.estado == estado_contacto
        if resultado == "no_llamar":
            assert cc.contacto.no_llamar
            assert s.scalar(select(ListaInterna)).telefono == "+34600000001"
        if resultado == "no_contesta":
            assert cc.proximo_intento.replace(tzinfo=timezone.utc) == LABORABLE + timedelta(minutes=settings.reintento_minutos)


def test_sin_contacto_tras_max_intentos(fabrica, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    crear(fabrica, [{"telefono": "600000001"}])
    for _ in range(3):
        with fabrica.begin() as s:
            cc, llamada = _cc_y_llamada(fabrica, s)
            aplicar_resultado(s, cc, llamada, "no_contesta", None, settings, LABORABLE)
    assert estados(fabrica)["600000001"] == ("completado", "sin_contacto_tras_intentos")


def test_reprogramar_usa_hora_pedida(fabrica, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    crear(fabrica, [{"telefono": "600000001"}])
    d = Dialogo("", settings=settings, reloj=lambda: LABORABLE.astimezone(MADRID))
    d.inicio()
    d.procesar("ahora no puedo, llámame mañana a las cinco")
    with fabrica.begin() as s:
        cc, llamada = _cc_y_llamada(fabrica, s)
        aplicar_resultado(s, cc, llamada, "reprogramar", d.datos, settings, LABORABLE)
    with fabrica() as s:
        cc = s.scalar(select(CampanaContacto))
        esperado = datetime(2026, 11, 11, 17, 0, tzinfo=MADRID).astimezone(timezone.utc)
        assert cc.estado == "reintentar" and cc.proximo_intento.replace(tzinfo=timezone.utc) == esperado


def test_oyente_guarda_cita_y_no_llamar(fabrica, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    crear(fabrica, [{"telefono": "600000001"}, {"telefono": "600000002"}])
    with fabrica.begin() as s:
        c1, c2 = s.scalars(select(Contacto)).all()
        l1, l2 = Llamada(contacto_id=c1.id), Llamada(contacto_id=c2.id)
        s.add_all([l1, l2])
        s.flush()
        ids = (c1.id, l1.id, c2.id, l2.id)

    d = Dialogo("", settings=settings, reloj=lambda: LABORABLE.astimezone(MADRID))
    d.inicio()
    for frase in ("sí", "un piso", "las persianas", "seguridad", "el jueves a las cinco"):
        d.procesar(frase)
    OyenteBD(fabrica, ids[1], ids[0]).al_accion(Accion.GUARDAR_CITA, d)
    OyenteBD(fabrica, ids[3], ids[2]).al_accion(Accion.MARCAR_NO_LLAMAR, d)
    OyenteBD(fabrica, ids[1], ids[0]).al_hablar("Hola", "saludo", 900)

    with fabrica() as s:
        cita = s.scalar(select(Cita))
        assert cita.tipo == "visita" and cita.cuando_texto == "el jueves 12 a las cinco de la tarde"
        assert s.get(Contacto, ids[2]).no_llamar
        assert s.get(Llamada, ids[1]).transcripciones[0].latencia_ms == 900


class TelefoniaFalsa:
    def __init__(self, estado: EstadoMarcacion) -> None:
        self.estado = estado
        self.numeros: list[str] = []

    async def originar(self, numero: str, caller_id: str = "", timeout_s: int = 30) -> ResultadoMarcacion:
        self.numeros.append(numero)
        return ResultadoMarcacion(self.estado, detalle="prueba")


def test_ejecutor_no_contesta_programa_reintento(fabrica, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    crear(fabrica, [{"telefono": "600000001"}])
    tel = TelefoniaFalsa(EstadoMarcacion.OCUPADO)
    ejecutor = EjecutorLlamadas(fabrica, tel, None, None, None, settings)
    with fabrica() as s:
        cc_id = s.scalar(select(CampanaContacto)).id
    asyncio.run(ejecutor(cc_id))
    assert tel.numeros == ["600000001"]
    with fabrica() as s:
        llamada = s.scalar(select(Llamada))
        assert llamada.resultado == "ocupado" and llamada.error == "prueba"
        assert s.scalar(select(CampanaContacto)).estado == "reintentar"
        eventos = [a.evento for a in s.scalars(select(Auditoria))]
        assert eventos == ["marcacion", "llamada_finalizada"]


def test_resultados_conocidos() -> None:
    # Todos los resultados del diálogo tienen tratamiento explícito o el genérico.
    assert {r.value for r in Resultado} >= {"cita", "reprogramar", "no_interesa", "no_llamar"}
