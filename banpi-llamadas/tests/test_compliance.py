"""Pruebas del módulo de cumplimiento."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.compliance import (
    Motivo,
    ReglasCumplimiento,
    aviso_inicial,
    cargar_festivos_extra,
    cargar_lista_telefonos,
    domingo_de_pascua,
    festivos_nacionales,
    normalizar_telefono,
    se_opone_a_grabacion,
)

MADRID = ZoneInfo("Europe/Madrid")


@dataclass
class C:
    telefono: str = "600112233"
    consentimiento: bool = True
    base_legal: str = "consentimiento"
    no_llamar: bool = False


def madrid(anio: int, mes: int, dia: int, h: int, m: int = 0) -> datetime:
    return datetime(anio, mes, dia, h, m, tzinfo=MADRID)


@pytest.fixture
def reglas() -> ReglasCumplimiento:
    return ReglasCumplimiento(zona=MADRID)


# Martes 10 de noviembre de 2026, día laborable normal.
LABORABLE = madrid(2026, 11, 10, 11)


class TestTelefonos:
    @pytest.mark.parametrize(
        "entrada,esperado",
        [
            ("600 11 22 33", "+34600112233"),
            ("954-123-456", "+34954123456"),
            ("0034 600112233", "+34600112233"),
            ("+34 600 112 233", "+34600112233"),
            ("34600112233", "+34600112233"),
            ("+44 20 7946 0958", "+442079460958"),
        ],
    )
    def test_normalizar(self, entrada: str, esperado: str) -> None:
        assert normalizar_telefono(entrada) == esperado

    def test_cargar_lista(self, tmp_path: Path) -> None:
        f = tmp_path / "robinson.txt"
        f.write_text("# comentario\n600 11 22 33\n954123456;extra\n\n", encoding="utf-8")
        assert cargar_lista_telefonos(f) == {"+34600112233", "+34954123456"}

    def test_lista_inexistente(self, tmp_path: Path) -> None:
        assert cargar_lista_telefonos(tmp_path / "no.txt") == set()


class TestFestivos:
    def test_pascua(self) -> None:
        assert domingo_de_pascua(2026) == date(2026, 4, 5)
        assert domingo_de_pascua(2027) == date(2027, 3, 28)
        assert domingo_de_pascua(2024) == date(2024, 3, 31)

    def test_viernes_santo_es_festivo(self) -> None:
        assert date(2026, 4, 3) in festivos_nacionales(2026)
        assert date(2027, 3, 26) in festivos_nacionales(2027)

    def test_fijos(self) -> None:
        f = festivos_nacionales(2026)
        for mes, dia in [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]:
            assert date(2026, mes, dia) in f
        assert len(f) == 10

    def test_festivos_extra(self, tmp_path: Path) -> None:
        f = tmp_path / "festivos.txt"
        f.write_text("2027-02-28  # Día de Andalucía\n", encoding="utf-8")
        assert cargar_festivos_extra(f) == {date(2027, 2, 28)}


class TestHorario:
    @pytest.mark.parametrize(
        "momento,esperado",
        [
            (madrid(2026, 11, 10, 8, 59), False),
            (madrid(2026, 11, 10, 9, 0), True),
            (madrid(2026, 11, 10, 20, 59), True),
            (madrid(2026, 11, 10, 21, 0), False),
            (madrid(2026, 11, 14, 11), False),  # sábado
            (madrid(2026, 11, 15, 11), False),  # domingo
            (madrid(2026, 12, 8, 11), False),  # Inmaculada (martes)
            (madrid(2026, 4, 3, 11), False),  # Viernes Santo
        ],
    )
    def test_en_horario(self, reglas: ReglasCumplimiento, momento: datetime, esperado: bool) -> None:
        assert reglas.en_horario(momento) is esperado

    def test_convierte_desde_utc(self, reglas: ReglasCumplimiento) -> None:
        # 8:30 UTC en invierno son las 9:30 en Madrid: permitido.
        assert reglas.en_horario(datetime(2026, 11, 10, 8, 30, tzinfo=timezone.utc))
        # 19:30 UTC en verano son las 21:30 en Madrid: no permitido.
        assert not reglas.en_horario(datetime(2026, 7, 7, 19, 30, tzinfo=timezone.utc))

    def test_festivo_extra(self) -> None:
        r = ReglasCumplimiento(zona=MADRID, festivos_extra={date(2026, 11, 10)})
        assert not r.en_horario(LABORABLE)

    def test_siguiente_ventana_viernes_noche(self, reglas: ReglasCumplimiento) -> None:
        viernes = madrid(2026, 11, 13, 21, 30)
        assert reglas.siguiente_ventana(viernes) == madrid(2026, 11, 16, 9).astimezone(timezone.utc)

    def test_siguiente_ventana_madrugada(self, reglas: ReglasCumplimiento) -> None:
        assert reglas.siguiente_ventana(madrid(2026, 11, 10, 7)) == madrid(2026, 11, 10, 9).astimezone(
            timezone.utc
        )

    def test_siguiente_ventana_puente(self, reglas: ReglasCumplimiento) -> None:
        # Lunes 7 de diciembre por la noche; el 8 es festivo: se salta al miércoles 9.
        assert reglas.siguiente_ventana(madrid(2026, 12, 7, 22)) == madrid(2026, 12, 9, 9).astimezone(
            timezone.utc
        )

    def test_siguiente_ventana_dentro_de_horario(self, reglas: ReglasCumplimiento) -> None:
        assert reglas.siguiente_ventana(LABORABLE) == LABORABLE.astimezone(timezone.utc)


class TestComprobar:
    def test_permitido(self, reglas: ReglasCumplimiento) -> None:
        d = reglas.comprobar(C(), [], LABORABLE)
        assert d.permitido and d.motivo is Motivo.PERMITIDO

    def test_sin_consentimiento(self, reglas: ReglasCumplimiento) -> None:
        d = reglas.comprobar(C(consentimiento=False, base_legal=""), [], LABORABLE)
        assert not d.permitido and d.motivo is Motivo.SIN_CONSENTIMIENTO and d.permanente

    def test_interes_legitimo_va_a_humano(self, reglas: ReglasCumplimiento) -> None:
        d = reglas.comprobar(C(consentimiento=False, base_legal="interes_legitimo"), [], LABORABLE)
        assert not d.permitido and d.motivo is Motivo.REQUIERE_HUMANO

    def test_no_llamar(self, reglas: ReglasCumplimiento) -> None:
        d = reglas.comprobar(C(no_llamar=True), [], LABORABLE)
        assert d.motivo is Motivo.NO_LLAMAR

    def test_robinson(self) -> None:
        r = ReglasCumplimiento(zona=MADRID, robinson={"+34600112233"})
        assert r.comprobar(C(telefono="600 112 233"), [], LABORABLE).motivo is Motivo.ROBINSON

    def test_lista_interna(self) -> None:
        r = ReglasCumplimiento(zona=MADRID, lista_interna={"+34600112233"})
        assert r.comprobar(C(), [], LABORABLE).motivo is Motivo.LISTA_INTERNA

    def test_exclusion_tiene_prioridad_sobre_horario(self) -> None:
        r = ReglasCumplimiento(zona=MADRID, robinson={"+34600112233"})
        assert r.comprobar(C(), [], madrid(2026, 11, 15, 11)).motivo is Motivo.ROBINSON

    def test_telefono_invalido(self, reglas: ReglasCumplimiento) -> None:
        assert reglas.comprobar(C(telefono="123"), [], LABORABLE).motivo is Motivo.TELEFONO_INVALIDO

    def test_fuera_de_horario_da_reintento(self, reglas: ReglasCumplimiento) -> None:
        d = reglas.comprobar(C(), [], madrid(2026, 11, 14, 11))
        assert d.motivo is Motivo.FUERA_DE_HORARIO and not d.permanente
        assert d.reintentar_desde == madrid(2026, 11, 16, 9).astimezone(timezone.utc)

    def test_max_tres_intentos_en_siete_dias(self, reglas: ReglasCumplimiento) -> None:
        intentos = [LABORABLE - timedelta(days=d) for d in (1, 2, 6)]
        d = reglas.comprobar(C(), intentos, LABORABLE)
        assert d.motivo is Motivo.MAX_INTENTOS
        # El más antiguo (hace 6 días) sale de la ventana en 1 día.
        assert d.reintentar_desde is not None
        assert d.reintentar_desde >= (LABORABLE + timedelta(days=1)).astimezone(timezone.utc)

    def test_intentos_antiguos_no_cuentan(self, reglas: ReglasCumplimiento) -> None:
        intentos = [LABORABLE - timedelta(days=d) for d in (1, 2, 8)]
        assert reglas.comprobar(C(), intentos, LABORABLE).permitido

    def test_dos_intentos_permitido(self, reglas: ReglasCumplimiento) -> None:
        intentos = [LABORABLE - timedelta(hours=h) for h in (2, 30)]
        assert reglas.comprobar(C(), intentos, LABORABLE).permitido


class TestGrabacionYAviso:
    @pytest.mark.parametrize(
        "texto",
        [
            "No quiero que me grabe",
            "no me grabes, por favor",
            "Prefiero que no se grabe la conversación",
            "Me opongo a la grabación",
            "no autorizo la grabación",
            "sin grabar, gracias",
        ],
    )
    def test_detecta_oposicion(self, texto: str) -> None:
        assert se_opone_a_grabacion(texto)

    @pytest.mark.parametrize("texto", ["sí, dígame", "vale, no pasa nada", "grabe lo que quiera"])
    def test_no_detecta_falsos_positivos(self, texto: str) -> None:
        assert not se_opone_a_grabacion(texto)

    def test_aviso_inicial_contiene_todo(self) -> None:
        t = aviso_inicial("Álex", "Banpi Domótica", "Sevilla", "María López")
        assert "María" in t and "López" not in t
        assert "inteligencia artificial" in t
        assert "Banpi Domótica" in t and "Sevilla" in t
        assert "grabarse" in t
        assert "no le llamemos más" in t
