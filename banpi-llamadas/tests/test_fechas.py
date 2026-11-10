"""Pruebas del analizador de fechas habladas."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from app.agent.fechas import describir, extraer_fecha_hora, hora_hablada

MADRID = ZoneInfo("Europe/Madrid")
# Martes 10 de noviembre de 2026, 11:00.
AHORA = datetime(2026, 11, 10, 11, 0, tzinfo=MADRID)


@pytest.mark.parametrize(
    "texto,dia,hora",
    [
        ("el jueves a las cinco", (2026, 11, 12), time(17, 0)),
        ("mañana por la mañana", (2026, 11, 11), time(10, 0)),
        ("mañana a las diez y media", (2026, 11, 11), time(10, 30)),
        ("pasado mañana a las 17:30", (2026, 11, 12), time(17, 30)),
        ("el lunes por la tarde", (2026, 11, 16), time(17, 0)),
        ("el 15 de diciembre a las 12", (2026, 12, 15), time(12, 0)),
        ("el día 20 a las once de la mañana", (2026, 11, 20), time(11, 0)),
        ("esta tarde a las seis", (2026, 11, 10), time(18, 0)),
        ("el martes a la una", (2026, 11, 17), time(13, 0)),
        ("el viernes a las 9 de la noche", (2026, 11, 13), time(21, 0)),
        ("el miércoles a las cinco menos cuarto", (2026, 11, 11), time(16, 45)),
    ],
)
def test_dia_y_hora(texto: str, dia: tuple[int, int, int], hora: time) -> None:
    f = extraer_fecha_hora(texto, AHORA)
    assert f.tiene_dia and f.tiene_hora
    assert f.momento is not None
    assert f.momento.date().timetuple()[:3] == dia
    assert f.momento.time() == hora


def test_solo_dia() -> None:
    f = extraer_fecha_hora("mejor el jueves", AHORA)
    assert f.tiene_dia and not f.tiene_hora


def test_solo_hora_hoy() -> None:
    f = extraer_fecha_hora("a las cinco", AHORA)
    assert f.tiene_hora and not f.tiene_dia
    assert f.momento is not None and f.momento.date() == AHORA.date()


def test_solo_hora_pasada_es_manana() -> None:
    f = extraer_fecha_hora("a las 10", AHORA)
    assert f.momento is not None and f.momento.day == 11


def test_vago() -> None:
    f = extraer_fecha_hora("llámame otro día", AHORA)
    assert f.encontrada and f.vago and f.momento is None


def test_sin_fecha() -> None:
    assert not extraer_fecha_hora("tengo un piso en Triana", AHORA).encontrada


def test_luego_es_dentro_de_dos_horas() -> None:
    f = extraer_fecha_hora("llámame luego", AHORA)
    assert f.momento is not None and f.momento.hour == 13


@pytest.mark.parametrize(
    "h,esperado",
    [
        (time(17, 0), "a las cinco de la tarde"),
        (time(10, 30), "a las diez y media de la mañana"),
        (time(13, 0), "a la una del mediodía"),
        (time(16, 45), "a las cinco menos cuarto de la tarde"),
    ],
)
def test_hora_hablada(h: time, esperado: str) -> None:
    assert hora_hablada(h) == esperado


def test_describir() -> None:
    f = extraer_fecha_hora("el jueves a las cinco", AHORA)
    assert describir(f, AHORA) == "el jueves 12 a las cinco de la tarde"
    f = extraer_fecha_hora("mañana a las diez", AHORA)
    assert describir(f, AHORA) == "mañana a las diez de la mañana"
