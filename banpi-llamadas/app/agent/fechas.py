"""Interpretación de fechas y horas habladas en español ("el martes a las cinco").

Es un analizador por reglas, deliberadamente sencillo y predecible: cubre las
formas más habituales en una llamada. Lo que no entiende se guarda como texto
para que un comercial lo revise.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional

DIAS_SEMANA = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
DIAS_SEMANA_HABLADOS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]  # fmt: skip
NUMEROS = {
    "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
}  # fmt: skip
NUMEROS_HABLADOS = {v: k for k, v in NUMEROS.items() if k != "uno"}


@dataclass(frozen=True)
class FechaHablada:
    """Resultado de interpretar una fecha/hora en texto libre."""

    momento: Optional[datetime]
    tiene_dia: bool
    tiene_hora: bool
    vago: bool = False  # "luego", "más tarde": sin día ni hora concretos

    @property
    def encontrada(self) -> bool:
        return self.tiene_dia or self.tiene_hora or self.vago


def _normalizar(texto: str) -> str:
    sin = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sin)


def _num(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    return NUMEROS.get(token)


_NUM = r"(\d{1,2}|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce)"


def _extraer_dia(t: str, hoy: date) -> Optional[date]:
    if "pasado manana" in t:
        return hoy + timedelta(days=2)
    # "mañana" como día, no "por la mañana" / "de la mañana" / "esta mañana".
    if re.search(r"(?<!la )(?<!esta )\bmanana\b", t):
        return hoy + timedelta(days=1)
    if re.search(r"\bhoy\b|\besta (tarde|manana|noche)\b|\bluego\b|\bmas tarde\b", t):
        return hoy

    # "el 15 de diciembre", "el día 15"
    m = re.search(r"\b(?:el )?(?:dia )?(\d{1,2}) de (" + "|".join(MESES) + r")\b", t)
    if m:
        dia, mes = int(m.group(1)), MESES.index(m.group(2)) + 1
        anio = hoy.year
        try:
            candidato = date(anio, mes, dia)
            if candidato < hoy:
                candidato = date(anio + 1, mes, dia)
            return candidato
        except ValueError:
            return None
    m = re.search(r"\bel dia (\d{1,2})\b", t)
    if m:
        dia = int(m.group(1))
        for meses in range(0, 3):
            mes = (hoy.month - 1 + meses) % 12 + 1
            anio = hoy.year + (hoy.month - 1 + meses) // 12
            try:
                candidato = date(anio, mes, dia)
            except ValueError:
                continue
            if candidato >= hoy:
                return candidato

    for i, nombre in enumerate(DIAS_SEMANA):
        if re.search(rf"\b{nombre}\b", t):
            delta = (i - hoy.weekday()) % 7
            if delta == 0:
                delta = 7
            elif re.search(rf"{nombre} (que viene|proximo)|proximo {nombre}", t) and delta < 3:
                # "el jueves que viene" dicho un martes suele significar el de la semana próxima
                delta += 7
            return hoy + timedelta(days=delta)

    if re.search(r"semana que viene|proxima semana", t):
        return hoy + timedelta(days=7 - hoy.weekday())  # lunes próximo
    return None


def _extraer_hora(t: str) -> Optional[time]:
    franja_tarde = bool(re.search(r"de la tarde|por la tarde|esta tarde|de la noche|por la noche", t))
    franja_manana = bool(re.search(r"de la manana|por la manana|esta manana", t))

    m = re.search(r"\b(\d{1,2})[:.h](\d{2})\b", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(rf"\b(?:a las|a la|sobre las|hacia las|las) {_NUM}(?:h\b| horas)?(?: y (media|cuarto))?", t)
        if not m:
            m = re.search(r"\b(\d{1,2}) ?h\b", t)
        if m:
            h_ = _num(m.group(1))
            if h_ is None:
                return None
            h = h_
            extra = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            mi = 30 if extra == "media" else 15 if extra == "cuarto" else 0
            if re.search(rf"{re.escape(m.group(0))} menos cuarto", t):
                h, mi = h - 1, 45
        elif "mediodia" in t:
            return time(13, 0)
        elif franja_tarde:
            return time(17, 0)
        elif franja_manana:
            return time(10, 0)
        else:
            return None

    if not 0 <= h <= 23 or not 0 <= mi <= 59:
        return None
    if h < 12 and (franja_tarde or (h <= 8 and not franja_manana)):
        # "a las cinco" en contexto comercial casi siempre es por la tarde.
        h += 12
    return time(h, mi)


def extraer_fecha_hora(texto: str, ahora: datetime) -> FechaHablada:
    """Interpreta día y hora en `texto` tomando `ahora` (con zona) como referencia."""
    t = _normalizar(texto)
    hoy = ahora.date()
    dia = _extraer_dia(t, hoy)
    hora = _extraer_hora(t)
    vago = bool(re.search(r"\bluego\b|\bmas tarde\b|\botro dia\b|\botro rato\b|\ben otro momento\b", t))

    if dia is None and hora is None:
        if vago:
            return FechaHablada(None, False, False, vago=True)
        return FechaHablada(None, False, False)

    if dia is None and hora is not None:
        # Solo hora: hoy si aún queda margen, si no mañana.
        candidato = datetime.combine(hoy, hora, tzinfo=ahora.tzinfo)
        if candidato < ahora + timedelta(minutes=30):
            candidato += timedelta(days=1)
        return FechaHablada(candidato, tiene_dia=False, tiene_hora=True)

    assert dia is not None
    hora_final = hora or time(10, 0)
    momento = datetime.combine(dia, hora_final, tzinfo=ahora.tzinfo)
    if re.search(r"\bluego\b|\bmas tarde\b", t) and hora is None and dia == hoy:
        momento = ahora + timedelta(hours=2)
        return FechaHablada(momento, tiene_dia=True, tiene_hora=False, vago=True)
    return FechaHablada(momento, tiene_dia=True, tiene_hora=hora is not None)


def hora_hablada(t: time) -> str:
    """Hora en palabras, natural para la voz: 'a las cinco y media de la tarde'."""
    h12 = t.hour % 12 or 12
    nombre = NUMEROS_HABLADOS.get(h12, str(h12))
    art = "a la" if h12 == 1 else "a las"
    minutos = {0: "", 15: " y cuarto", 30: " y media", 45: " menos cuarto"}.get(t.minute)
    if t.minute == 45:
        h_sig = (t.hour + 1) % 12 or 12
        nombre = NUMEROS_HABLADOS.get(h_sig, str(h_sig))
        art = "a la" if h_sig == 1 else "a las"
    if minutos is None:
        minutos = f" y {t.minute}"
    if t.hour < 12:
        franja = " de la mañana"
    elif t.hour < 14:
        franja = " del mediodía"
    elif t.hour < 21:
        franja = " de la tarde"
    else:
        franja = " de la noche"
    return f"{art} {nombre}{minutos}{franja}"


def describir(f: FechaHablada, ahora: datetime) -> str:
    """Describe la fecha para confirmarla en voz alta: 'el martes 17 a las cinco de la tarde'."""
    if f.momento is None:
        return "más adelante"
    d = f.momento.date()
    if d == ahora.date():
        dia_txt = "hoy"
    elif d == ahora.date() + timedelta(days=1):
        dia_txt = "mañana"
    else:
        dia_txt = f"el {DIAS_SEMANA_HABLADOS[d.weekday()]} {d.day}"
        if d.month != ahora.month:
            dia_txt += f" de {MESES[d.month - 1]}"
    if not f.tiene_hora:
        return dia_txt
    return f"{dia_txt} {hora_hablada(f.momento.time())}"
