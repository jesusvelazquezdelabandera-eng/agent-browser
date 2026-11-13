"""Reglas legales y de consentimiento para llamadas comerciales en España.

AVISO: este módulo implementa una interpretación técnica razonable de la normativa,
pero no sustituye al asesoramiento legal. Revisa con un profesional el RGPD, la
LOPDGDD, la Ley 11/2022 General de Telecomunicaciones (art. 66), la Circular
1/2023 de la AEPD, la Ley 10/2025 de servicios de atención a la clientela, el
texto refundido de la Ley General para la Defensa de los Consumidores y Usuarios
y el Reglamento (UE) 2024/1689 de Inteligencia Artificial (art. 50).

Decisiones de diseño:
- Álex es un sistema automático sin intervención humana. El art. 66.1.a de la
  Ley 11/2022 exige consentimiento previo para ese tipo de llamadas comerciales,
  así que el interés legítimo NO basta para que llame la IA: esos contactos se
  derivan a una cola para que los llame una persona del equipo comercial.
- Horario: lunes a viernes de 9:00 a 21:00 (hora de Madrid), sin festivos
  nacionales. Se pueden añadir festivos autonómicos y locales en un archivo.
- Máximo de 3 intentos por contacto en una ventana móvil de 7 días.
- Todas las funciones de este módulo son puras (no tocan la base de datos) para
  poder probarlas fácilmente. El registro de auditoría vive en `registrar_auditoria`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol
from zoneinfo import ZoneInfo

from app.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Teléfonos
# ---------------------------------------------------------------------------


def normalizar_telefono(numero: str) -> str:
    """Normaliza un teléfono a formato E.164.

    Los números españoles de 9 cifras (6, 7, 8 o 9 inicial) reciben el prefijo +34.

    >>> normalizar_telefono("600 11 22 33")
    '+34600112233'
    >>> normalizar_telefono("0034 954-123-456")
    '+34954123456'
    """
    limpio = re.sub(r"[^\d+]", "", numero or "")
    if limpio.startswith("00"):
        limpio = "+" + limpio[2:]
    if limpio.startswith("+"):
        return "+" + re.sub(r"\D", "", limpio[1:])
    if len(limpio) == 9 and limpio[0] in "6789":
        return "+34" + limpio
    if len(limpio) == 11 and limpio.startswith("34"):
        return "+" + limpio
    return limpio


def telefono_valido(numero: str) -> bool:
    """Comprueba que el número normalizado tiene un formato plausible."""
    return re.fullmatch(r"\+\d{8,15}", normalizar_telefono(numero)) is not None


def cargar_lista_telefonos(ruta: Path) -> set[str]:
    """Carga un archivo con un teléfono por línea (admite comentarios con #).

    Se usa para la exportación local de la lista Robinson. Si el archivo no
    existe se devuelve un conjunto vacío.
    """
    if not ruta.is_file():
        return set()
    numeros: set[str] = set()
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()
        if not linea:
            continue
        # Admite CSV: se queda con la primera columna.
        linea = re.split(r"[;,\t]", linea)[0]
        numeros.add(normalizar_telefono(linea))
    return numeros


# ---------------------------------------------------------------------------
# Festivos y horario
# ---------------------------------------------------------------------------


def domingo_de_pascua(anio: int) -> date:
    """Calcula el Domingo de Pascua (algoritmo anónimo gregoriano)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(anio, mes, dia)


def festivos_nacionales(anio: int) -> set[date]:
    """Festivos nacionales comunes a toda España.

    Incluye Viernes Santo, que es festivo en todas las comunidades. No incluye
    los festivos que cada comunidad elige (por ejemplo, Jueves Santo o el 28 de
    febrero en Andalucía): añádelos en el archivo de festivos extra.
    """
    return {
        date(anio, 1, 1),  # Año Nuevo
        date(anio, 1, 6),  # Epifanía del Señor
        domingo_de_pascua(anio) - timedelta(days=2),  # Viernes Santo
        date(anio, 5, 1),  # Fiesta del Trabajo
        date(anio, 8, 15),  # Asunción de la Virgen
        date(anio, 10, 12),  # Fiesta Nacional de España
        date(anio, 11, 1),  # Todos los Santos
        date(anio, 12, 6),  # Día de la Constitución
        date(anio, 12, 8),  # Inmaculada Concepción
        date(anio, 12, 25),  # Navidad
    }


def cargar_festivos_extra(ruta: Path) -> set[date]:
    """Carga festivos adicionales en formato AAAA-MM-DD, uno por línea."""
    if not ruta.is_file():
        return set()
    fechas: set[date] = set()
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()
        if linea:
            fechas.add(date.fromisoformat(linea))
    return fechas


# ---------------------------------------------------------------------------
# Decisiones
# ---------------------------------------------------------------------------


class Motivo(str, Enum):
    """Motivo de una decisión de cumplimiento."""

    PERMITIDO = "permitido"
    NO_LLAMAR = "no_llamar"  # el contacto pidió no ser llamado
    LISTA_INTERNA = "lista_interna"
    ROBINSON = "lista_robinson"
    SIN_CONSENTIMIENTO = "sin_consentimiento"
    REQUIERE_HUMANO = "requiere_humano"  # interés legítimo: solo puede llamar una persona
    TELEFONO_INVALIDO = "telefono_invalido"
    FUERA_DE_HORARIO = "fuera_de_horario"
    MAX_INTENTOS = "max_intentos"


# Motivos que no cambian con el tiempo: el contacto queda excluido de la campaña.
MOTIVOS_PERMANENTES = {
    Motivo.NO_LLAMAR,
    Motivo.LISTA_INTERNA,
    Motivo.ROBINSON,
    Motivo.SIN_CONSENTIMIENTO,
    Motivo.TELEFONO_INVALIDO,
}


@dataclass(frozen=True)
class Decision:
    """Resultado de comprobar si se puede llamar a un contacto."""

    permitido: bool
    motivo: Motivo
    detalle: str = ""
    # Si el bloqueo es temporal, cuándo se podría volver a intentar.
    reintentar_desde: Optional[datetime] = None

    @property
    def permanente(self) -> bool:
        return self.motivo in MOTIVOS_PERMANENTES


class ContactoLike(Protocol):
    """Lo mínimo que necesita el módulo de un contacto (modelo o dataclass)."""

    telefono: str
    consentimiento: bool
    base_legal: str
    no_llamar: bool


@dataclass
class ReglasCumplimiento:
    """Reglas de cumplimiento con las listas y festivos cargados."""

    zona: ZoneInfo
    hora_inicio: int = 9
    hora_fin: int = 21
    max_intentos: int = 3
    ventana_dias: int = 7
    robinson: set[str] = field(default_factory=set)
    lista_interna: set[str] = field(default_factory=set)
    festivos_extra: set[date] = field(default_factory=set)

    @classmethod
    def desde_settings(
        cls, settings: Optional[Settings] = None, lista_interna: Iterable[str] = ()
    ) -> "ReglasCumplimiento":
        s = settings or get_settings()
        return cls(
            zona=ZoneInfo(s.zona_horaria),
            hora_inicio=s.hora_inicio,
            hora_fin=s.hora_fin,
            max_intentos=s.max_intentos,
            ventana_dias=s.ventana_intentos_dias,
            robinson=cargar_lista_telefonos(s.robinson_file),
            lista_interna={normalizar_telefono(t) for t in lista_interna},
            festivos_extra=cargar_festivos_extra(s.festivos_extra_file),
        )

    # --- Horario -----------------------------------------------------------

    def es_festivo(self, dia: date) -> bool:
        return dia in festivos_nacionales(dia.year) or dia in self.festivos_extra

    def es_dia_habil(self, dia: date) -> bool:
        return dia.weekday() < 5 and not self.es_festivo(dia)

    def en_horario(self, momento: datetime) -> bool:
        """True si `momento` cae en horario comercial permitido (hora de Madrid)."""
        local = _a_zona(momento, self.zona)
        if not self.es_dia_habil(local.date()):
            return False
        return time(self.hora_inicio) <= local.time() < time(self.hora_fin)

    def siguiente_ventana(self, momento: datetime) -> datetime:
        """Primer instante permitido a partir de `momento` (devuelto en UTC)."""
        local = _a_zona(momento, self.zona)
        if self.en_horario(local):
            return local.astimezone(timezone.utc)
        dia = local.date()
        if local.time() >= time(self.hora_inicio) or not self.es_dia_habil(dia):
            dia += timedelta(days=1)
        # Como mucho hay unos pocos días seguidos no hábiles; 30 es un límite seguro.
        for _ in range(30):
            if self.es_dia_habil(dia):
                inicio = datetime.combine(dia, time(self.hora_inicio), tzinfo=self.zona)
                return inicio.astimezone(timezone.utc)
            dia += timedelta(days=1)
        raise RuntimeError("No se encontró un día hábil en los próximos 30 días")

    # --- Intentos ----------------------------------------------------------

    def intentos_en_ventana(self, intentos: Iterable[datetime], momento: datetime) -> list[datetime]:
        limite = _utc(momento) - timedelta(days=self.ventana_dias)
        return sorted(_utc(i) for i in intentos if _utc(i) > limite)

    # --- Comprobación completa --------------------------------------------

    def comprobar(
        self,
        contacto: ContactoLike,
        intentos_previos: Iterable[datetime],
        momento: Optional[datetime] = None,
    ) -> Decision:
        """Decide si se puede llamar ahora a `contacto`.

        El orden importa: primero las exclusiones permanentes (petición del
        cliente, listas, consentimiento) y después las temporales (horario e
        intentos), para que el motivo registrado sea el más relevante.
        """
        momento = _utc(momento or datetime.now(timezone.utc))
        telefono = normalizar_telefono(contacto.telefono)

        if not telefono_valido(telefono):
            return Decision(False, Motivo.TELEFONO_INVALIDO, telefono)
        if contacto.no_llamar:
            return Decision(False, Motivo.NO_LLAMAR, "El contacto pidió no ser llamado")
        if telefono in self.lista_interna:
            return Decision(False, Motivo.LISTA_INTERNA, "Número en la lista interna de no llamar")
        if telefono in self.robinson:
            return Decision(False, Motivo.ROBINSON, "Número inscrito en la lista Robinson")
        if not contacto.consentimiento:
            if (contacto.base_legal or "").strip().lower() == "interes_legitimo":
                return Decision(
                    False,
                    Motivo.REQUIERE_HUMANO,
                    "Interés legítimo sin consentimiento: la IA no puede llamar "
                    "(art. 66.1.a Ley 11/2022). Derivar a un comercial humano.",
                )
            return Decision(False, Motivo.SIN_CONSENTIMIENTO, "Sin consentimiento registrado")

        if not self.en_horario(momento):
            return Decision(
                False,
                Motivo.FUERA_DE_HORARIO,
                "Fuera del horario comercial permitido",
                reintentar_desde=self.siguiente_ventana(momento),
            )

        recientes = self.intentos_en_ventana(intentos_previos, momento)
        if len(recientes) >= self.max_intentos:
            # Se libera un hueco cuando el intento más antiguo sale de la ventana.
            libre = recientes[-self.max_intentos] + timedelta(days=self.ventana_dias)
            return Decision(
                False,
                Motivo.MAX_INTENTOS,
                f"{len(recientes)} intentos en {self.ventana_dias} días",
                reintentar_desde=self.siguiente_ventana(libre + timedelta(seconds=1)),
            )

        return Decision(True, Motivo.PERMITIDO)


# ---------------------------------------------------------------------------
# Grabación y avisos
# ---------------------------------------------------------------------------


def _sin_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )


_PATRONES_OPOSICION_GRABACION = [
    r"\bno (me )?grab",
    r"\bno (quiero|deseo|acepto|autorizo|doy permiso)\b.{0,25}\bgrab",
    r"\bsin grabar",
    r"\bme opongo\b.{0,25}\bgrab",
    r"\bque no se grabe",
    r"\bno se (puede|debe) grabar",
]


def se_opone_a_grabacion(texto: str) -> bool:
    """Detecta si el cliente se opone a que se grabe la llamada."""
    t = _sin_acentos(texto)
    return any(re.search(p, t) for p in _PATRONES_OPOSICION_GRABACION)


def aviso_inicial(nombre_agente: str, empresa: str, ciudad: str, nombre_cliente: str = "") -> str:
    """Texto del saludo con todos los avisos obligatorios.

    Incluye: identidad de la empresa, que es una IA (AI Act art. 50), la
    finalidad comercial y el derecho de oposición (Circular 1/2023, art. 6) y el
    aviso de grabación.
    """
    saludo = f"Hola, {nombre_cliente.split()[0]}. " if nombre_cliente.strip() else "Hola. "
    return (
        f"{saludo}Soy {nombre_agente}, un asistente de inteligencia artificial de {empresa}, "
        f"de {ciudad}, y le llamo para ofrecerle servicios de domótica. "
        "La llamada puede grabarse; si no quiere, o no quiere más llamadas, dígamelo. "
        "¿Tiene un minuto?"
    )


# ---------------------------------------------------------------------------
# Auditoría
# ---------------------------------------------------------------------------


def registrar_auditoria(
    sesion: Any,
    evento: str,
    *,
    contacto_id: Optional[int] = None,
    llamada_id: Optional[int] = None,
    telefono: str = "",
    **detalle: Any,
) -> None:
    """Inserta una fila en la tabla de auditoría (no hace commit)."""
    from app.db.models import Auditoria  # import local para mantener el módulo puro

    sesion.add(
        Auditoria(
            evento=evento,
            contacto_id=contacto_id,
            llamada_id=llamada_id,
            telefono=normalizar_telefono(telefono) if telefono else "",
            detalle={k: _serializable(v) for k, v in detalle.items()},
        )
    )


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _utc(momento: datetime) -> datetime:
    if momento.tzinfo is None:
        return momento.replace(tzinfo=timezone.utc)
    return momento.astimezone(timezone.utc)


def _a_zona(momento: datetime, zona: ZoneInfo) -> datetime:
    return _utc(momento).astimezone(zona)


def _serializable(valor: Any) -> Any:
    if isinstance(valor, (datetime, date)):
        return valor.isoformat()
    if isinstance(valor, Enum):
        return valor.value
    return valor
