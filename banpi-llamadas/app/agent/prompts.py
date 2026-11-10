"""Prompt de sistema de Álex, instrucciones por estado y frases de respaldo.

El LLM solo redacta. Las decisiones (cambiar de estado, colgar, marcar como no
contactar, guardar la cita) las toma la máquina de estados en `dialog.py`. Cada
instrucción tiene una frase de respaldo que se usa si Ollama no responde a tiempo
o si se ejecuta sin modelo (`LLM_BACKEND=plantillas`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings


def prompt_sistema(s: Settings) -> str:
    """Prompt de sistema con la personalidad y las reglas de Álex."""
    contacto = ""
    if s.telefono_empresa or s.web_empresa:
        contacto = (
            "Datos de contacto que puedes dar si los piden: "
            + ", ".join(x for x in (s.telefono_empresa, s.web_empresa) if x)
            + ".\n"
        )
    return f"""Eres {s.nombre_agente}, un asistente virtual con inteligencia artificial que hace llamadas telefónicas para {s.nombre_empresa}, una empresa de domótica de {s.ciudad_empresa} (España).

Servicios de {s.nombre_empresa}: {s.servicios}. Zona de trabajo: {s.zona_servicio}.
{contacto}
Cómo hablas:
- Español de España, tratando de usted, con tono cercano, profesional y natural, como una persona amable al teléfono.
- Frases cortas. Responde con UNA o, como mucho, DOS frases. Nunca sueltes un discurso.
- Nada de listas, viñetas, emojis, asteriscos ni formato: todo lo que escribes se convierte en voz.
- Escribe los números y las horas como se dicen ("a las cinco de la tarde").
- No repitas el saludo ni te vuelvas a presentar.

Reglas que nunca rompes:
- Si te preguntan, reconoces siempre que eres una inteligencia artificial.
- No inventas precios, plazos, descuentos, marcas ni promesas. Si no sabes algo, dices que lo consultará un comercial.
- Si el cliente no está interesado, lo aceptas con educación, sin insistir.
- No pides datos bancarios ni datos sensibles.
- Sigue siempre la INSTRUCCIÓN DEL MOMENTO que recibirás: es lo que debes conseguir en esta respuesta."""


@dataclass(frozen=True)
class Instruccion:
    """Lo que el LLM debe conseguir en un turno, con su frase de respaldo."""

    objetivo: str
    respaldo: str


# Frases de presentación de cada servicio para la propuesta de respaldo.
FRASES_INTERES = {
    "persianas": "las persianas suben y bajan solas según la hora o el sol",
    "luces": "las luces se encienden y se apagan solas o desde el móvil",
    "climatizacion": "la climatización se ajusta sola y no se queda encendida sin nadie en casa",
    "alarma": "la alarma le avisa al móvil al momento",
    "camaras": "ve las cámaras desde el móvil esté donde esté",
}


def instruccion_descubrimiento(paso: str, datos_resumen: str) -> Instruccion:
    """Instrucción para la fase de descubrimiento según lo que falte por saber."""
    if paso == "inmueble":
        return Instruccion(
            "El cliente acepta hablar. Agradéceselo en pocas palabras y pregúntale si lo que tiene "
            "es una vivienda o un local o negocio.",
            "Muchas gracias. Para no hacerle perder el tiempo, ¿lo suyo es una vivienda o un local?",
        )
    if paso == "intereses":
        return Instruccion(
            f"Lo que sabes del cliente: {datos_resumen}. Pregúntale si ha pensado en automatizar "
            "algo de su casa o local: persianas, luces, climatización, alarma o cámaras.",
            "¿Y ha pensado alguna vez en automatizar algo, como las persianas, las luces, "
            "la climatización o la alarma?",
        )
    return Instruccion(
        f"Lo que sabes del cliente: {datos_resumen}. Pregúntale qué es lo que más le importa o le "
        "preocupa: ahorrar en la factura, la seguridad o la comodidad.",
        "¿Y qué es lo que más le importa: ahorrar en la factura, la seguridad o la comodidad?",
    )


def instruccion_propuesta(datos_resumen: str, intereses: list[str], empresa: str) -> Instruccion:
    """Propuesta breve adaptada a lo que ha contado el cliente."""
    frases = [FRASES_INTERES[i] for i in intereses if i in FRASES_INTERES][:2]
    if frases:
        beneficio = " y ".join(frases)
        respaldo = (
            f"Eso lo hacemos mucho: con la domótica {beneficio}, todo desde el móvil. "
            "¿Le vendría bien una visita gratuita y sin compromiso, o prefiere que le llame un comercial?"
        )
    else:
        respaldo = (
            f"En {empresa} le ayudamos a controlar persianas, luces, clima y seguridad desde el móvil. "
            "¿Le vendría bien una visita gratuita y sin compromiso, o prefiere que le llame un comercial?"
        )
    return Instruccion(
        f"Lo que sabes del cliente: {datos_resumen}. En una frase, explica cómo le ayudaría la domótica "
        "en lo que ha dicho, sin precios ni promesas. En la segunda frase, propón una visita gratuita "
        "y sin compromiso o una llamada con un comercial, y pregúntale qué prefiere.",
        respaldo,
    )


def instruccion_respuesta_libre(datos_resumen: str) -> Instruccion:
    """El cliente pregunta o comenta algo durante el cierre."""
    return Instruccion(
        f"Lo que sabes del cliente: {datos_resumen}. Responde con brevedad a lo que acaba de decir; "
        "si no lo sabes con seguridad, di que se lo concretará un comercial. Después vuelve a proponer "
        "la visita gratuita o la llamada con un comercial.",
        "Eso se lo concreta mejor un comercial, que lo verá con usted sin compromiso. "
        "¿Le parece que le agendemos una visita gratuita?",
    )


def instruccion_aclarar_momento() -> Instruccion:
    """El cliente responde algo ambiguo al preguntarle si tiene un minuto."""
    return Instruccion(
        "El cliente ha respondido algo que no es un sí ni un no claro a si tiene un minuto. "
        "Explica en una frase que llamas para contarle cómo automatizar su casa o local, y pregúntale "
        "si lo que tiene es una vivienda o un local.",
        "Le llamo porque ayudamos a automatizar casas y locales, como persianas, luces o alarma. "
        "¿Lo suyo es una vivienda o un local?",
    )


# ---------------------------------------------------------------------------
# Limpieza de la salida del LLM
# ---------------------------------------------------------------------------

_PATRON_PRECIO = re.compile(
    r"\d[\d.,]*\s*(€|euros?|eur\b)|(€|euros?)\s*\d|\bpor solo\b|\bdescuento del?\b|\bgratis para siempre\b",
    re.IGNORECASE,
)
_FRASE_SIN_PRECIO = "El precio exacto se lo concreta un comercial, porque depende de cada caso."


def contiene_precio(texto: str) -> bool:
    return bool(_PATRON_PRECIO.search(texto))


def limpiar_frase(frase: str) -> str:
    """Quita formato, prefijos de rol y emojis, y sustituye frases con precios."""
    f = frase.strip()
    f = re.sub(r"^(álex|alex|asistente|agente)\s*:\s*", "", f, flags=re.IGNORECASE)
    f = re.sub(r"[*_#`>\[\]]", "", f)
    f = re.sub(r"^\s*[-•]\s*", "", f)
    # Elimina emojis y símbolos pictográficos.
    f = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", f)
    f = re.sub(r"\s+", " ", f).strip()
    if contiene_precio(f):
        return _FRASE_SIN_PRECIO
    return f


_FIN_FRASE = re.compile(r"(?<=[.!?…])\s+(?=[¿¡A-ZÁÉÍÓÚÑ0-9])")


def dividir_frases(texto: str) -> list[str]:
    """Divide un texto en frases."""
    return [p.strip() for p in _FIN_FRASE.split(texto.strip()) if p.strip()]


def recortar(texto: str, max_frases: int = 2) -> str:
    """Limpia y deja como mucho `max_frases` frases."""
    frases = [limpiar_frase(f) for f in dividir_frases(texto)]
    return " ".join(f for f in frases[:max_frases] if f)
