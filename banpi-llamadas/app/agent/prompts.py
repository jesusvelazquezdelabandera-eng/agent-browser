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
- No repitas lo que el cliente ya te ha contado ni le vuelvas a preguntar algo que ya ha contestado.

Reglas que nunca rompes:
- Si te preguntan, reconoces siempre que eres una inteligencia artificial.
- No inventas precios, plazos, descuentos, marcas ni promesas. Si no sabes algo, dices que lo consultará un comercial.
- Nunca digas que algo es compatible con un aparato o una marca (Alexa, Google, una caldera concreta...): eso lo confirma el comercial en la visita.
- Si preguntan por un municipio, contesta según la zona de trabajo indicada arriba; si no está claro, di que lo confirma el comercial.
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
    "persianas": "las persianas suben y bajan solas",
    "luces": "las luces se apagan solas cuando no hay nadie",
    "climatizacion": "el clima se ajusta solo",
    "alarma": "la alarma le avisa al momento",
    "camaras": "ve las cámaras esté donde esté",
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
            f"(Contexto, no lo repitas: {datos_resumen}.) Pregúntale si ha pensado en automatizar algo, "
            "nombrando los ejemplos: persianas, luces, climatización, alarma o cámaras.",
            "¿Y ha pensado alguna vez en automatizar algo, como las persianas, las luces, "
            "la climatización o la alarma?",
        )
    return Instruccion(
        f"(Contexto, no lo repitas: {datos_resumen}.) Pregúntale qué es lo que más le importa: "
        "ahorrar en la factura, la seguridad o la comodidad.",
        "¿Y qué es lo que más le importa: ahorrar en la factura, la seguridad o la comodidad?",
    )


def instruccion_propuesta(datos_resumen: str, intereses: list[str], empresa: str) -> Instruccion:
    """Propuesta breve adaptada a lo que ha contado el cliente."""
    frases = [FRASES_INTERES[i] for i in intereses if i in FRASES_INTERES][:2]
    if frases:
        beneficio = " y ".join(frases)
        respaldo = (
            f"Eso lo hacemos mucho: {beneficio}, y lo controla todo desde el móvil. "
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


def instruccion_respuesta_libre(datos_resumen: str, texto_cliente: str = "", zona: str = "") -> Instruccion:
    """El cliente pregunta o comenta algo durante el cierre."""
    zona_txt = f" Si pregunta por la zona: trabajamos en {zona}." if zona else ""
    return Instruccion(
        f'El cliente acaba de decir: "{texto_cliente}". Responde EXACTAMENTE a eso en UNA sola frase corta, '
        "sin hacer ninguna pregunta y sin repetir lo que ya sabes de él; si es algo que no sabes con seguridad "
        f"(compatibilidades, marcas, plazos, detalles técnicos), di que se lo confirma el comercial.{zona_txt} "
        f"(Contexto, no lo repitas: {datos_resumen}.)",
        "Eso se lo concreta mejor un comercial, que lo verá con usted sin compromiso.",
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
_PATRON_COMPATIBLE = re.compile(r"\b(es|son|sería|serían|será|serán)\s+(totalmente\s+|100 % )?compatibles?\b", re.IGNORECASE)
_FRASE_SIN_COMPATIBILIDAD = "Eso se lo confirma el comercial en la visita, según lo que ya tenga instalado."


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
    if _PATRON_COMPATIBLE.search(f):
        return _FRASE_SIN_COMPATIBILIDAD
    return f


_FIN_FRASE = re.compile(r"(?<=[.!?…])\s+(?=[¿¡A-ZÁÉÍÓÚÑ0-9])")


def dividir_frases(texto: str) -> list[str]:
    """Divide un texto en frases."""
    return [p.strip() for p in _FIN_FRASE.split(texto.strip()) if p.strip()]


def recortar(texto: str, max_frases: int = 2) -> str:
    """Limpia y deja como mucho `max_frases` frases."""
    frases = [limpiar_frase(f) for f in dividir_frases(texto)]
    return " ".join(f for f in frases[:max_frases] if f)
