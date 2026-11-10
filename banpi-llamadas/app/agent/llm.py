"""Cliente de Ollama con historial de conversación y respuesta en streaming.

Para cumplir el objetivo de latencia (menos de 1,5 s), la respuesta se genera en
streaming y se corta por frases: cada frase completa se manda a síntesis de voz
en cuanto está lista, sin esperar a que el modelo termine.

Hay dos implementaciones:
- `OllamaLLM`: modelo local servido por Ollama (`/api/chat`).
- `PlantillasLLM`: no usa ningún modelo, devuelve la frase de respaldo del estado.
  Sirve para pruebas y para la simulación sin Ollama.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Protocol

import httpx

from app.agent import prompts
from app.config import Settings, get_settings

log = logging.getLogger(__name__)

Mensaje = dict[str, str]


class LLM(Protocol):
    """Interfaz de un modelo de lenguaje que genera texto en streaming."""

    def generar(self, mensajes: list[Mensaje], respaldo: str = "") -> AsyncIterator[str]:
        """Devuelve los fragmentos de texto según se generan."""
        ...

    async def calentar(self) -> None:
        """Carga el modelo en memoria para que la primera respuesta sea rápida."""
        ...


class OllamaLLM:
    """Cliente de la API de chat de Ollama."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.s = settings or get_settings()
        self._cliente = httpx.AsyncClient(
            base_url=self.s.ollama_url, timeout=httpx.Timeout(self.s.ollama_timeout_s, connect=5.0)
        )

    async def generar(self, mensajes: list[Mensaje], respaldo: str = "") -> AsyncIterator[str]:
        cuerpo = {
            "model": self.s.ollama_model,
            "messages": mensajes,
            "stream": True,
            "keep_alive": self.s.ollama_keep_alive,
            "options": {
                "temperature": self.s.ollama_temperatura,
                # Dos frases cortas caben de sobra en 120 tokens.
                "num_predict": 120,
            },
        }
        async with self._cliente.stream("POST", "/api/chat", json=cuerpo) as resp:
            resp.raise_for_status()
            async for linea in resp.aiter_lines():
                if not linea:
                    continue
                datos = json.loads(linea)
                if "error" in datos:
                    raise RuntimeError(f"Ollama: {datos['error']}")
                trozo = datos.get("message", {}).get("content", "")
                if trozo:
                    yield trozo
                if datos.get("done"):
                    break

    async def calentar(self) -> None:
        try:
            await self._cliente.post(
                "/api/generate",
                json={"model": self.s.ollama_model, "keep_alive": self.s.ollama_keep_alive},
            )
        except httpx.HTTPError as e:  # pragma: no cover - depende del entorno
            log.warning("No se pudo precargar el modelo de Ollama: %s", e)

    async def cerrar(self) -> None:
        await self._cliente.aclose()


class PlantillasLLM:
    """'Modelo' sin IA que devuelve la frase de respaldo, troceada como si fuera streaming."""

    def __init__(self, retardo_s: float = 0.0) -> None:
        self.retardo_s = retardo_s

    async def generar(self, mensajes: list[Mensaje], respaldo: str = "") -> AsyncIterator[str]:
        for palabra in re.findall(r"\S+\s*", respaldo):
            if self.retardo_s:
                await asyncio.sleep(self.retardo_s)
            yield palabra

    async def calentar(self) -> None:
        return None


def crear_llm(settings: Optional[Settings] = None) -> LLM:
    """Crea el LLM indicado por `LLM_BACKEND`."""
    s = settings or get_settings()
    if s.llm_backend == "plantillas":
        return PlantillasLLM()
    return OllamaLLM(s)


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------


@dataclass
class Conversacion:
    """Historial de la llamada en el formato de mensajes de Ollama."""

    sistema: str
    historial: list[Mensaje] = field(default_factory=list)
    max_mensajes: int = 16

    def cliente(self, texto: str) -> None:
        self.historial.append({"role": "user", "content": texto})

    def agente(self, texto: str) -> None:
        if not texto.strip():
            return
        if self.historial and self.historial[-1]["role"] == "assistant":
            self.historial[-1]["content"] += " " + texto
        else:
            self.historial.append({"role": "assistant", "content": texto})

    def mensajes(self, instruccion: str, texto_previo: str = "") -> list[Mensaje]:
        """Mensajes para el LLM: sistema, historial reciente e instrucción del momento."""
        extra = ""
        if texto_previo:
            extra = (
                f' Antes de tu respuesta ya has dicho: "{texto_previo}". Continúa sin repetirlo.'
            )
        return [
            {"role": "system", "content": self.sistema},
            *self.historial[-self.max_mensajes :],
            {
                "role": "system",
                "content": f"INSTRUCCIÓN DEL MOMENTO: {instruccion}{extra} "
                "Responde solo con lo que vas a decir en voz alta, en una o dos frases cortas.",
            },
        ]


# ---------------------------------------------------------------------------
# Streaming por frases
# ---------------------------------------------------------------------------

_FIN = re.compile(r"[.!?…]+[\"')»]*\s")


async def frases_en_streaming(trozos: AsyncIterator[str], max_frases: int = 2) -> AsyncIterator[str]:
    """Agrupa los trozos del LLM en frases completas y las va devolviendo.

    Se detiene al llegar a `max_frases` (regla de máximo dos frases).
    """
    buffer = ""
    emitidas = 0
    async for trozo in trozos:
        buffer += trozo
        while True:
            m = _FIN.search(buffer)
            # Evita cortar en abreviaturas muy cortas como "Sr." o números "1.".
            if not m or len(buffer[: m.end()].strip()) < 4:
                break
            frase, buffer = buffer[: m.end()], buffer[m.end() :]
            frase = prompts.limpiar_frase(frase)
            if frase:
                yield frase
                emitidas += 1
                if emitidas >= max_frases:
                    return
    resto = prompts.limpiar_frase(buffer)
    if resto and emitidas < max_frases:
        yield resto


async def respuesta_en_frases(
    llm: LLM,
    mensajes: list[Mensaje],
    respaldo: str,
    timeout_primera_s: float = 4.0,
    max_frases: int = 2,
) -> AsyncIterator[str]:
    """Genera la respuesta frase a frase; si el LLM falla o tarda, usa el respaldo.

    Si el fallo ocurre después de haber dicho alguna frase, simplemente se termina
    (no se mezcla el respaldo con una respuesta a medias).
    """
    generador = frases_en_streaming(llm.generar(mensajes, respaldo), max_frases)
    emitidas = 0
    try:
        primera = await asyncio.wait_for(generador.__anext__(), timeout=timeout_primera_s)
        emitidas += 1
        yield primera
        async for frase in generador:
            emitidas += 1
            yield frase
    except StopAsyncIteration:
        pass
    except (asyncio.TimeoutError, httpx.HTTPError, RuntimeError, json.JSONDecodeError) as e:
        log.warning("LLM sin respuesta válida (%s); se usa la frase de respaldo", e)
    finally:
        await generador.aclose()
    if emitidas == 0 and respaldo:
        for frase in prompts.dividir_frases(prompts.recortar(respaldo, max_frases)):
            yield frase
