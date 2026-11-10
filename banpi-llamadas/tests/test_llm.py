"""Pruebas del streaming por frases y de la recuperación ante fallos del LLM."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import httpx
import pytest

from app.agent.llm import (
    Conversacion,
    OllamaLLM,
    PlantillasLLM,
    frases_en_streaming,
    respuesta_en_frases,
)
from app.config import Settings


async def _trozos(*partes: str) -> AsyncIterator[str]:
    for p in partes:
        yield p


async def _lista(gen: AsyncIterator[str]) -> list[str]:
    return [x async for x in gen]


def test_frases_en_streaming_corta_y_limita() -> None:
    gen = frases_en_streaming(_trozos("Claro", ", le cuento. Es muy", " fácil. Tercera", " frase."))
    assert asyncio.run(_lista(gen)) == ["Claro, le cuento.", "Es muy fácil."]


def test_frases_en_streaming_ultimo_trozo_sin_espacio() -> None:
    gen = frases_en_streaming(_trozos("¿Le vendría bien", " el martes?"))
    assert asyncio.run(_lista(gen)) == ["¿Le vendría bien el martes?"]


def test_frase_con_precio_se_sustituye() -> None:
    gen = frases_en_streaming(_trozos("Le saldría por 300 euros. ", "¿Le apunto?"))
    frases = asyncio.run(_lista(gen))
    assert "300" not in frases[0] and frases[1] == "¿Le apunto?"


class LLMLento:
    async def generar(self, mensajes, respaldo=""):  # type: ignore[no-untyped-def]
        await asyncio.sleep(10)
        yield "nunca"

    async def calentar(self) -> None:
        return None


class LLMRoto:
    async def generar(self, mensajes, respaldo=""):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("sin conexión")
        yield ""  # pragma: no cover

    async def calentar(self) -> None:
        return None


@pytest.mark.parametrize("llm", [LLMLento(), LLMRoto()])
def test_respaldo_si_falla(llm) -> None:  # type: ignore[no-untyped-def]
    gen = respuesta_en_frases(llm, [], "Frase de respaldo. Segunda.", timeout_primera_s=0.05)
    assert asyncio.run(_lista(gen)) == ["Frase de respaldo.", "Segunda."]


def test_plantillas() -> None:
    gen = respuesta_en_frases(PlantillasLLM(), [], "Hola, ¿qué tal? Muy bien.")
    assert asyncio.run(_lista(gen)) == ["Hola, ¿qué tal?", "Muy bien."]


def test_conversacion_mensajes() -> None:
    c = Conversacion("SISTEMA")
    c.agente("Hola.")
    c.agente("¿Tiene un minuto?")
    c.cliente("Sí")
    m = c.mensajes("Pregunta X", texto_previo="Gracias.")
    assert m[0] == {"role": "system", "content": "SISTEMA"}
    assert m[1] == {"role": "assistant", "content": "Hola. ¿Tiene un minuto?"}
    assert m[2]["role"] == "user"
    assert "Pregunta X" in m[-1]["content"] and "Gracias." in m[-1]["content"]


def test_ollama_streaming_con_servidor_falso() -> None:
    """Comprueba el formato de la petición y la lectura de líneas NDJSON de Ollama."""
    recibido: dict = {}

    def manejador(req: httpx.Request) -> httpx.Response:
        recibido.update(json.loads(req.content))
        lineas = [
            {"message": {"content": "Claro, "}, "done": False},
            {"message": {"content": "le cuento."}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lineas))

    s = Settings(ollama_model="qwen2.5:7b")
    llm = OllamaLLM(s)
    llm._cliente = httpx.AsyncClient(transport=httpx.MockTransport(manejador), base_url="http://x")
    frases = asyncio.run(_lista(respuesta_en_frases(llm, [{"role": "user", "content": "hola"}], "R.")))
    assert frases == ["Claro, le cuento."]
    assert recibido["model"] == "qwen2.5:7b" and recibido["stream"] is True
