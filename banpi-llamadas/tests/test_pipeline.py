"""Pruebas del pipeline completo con una llamada simulada (sin modelos)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from app.agent.dialog import Accion, Dialogo, Resultado
from app.agent.llm import PlantillasLLM
from app.agent.pipeline import SesionLlamada
from app.audio.tts import TonoTTS
from app.config import Settings
from app.telephony.simulada import FraseCliente, LlamadaGuionada

AHORA = datetime(2026, 11, 10, 11, 0, tzinfo=ZoneInfo("Europe/Madrid"))
SR = 8000


def voz(ms: int = 500) -> bytes:
    """'Voz' sintética: un tono que el VAD de energía detecta como habla."""
    t = np.arange(int(SR * ms / 1000)) / SR
    return (6000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16).tobytes()


class STTGuionado:
    """Transcriptor falso que devuelve los textos del guion en orden."""

    def __init__(self, textos: list[str]) -> None:
        self.textos = list(textos)

    async def transcribir(self, audio: bytes, sample_rate: int) -> str:
        return self.textos.pop(0) if self.textos else ""

    async def calentar(self) -> None:
        return None


class Registro:
    def __init__(self) -> None:
        self.agente: list[tuple[str, str, Optional[int]]] = []
        self.cliente: list[str] = []
        self.acciones: list[Accion] = []

    def al_hablar(self, texto: str, estado: str, latencia_ms: Optional[int]) -> None:
        self.agente.append((texto, estado, latencia_ms))

    def al_escuchar(self, texto: str, estado: str) -> None:
        self.cliente.append(texto)

    def al_accion(self, accion: Accion, dialogo: Dialogo) -> None:
        self.acciones.append(accion)


def _settings() -> Settings:
    return Settings(vad_backend="energia", vad_silencio_fin_ms=300, barge_in_ms=200, max_duracion_s=60)


def ejecutar(textos: list[str], guion: list[FraseCliente]) -> tuple[Optional[Resultado], Registro, LlamadaGuionada, SesionLlamada]:
    s = _settings()
    llamada = LlamadaGuionada(guion, sample_rate=SR, velocidad=8.0)
    registro = Registro()
    sesion = SesionLlamada(
        llamada,
        Dialogo("María López", settings=s, reloj=lambda: AHORA),
        PlantillasLLM(),
        STTGuionado(textos),
        TonoTTS(SR, ms_por_caracter=5),
        settings=s,
        oyente=registro,
        silencio_max_s=5,
    )
    resultado = asyncio.run(sesion.ejecutar())
    return resultado, registro, llamada, sesion


def test_conversacion_completa_hasta_cita() -> None:
    textos = ["Sí, dígame", "Tengo un piso", "Las persianas y la alarma", "Una visita", "El jueves a las cinco"]
    resultado, reg, llamada, sesion = ejecutar(textos, [FraseCliente(voz()) for _ in textos])
    assert resultado is Resultado.CITA
    assert reg.cliente == textos
    assert Accion.GUARDAR_CITA in reg.acciones
    assert "inteligencia artificial" in reg.agente[0][0]
    assert "jueves 12" in reg.agente[-1][0]
    assert llamada.colgada.is_set()
    # Se midió la latencia de cada respuesta al cliente.
    assert len(sesion.metricas.latencias_ms) == len(textos)


def test_no_llamar_se_registra_y_cuelga() -> None:
    resultado, reg, llamada, _ = ejecutar(["No me llaméis más"], [FraseCliente(voz())])
    assert resultado is Resultado.NO_LLAMAR
    assert reg.acciones == [Accion.MARCAR_NO_LLAMAR]
    assert llamada.colgada.is_set()


def test_interrupcion_corta_el_audio() -> None:
    textos = ["Sí", "Tengo un piso", "Las persianas", "No me interesa"]
    guion = [FraseCliente(voz()) for _ in range(3)]
    # El cliente interrumpe 300 ms después de que el agente empiece a hablar.
    guion.append(FraseCliente(voz(600), interrumpir_tras_ms=300))
    resultado, reg, llamada, sesion = ejecutar(textos, guion)
    assert sesion.metricas.interrupciones >= 1
    assert llamada.vaciados >= 1
    assert resultado is Resultado.NO_INTERESA


def test_saludo_no_se_interrumpe() -> None:
    # El cliente habla en mitad del saludo: el saludo se oye entero y luego se responde.
    guion = [FraseCliente(voz(400), interrumpir_tras_ms=200)]
    resultado, reg, llamada, sesion = ejecutar(["No me interesa"], guion)
    assert sesion.metricas.interrupciones == 0
    assert llamada.vaciados == 0
    assert resultado is Resultado.NO_INTERESA
    saludo_audio = asyncio.run(TonoTTS(SR, 5).sintetizar(reg.agente[0][0]))
    assert len(llamada.audio_agente) >= len(saludo_audio)


def test_oposicion_grabacion_detiene_y_borra() -> None:
    textos = ["No quiero que se grabe", "No me interesa"]
    resultado, reg, llamada, _ = ejecutar(textos, [FraseCliente(voz()) for _ in textos])
    assert Accion.DESACTIVAR_GRABACION in reg.acciones
    assert llamada.grabacion_borrada


def test_silencio_prolongado_termina() -> None:
    resultado, reg, llamada, _ = ejecutar([], [])
    assert resultado is Resultado.SIN_RESPUESTA
    assert any("oye bien" in t for t, _, _ in reg.agente)
