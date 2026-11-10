"""Prueba de extremo a extremo con un Asterisk real (sin proveedor SIP).

No se ejecuta por defecto. Requisitos:
- Asterisk en marcha con la configuración de `asterisk/conf/` (ARI en 127.0.0.1:8088
  y el contexto `[prueba]` de `extensions.conf`).
- Modelos de Whisper y Piper descargados (o conexión para descargarlos).
- Permiso de escritura en la carpeta de sonidos de Asterisk.

    BANPI_E2E_ASTERISK=1 python -m pytest tests/test_e2e_asterisk.py -s

Un canal Local hace de cliente: contesta, espera al saludo y dice "No me interesa,
gracias". Se comprueba que el audio llega por External Media, que Whisper lo
entiende y que el agente se despide y cuelga.
"""

from __future__ import annotations

import asyncio
import os
import wave
from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BANPI_E2E_ASTERISK") != "1", reason="Prueba con Asterisk real: BANPI_E2E_ASTERISK=1"
)

SONIDOS = Path(os.environ.get("ASTERISK_SONIDOS", "/usr/share/asterisk/sounds"))


async def _llamada() -> None:
    from app.agent.dialog import Dialogo, Resultado
    from app.agent.llm import PlantillasLLM
    from app.agent.pipeline import SesionLlamada
    from app.audio.stt import WhisperSTT
    from app.audio.tts import PiperTTS
    from app.config import get_settings
    from app.telephony.asterisk_ari import AsteriskARI
    from app.telephony.base import EstadoMarcacion

    s = replace(get_settings(), sip_endpoint="Local/100@prueba", vad_backend="webrtc")
    tts = PiperTTS(8000, s)
    audio = await tts.sintetizar("No me interesa, gracias.")
    (SONIDOS / "custom").mkdir(parents=True, exist_ok=True)
    with wave.open(str(SONIDOS / "custom" / "cliente.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(audio)

    ari = AsteriskARI(s)
    await ari.iniciar()
    stt = WhisperSTT(s)
    await stt.calentar()
    try:
        marcacion = await ari.originar("+34600000000", timeout_s=10)
        assert marcacion.estado is EstadoMarcacion.CONTESTADA and marcacion.llamada
        llamada = marcacion.llamada
        sesion = SesionLlamada(llamada, Dialogo("María", settings=s), PlantillasLLM(), stt, tts, s)
        resultado = await sesion.ejecutar()
        assert llamada.rtp.paquetes_recibidos > 50 and llamada.rtp.paquetes_enviados > 50
        assert resultado is Resultado.NO_INTERESA
        print("Latencias (ms):", sesion.metricas.latencias_ms)
    finally:
        await ari.detener()


def test_llamada_real_con_asterisk() -> None:
    asyncio.run(_llamada())
