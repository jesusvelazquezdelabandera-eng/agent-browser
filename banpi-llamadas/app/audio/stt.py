"""Reconocimiento de voz en local con faster-whisper.

faster-whisper no transcribe en streaming de forma nativa. Para que el turno se
transcriba en cuanto el cliente termina de hablar, el VAD corta el audio del turno
(`FinVoz`) y aquí se transcribe ese fragmento al momento, en un hilo aparte para no
bloquear el bucle de audio. Con el modelo `small` en GPU tarda unos 100 a 300 ms
por turno; en CPU, entre 0,5 y 1,5 s según el procesador.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Optional, Protocol

import numpy as np

from app.audio.resample import remuestrear
from app.config import Settings, get_settings

log = logging.getLogger(__name__)

# Contexto que ayuda a Whisper con el vocabulario de la llamada.
PROMPT_INICIAL = (
    "Llamada de teléfono en español de España sobre domótica: persianas, luces, "
    "climatización, alarma, cámaras. Sevilla, Triana, Nervión, Dos Hermanas."
)

# Alucinaciones típicas de Whisper con silencio o ruido.
_ALUCINACIONES = re.compile(
    r"(subt[ií]tulos|amara\.org|gracias por ver|suscr[ií]bete|www\.|\.com|♪)", re.IGNORECASE
)


class Transcriptor(Protocol):
    async def transcribir(self, audio: bytes, sample_rate: int) -> str: ...

    async def calentar(self) -> None: ...


class WhisperSTT:
    """Transcriptor con faster-whisper. El modelo se carga en el primer uso."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.s = settings or get_settings()
        self._modelo = None
        self._lock = threading.Lock()

    def _cargar(self):  # type: ignore[no-untyped-def]
        if self._modelo is None:
            from faster_whisper import WhisperModel

            log.info("Cargando Whisper '%s' (%s)...", self.s.whisper_model, self.s.whisper_device)
            self.s.whisper_dir.mkdir(parents=True, exist_ok=True)
            self._modelo = WhisperModel(
                self.s.whisper_model,
                device=self.s.whisper_device,
                compute_type=self.s.whisper_compute_type,
                download_root=str(self.s.whisper_dir),
            )
        return self._modelo

    def _transcribir_sync(self, audio16k: bytes) -> str:
        with self._lock:
            modelo = self._cargar()
            muestras = np.frombuffer(audio16k, dtype=np.int16).astype(np.float32) / 32768.0
            segmentos, _info = modelo.transcribe(
                muestras,
                language="es",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,  # ya lo hace nuestro VAD
                condition_on_previous_text=False,
                without_timestamps=True,
                initial_prompt=PROMPT_INICIAL,
            )
            partes = []
            for seg in segmentos:
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                    continue
                partes.append(seg.text.strip())
        texto = " ".join(p for p in partes if p).strip()
        if _ALUCINACIONES.search(texto):
            log.debug("Descartada posible alucinación de Whisper: %r", texto)
            return ""
        return texto

    async def transcribir(self, audio: bytes, sample_rate: int) -> str:
        """Transcribe un turno completo de audio PCM16."""
        audio16k = remuestrear(audio, sample_rate, 16000)
        return await asyncio.to_thread(self._transcribir_sync, audio16k)

    async def calentar(self) -> None:
        """Carga el modelo y hace una transcripción vacía para evitar la espera inicial."""
        await asyncio.to_thread(self._transcribir_sync, b"\x00\x00" * 8000)
