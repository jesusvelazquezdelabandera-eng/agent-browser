"""Síntesis de voz en local con Piper, con caché de frases frecuentes.

La voz por defecto es `es_ES-davefx-medium` (español de España, 22 050 Hz). La
salida se convierte a la frecuencia de la línea (8 kHz en telefonía).

Caché en dos niveles:
- Memoria (LRU) para cualquier frase sintetizada durante la ejecución.
- Disco para las frases fijas (saludo, despedidas...), que se precalculan al
  arrancar con `precalentar()` y así suenan sin ninguna espera.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np

from app.audio.resample import remuestrear
from app.config import Settings, get_settings

log = logging.getLogger(__name__)


class Sintetizador(Protocol):
    sample_rate: int

    async def sintetizar(self, texto: str) -> bytes: ...


class PiperTTS:
    """Sintetizador Piper. Devuelve PCM16 mono a `sample_rate`."""

    def __init__(
        self,
        sample_rate: int = 8000,
        settings: Optional[Settings] = None,
        max_memoria: int = 256,
    ) -> None:
        self.s = settings or get_settings()
        self.sample_rate = sample_rate
        self._voz = None
        self._lock = threading.Lock()
        self._memoria: "OrderedDict[str, bytes]" = OrderedDict()
        self._max_memoria = max_memoria
        self._dir_cache = self.s.tts_cache_dir / f"{self.s.piper_voice}-{sample_rate}"

    # --- carga -------------------------------------------------------------

    def ruta_modelo(self) -> Path:
        return self.s.piper_dir / f"{self.s.piper_voice}.onnx"

    def _cargar(self):  # type: ignore[no-untyped-def]
        if self._voz is None:
            from piper import PiperVoice

            ruta = self.ruta_modelo()
            if not ruta.is_file():
                log.info("Descargando la voz de Piper %s...", self.s.piper_voice)
                from piper.download_voices import download_voice

                self.s.piper_dir.mkdir(parents=True, exist_ok=True)
                download_voice(self.s.piper_voice, self.s.piper_dir)
            self._voz = PiperVoice.load(ruta, use_cuda=self.s.piper_cuda)
        return self._voz

    # --- síntesis ----------------------------------------------------------

    def _clave(self, texto: str) -> str:
        return hashlib.sha1(f"{self.s.piper_voice}|{self.sample_rate}|{self.s.piper_length_scale}|{texto}".encode()).hexdigest()

    def _sintetizar_sync(self, texto: str) -> bytes:
        from piper import SynthesisConfig

        with self._lock:
            voz = self._cargar()
            config = SynthesisConfig(length_scale=self.s.piper_length_scale)
            trozos = list(voz.synthesize(texto, syn_config=config))
        if not trozos:
            return b""
        origen = trozos[0].sample_rate
        audio = b"".join(t.audio_int16_bytes for t in trozos)
        return remuestrear(audio, origen, self.sample_rate)

    async def sintetizar(self, texto: str) -> bytes:
        """Sintetiza `texto`, usando la caché si ya se dijo antes."""
        texto = texto.strip()
        if not texto:
            return b""
        clave = self._clave(texto)
        if clave in self._memoria:
            self._memoria.move_to_end(clave)
            return self._memoria[clave]
        archivo = self._dir_cache / f"{clave}.pcm"
        if archivo.is_file():
            audio = archivo.read_bytes()
        else:
            audio = await asyncio.to_thread(self._sintetizar_sync, texto)
        self._memoria[clave] = audio
        if len(self._memoria) > self._max_memoria:
            self._memoria.popitem(last=False)
        return audio

    async def precalentar(self, frases: Iterable[str]) -> None:
        """Sintetiza y guarda en disco las frases fijas para que suenen al instante."""
        self._dir_cache.mkdir(parents=True, exist_ok=True)
        for frase in frases:
            archivo = self._dir_cache / f"{self._clave(frase.strip())}.pcm"
            if not archivo.is_file():
                audio = await asyncio.to_thread(self._sintetizar_sync, frase.strip())
                archivo.write_bytes(audio)
            await self.sintetizar(frase)


class TonoTTS:
    """Sintetizador falso para pruebas: un tono cuya duración depende del texto."""

    def __init__(self, sample_rate: int = 8000, ms_por_caracter: int = 10) -> None:
        self.sample_rate = sample_rate
        self.ms_por_caracter = ms_por_caracter

    async def sintetizar(self, texto: str) -> bytes:
        n = int(self.sample_rate * len(texto) * self.ms_por_caracter / 1000)
        t = np.arange(n) / self.sample_rate
        onda = 3000 * np.sin(2 * math.pi * 440 * t)
        return onda.astype(np.int16).tobytes()
