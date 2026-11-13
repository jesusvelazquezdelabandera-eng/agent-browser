"""Detección de actividad de voz (VAD) y de silencios.

Trabaja con tramas de 20 ms y genera dos eventos:
- `InicioVoz`: el cliente lleva hablando `inicio_ms`. Sirve para interrumpir al
  agente (barge-in) en cuanto el cliente empieza a hablar.
- `FinVoz`: el cliente ha dejado de hablar durante `silencio_fin_ms`. Lleva el
  audio completo del turno (con un poco de audio previo para no cortar la primera
  sílaba) listo para transcribir.

Backends:
- `webrtc`: webrtcvad (rápido, sin modelos). Admite 8 kHz y 16 kHz, así que
  funciona directamente con el audio telefónico.
- `energia`: umbral de energía. Sin dependencias; útil en pruebas.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Union

from app.audio.resample import rms

TRAMA_MS = 20


@dataclass(frozen=True)
class InicioVoz:
    pass


@dataclass(frozen=True)
class FinVoz:
    audio: bytes
    duracion_ms: int


EventoVAD = Union[InicioVoz, FinVoz]


class _ClasificadorEnergia:
    def __init__(self, umbral: float) -> None:
        self.umbral = umbral

    def is_speech(self, trama: bytes, sample_rate: int) -> bool:  # misma firma que webrtcvad
        return rms(trama) > self.umbral


class DetectorVoz:
    """Máquina de estados del VAD sobre tramas de 20 ms."""

    def __init__(
        self,
        sample_rate: int = 8000,
        backend: str = "webrtc",
        agresividad: int = 2,
        inicio_ms: int = 250,
        silencio_fin_ms: int = 600,
        min_voz_ms: int = 250,
        previo_ms: int = 300,
        umbral_energia: float = 500.0,
    ) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("El VAD necesita 8, 16, 32 o 48 kHz")
        self.sample_rate = sample_rate
        self.bytes_trama = sample_rate * TRAMA_MS // 1000 * 2
        if backend == "webrtc":
            try:
                import webrtcvad
            except ImportError as e:  # pragma: no cover - depende de la instalación
                raise RuntimeError("Instala webrtcvad-wheels o usa VAD_BACKEND=energia") from e
            self._vad = webrtcvad.Vad(agresividad)
        else:
            self._vad = _ClasificadorEnergia(umbral_energia)
        self.tramas_inicio = max(1, inicio_ms // TRAMA_MS)
        self.tramas_fin = max(1, silencio_fin_ms // TRAMA_MS)
        self.tramas_min_voz = max(1, min_voz_ms // TRAMA_MS)
        self._previo: deque[bytes] = deque(maxlen=max(1, previo_ms // TRAMA_MS))
        self._pendiente = b""
        self.reiniciar()

    def reiniciar(self) -> None:
        self._hablando = False
        self._voz_seguida = 0
        self._silencio_seguido = 0
        self._tramas_voz = 0
        self._inicio_emitido = False
        self._audio: list[bytes] = []

    @property
    def hablando(self) -> bool:
        """True mientras se está capturando un turno del cliente."""
        return self._hablando

    def procesar(self, audio: bytes) -> list[EventoVAD]:
        """Procesa audio PCM16 de cualquier longitud y devuelve los eventos producidos."""
        self._pendiente += audio
        eventos: list[EventoVAD] = []
        while len(self._pendiente) >= self.bytes_trama:
            trama = self._pendiente[: self.bytes_trama]
            self._pendiente = self._pendiente[self.bytes_trama :]
            evento = self._procesar_trama(trama)
            if evento is not None:
                eventos.append(evento)
        return eventos

    def _procesar_trama(self, trama: bytes) -> Optional[EventoVAD]:
        es_voz = self._vad.is_speech(trama, self.sample_rate)

        if not self._hablando:
            self._previo.append(trama)
            self._voz_seguida = self._voz_seguida + 1 if es_voz else 0
            # Empieza a capturar con 2 tramas de voz seguidas (40 ms) para no perder el inicio.
            if self._voz_seguida >= min(2, self.tramas_inicio):
                self._hablando = True
                self._audio = list(self._previo)
                self._tramas_voz = self._voz_seguida
                self._silencio_seguido = 0
                self._inicio_emitido = False
            return None

        self._audio.append(trama)
        if es_voz:
            self._tramas_voz += 1
            self._silencio_seguido = 0
        else:
            self._silencio_seguido += 1

        if not self._inicio_emitido and self._tramas_voz >= self.tramas_inicio:
            self._inicio_emitido = True
            return InicioVoz()

        if self._silencio_seguido >= self.tramas_fin:
            audio = b"".join(self._audio)
            voz_suficiente = self._tramas_voz >= self.tramas_min_voz
            self.reiniciar()
            self._previo.clear()
            if voz_suficiente:
                return FinVoz(audio=audio, duracion_ms=len(audio) // 2 * 1000 // self.sample_rate)
        return None
