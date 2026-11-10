"""Llamadas simuladas, sin telefonía.

- `LlamadaGuionada`: el "cliente" dice frases de un guion en cuanto el agente
  termina de hablar (o le interrumpe si así se indica). El audio del cliente puede
  ser voz sintetizada real (para probar Whisper) o un tono (para pruebas rápidas).
  Graba todo lo que dice el agente para poder escucharlo.
- `LlamadaMicrofono`: usa el micrófono y los altavoces del ordenador.

Ambas implementan la misma interfaz `Llamada` que Asterisk, así que el pipeline
que se prueba aquí es exactamente el mismo que el de las llamadas reales.
"""

from __future__ import annotations

import asyncio
import queue
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from app.audio.resample import silencio
from app.telephony.base import BufferSalida, Llamada

TRAMA_MS = 20


@dataclass
class FraseCliente:
    """Una intervención del cliente en el guion."""

    audio: bytes
    texto: str = ""
    pausa_ms: int = 400  # silencio antes de hablar, tras terminar el agente
    interrumpir_tras_ms: Optional[int] = None  # habla mientras el agente aún habla


class LlamadaGuionada(Llamada):
    """Llamada simulada a partir de un guion de audio del cliente.

    `velocidad` > 1 acelera el reloj para que las pruebas sean rápidas.
    """

    def __init__(
        self,
        guion: list[FraseCliente],
        sample_rate: int = 8000,
        velocidad: float = 1.0,
        colgar_al_final: bool = True,
    ) -> None:
        super().__init__()
        self.id = "simulada"
        self.sample_rate = sample_rate
        self.guion = list(guion)
        self.velocidad = velocidad
        self.colgar_al_final = colgar_al_final
        self.salida = BufferSalida()
        self.audio_agente = bytearray()  # todo lo que se reprodujo
        self.vaciados = 0
        self.grabando = False
        self.grabacion_borrada = False
        self._bytes_trama = sample_rate * TRAMA_MS // 1000 * 2

    async def frames(self) -> AsyncIterator[bytes]:
        dt = TRAMA_MS / 1000 / self.velocidad
        audio_cliente = b""
        espera_ms = 0
        ms_agente_seguidos = 0
        # El cliente solo responde cuando el agente ha contestado a su frase anterior
        # (si no, hablaría mientras el agente aún está pensando la respuesta).
        agente_respondio = False
        ms_sin_respuesta = 0
        while not self.colgada.is_set():
            await asyncio.sleep(dt)
            # Reproducción del agente al ritmo del reloj.
            if self.salida.pendiente:
                self.audio_agente += self.salida.leer(self._bytes_trama)
                ms_agente_seguidos += TRAMA_MS
                agente_respondio = True
            else:
                ms_agente_seguidos = 0

            if not audio_cliente and self.guion:
                siguiente = self.guion[0]
                ms_sin_respuesta += TRAMA_MS
                if siguiente.interrumpir_tras_ms is not None:
                    if ms_agente_seguidos >= siguiente.interrumpir_tras_ms:
                        audio_cliente = self.guion.pop(0).audio
                        agente_respondio, ms_sin_respuesta = False, 0
                elif not self.salida.pendiente and (agente_respondio or ms_sin_respuesta > 10000):
                    espera_ms += TRAMA_MS
                    if espera_ms >= siguiente.pausa_ms:
                        audio_cliente = self.guion.pop(0).audio
                        espera_ms = 0
                        agente_respondio, ms_sin_respuesta = False, 0
                else:
                    espera_ms = 0

            if audio_cliente:
                trama, audio_cliente = audio_cliente[: self._bytes_trama], audio_cliente[self._bytes_trama :]
                trama = trama.ljust(self._bytes_trama, b"\x00")
            else:
                trama = silencio(TRAMA_MS, self.sample_rate)
            yield trama

            if self.colgar_al_final and not self.guion and not audio_cliente and espera_ms > 15000:
                self.colgada.set()

    async def enviar_audio(self, pcm: bytes) -> None:
        self.salida.escribir(pcm)

    def vaciar_salida(self) -> None:
        if self.salida.vaciar():
            self.vaciados += 1

    @property
    def reproduciendo(self) -> bool:
        return self.salida.pendiente > 0

    async def colgar(self) -> None:
        self.colgada.set()

    async def iniciar_grabacion(self, nombre: str) -> Optional[str]:
        self.grabando = True
        return nombre

    async def detener_grabacion(self, borrar: bool = False) -> None:
        self.grabando = False
        self.grabacion_borrada = borrar

    def guardar_wav(self, ruta: Path) -> None:
        """Guarda en WAV todo lo que dijo el agente."""
        with wave.open(str(ruta), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(bytes(self.audio_agente))


class LlamadaMicrofono(Llamada):  # pragma: no cover - necesita tarjeta de sonido
    """Llamada simulada con micrófono y altavoces (requiere `sounddevice`).

    Usa auriculares: con altavoces, el micrófono capta la voz del agente y este se
    interrumpe a sí mismo.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        super().__init__()
        import sounddevice as sd

        self.id = "microfono"
        self.sample_rate = sample_rate
        self.salida = BufferSalida()
        self._entrada: "queue.Queue[bytes]" = queue.Queue()
        bloque = sample_rate * TRAMA_MS // 1000

        def al_capturar(indata, frames, tiempo, estado):  # type: ignore[no-untyped-def]
            self._entrada.put(bytes(indata))

        def al_reproducir(outdata, frames, tiempo, estado):  # type: ignore[no-untyped-def]
            outdata[:] = self.salida.leer(frames * 2)

        self._in = sd.RawInputStream(samplerate=sample_rate, channels=1, dtype="int16",
                                     blocksize=bloque, callback=al_capturar)  # fmt: skip
        self._out = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16",
                                       blocksize=bloque, callback=al_reproducir)  # fmt: skip
        self._in.start()
        self._out.start()

    async def frames(self) -> AsyncIterator[bytes]:
        while not self.colgada.is_set():
            try:
                yield await asyncio.to_thread(self._entrada.get, True, 0.5)
            except queue.Empty:
                continue

    async def enviar_audio(self, pcm: bytes) -> None:
        self.salida.escribir(pcm)

    def vaciar_salida(self) -> None:
        self.salida.vaciar()

    @property
    def reproduciendo(self) -> bool:
        return self.salida.pendiente > 0

    async def colgar(self) -> None:
        self.colgada.set()
        for stream in (self._in, self._out):
            stream.stop()
            stream.close()
