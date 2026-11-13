"""Extremo RTP para el audio de Asterisk External Media.

Asterisk envía el audio del cliente por RTP/UDP a nuestro puerto, y nosotros le
devolvemos el audio del agente a la dirección desde la que llegan sus paquetes.

Formato: G.711 μ-law (payload type 0), 8 kHz, paquetes de 20 ms (160 muestras).
Se usa μ-law y no `slin16` a propósito: en Asterisk 20 a 24, `chan_rtp` descarta
sin avisar el audio entrante en formatos con payload type dinámico (slin16/slin48),
mientras que μ-law (tipo estático 0) funciona en ambos sentidos
(https://github.com/asterisk/asterisk/issues/2268).
"""

from __future__ import annotations

import asyncio
import logging
import random
import struct
import time
from dataclasses import dataclass
from typing import Optional

from app.audio.resample import pcm16_a_ulaw, ulaw_a_pcm16
from app.telephony.base import BufferSalida

log = logging.getLogger(__name__)

PT_ULAW = 0
SAMPLE_RATE = 8000
MUESTRAS_TRAMA = 160  # 20 ms a 8 kHz
BYTES_PCM_TRAMA = MUESTRAS_TRAMA * 2


@dataclass
class PaqueteRTP:
    payload_type: int
    secuencia: int
    timestamp: int
    ssrc: int
    payload: bytes
    marcador: bool = False

    def a_bytes(self) -> bytes:
        cabecera = struct.pack(
            "!BBHII",
            0x80,  # versión 2, sin padding, sin extensión, sin CSRC
            (0x80 if self.marcador else 0) | (self.payload_type & 0x7F),
            self.secuencia & 0xFFFF,
            self.timestamp & 0xFFFFFFFF,
            self.ssrc,
        )
        return cabecera + self.payload

    @classmethod
    def desde_bytes(cls, datos: bytes) -> Optional["PaqueteRTP"]:
        if len(datos) < 12:
            return None
        b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", datos[:12])
        if b0 >> 6 != 2:
            return None
        inicio = 12 + 4 * (b0 & 0x0F)  # CSRC
        if b0 & 0x10:  # extensión de cabecera
            if len(datos) < inicio + 4:
                return None
            longitud = struct.unpack("!H", datos[inicio + 2 : inicio + 4])[0]
            inicio += 4 + 4 * longitud
        fin = len(datos)
        if b0 & 0x20:  # padding
            fin -= datos[-1]
        if inicio > fin:
            return None
        return cls(b1 & 0x7F, seq, ts, ssrc, datos[inicio:fin], bool(b1 & 0x80))


class EndpointRTP(asyncio.DatagramProtocol):
    """Recibe y envía audio RTP μ-law a ritmo real."""

    def __init__(self) -> None:
        self.entrada: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)
        self.salida = BufferSalida()
        self.remoto: Optional[tuple[str, int]] = None
        self.transporte: Optional[asyncio.DatagramTransport] = None
        self._seq = random.randint(0, 0xFFFF)
        self._ts = random.randint(0, 0xFFFFFFFF)
        self._ssrc = random.randint(1, 0xFFFFFFFF)
        self._tarea_envio: Optional[asyncio.Task] = None
        self.paquetes_recibidos = 0
        self.paquetes_enviados = 0

    @classmethod
    async def crear(cls, host: str, puerto: int) -> "EndpointRTP":
        loop = asyncio.get_running_loop()
        _transporte, protocolo = await loop.create_datagram_endpoint(cls, local_addr=(host, puerto))
        protocolo._tarea_envio = asyncio.create_task(protocolo._bucle_envio())
        return protocolo

    # --- asyncio.DatagramProtocol -----------------------------------------

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transporte = transport  # type: ignore[assignment]

    def datagram_received(self, datos: bytes, direccion: tuple[str, int]) -> None:
        paquete = PaqueteRTP.desde_bytes(datos)
        if paquete is None or paquete.payload_type != PT_ULAW:
            return
        self.remoto = direccion  # RTP simétrico: respondemos a quien nos envía
        self.paquetes_recibidos += 1
        try:
            self.entrada.put_nowait(ulaw_a_pcm16(paquete.payload))
        except asyncio.QueueFull:
            # Si el consumidor se atasca, mejor perder audio antiguo que acumular retraso.
            try:
                self.entrada.get_nowait()
                self.entrada.put_nowait(ulaw_a_pcm16(paquete.payload))
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    # --- envío ---------------------------------------------------------------

    async def _bucle_envio(self) -> None:
        """Envía una trama cada 20 ms con reloj monotónico (sin deriva acumulada)."""
        siguiente = time.monotonic()
        hablando = False
        while True:
            siguiente += 0.02
            await asyncio.sleep(max(0.0, siguiente - time.monotonic()))
            if self.remoto is None or self.transporte is None:
                continue
            hay_audio = self.salida.pendiente > 0
            pcm = self.salida.leer(BYTES_PCM_TRAMA)
            paquete = PaqueteRTP(
                PT_ULAW,
                self._seq,
                self._ts,
                self._ssrc,
                pcm16_a_ulaw(pcm),
                marcador=hay_audio and not hablando,  # inicio de ráfaga de voz
            )
            hablando = hay_audio
            self.transporte.sendto(paquete.a_bytes(), self.remoto)
            self.paquetes_enviados += 1
            self._seq = (self._seq + 1) & 0xFFFF
            self._ts = (self._ts + MUESTRAS_TRAMA) & 0xFFFFFFFF

    def cerrar(self) -> None:
        if self._tarea_envio:
            self._tarea_envio.cancel()
        if self.transporte:
            self.transporte.close()


class AsignadorPuertos:
    """Reparte puertos UDP de un rango para las llamadas simultáneas."""

    def __init__(self, inicio: int, fin: int) -> None:
        self._libres = list(range(inicio, fin, 2))  # puertos pares, como es habitual en RTP
        self._lock = asyncio.Lock()

    async def tomar(self) -> int:
        async with self._lock:
            if not self._libres:
                raise RuntimeError("No quedan puertos RTP libres")
            return self._libres.pop(0)

    async def devolver(self, puerto: int) -> None:
        async with self._lock:
            if puerto not in self._libres:
                self._libres.append(puerto)
