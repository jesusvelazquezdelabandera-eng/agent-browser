"""Interfaz abstracta de telefonía.

El resto de la aplicación solo conoce estas clases, así que se puede cambiar de
Asterisk a Twilio (u otro proveedor) implementando `Telefonia` y `Llamada`.

Convenciones de audio: PCM16 mono little-endian a `Llamada.sample_rate`
(8000 Hz en telefonía), en tramas de unos 20 ms.
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional


class EstadoMarcacion(str, Enum):
    CONTESTADA = "contestada"
    NO_CONTESTA = "no_contesta"
    OCUPADO = "ocupado"
    RECHAZADA = "rechazada"
    FALLO = "fallo"


@dataclass
class ResultadoMarcacion:
    estado: EstadoMarcacion
    llamada: Optional["Llamada"] = None
    detalle: str = ""


class BufferSalida:
    """Cola de audio de salida, segura entre hilos.

    El emisor (RTP o tarjeta de sonido) lee tramas a ritmo real; `vaciar()` corta
    al instante lo pendiente, que es lo que permite que el cliente interrumpa.
    """

    def __init__(self) -> None:
        self._datos = bytearray()
        self._lock = threading.Lock()

    def escribir(self, pcm: bytes) -> None:
        with self._lock:
            self._datos.extend(pcm)

    def leer(self, n: int) -> bytes:
        """Lee `n` bytes; si no hay suficientes, completa con silencio."""
        with self._lock:
            trozo = bytes(self._datos[:n])
            del self._datos[:n]
        if len(trozo) < n:
            trozo += b"\x00" * (n - len(trozo))
        return trozo

    def vaciar(self) -> int:
        with self._lock:
            n = len(self._datos)
            self._datos.clear()
            return n

    @property
    def pendiente(self) -> int:
        with self._lock:
            return len(self._datos)


class Llamada(ABC):
    """Una llamada en curso con audio bidireccional."""

    id: str
    sample_rate: int = 8000

    def __init__(self) -> None:
        self.colgada = asyncio.Event()

    @abstractmethod
    def frames(self) -> AsyncIterator[bytes]:
        """Audio del cliente en tramas PCM16. Termina cuando se cuelga."""

    @abstractmethod
    async def enviar_audio(self, pcm: bytes) -> None:
        """Encola audio PCM16 para que lo oiga el cliente."""

    @abstractmethod
    def vaciar_salida(self) -> None:
        """Descarta todo el audio pendiente de reproducir (interrupción)."""

    @property
    @abstractmethod
    def reproduciendo(self) -> bool:
        """True si queda audio del agente por reproducir."""

    @abstractmethod
    async def colgar(self) -> None:
        """Cuelga la llamada y libera los recursos."""

    async def esperar_fin_reproduccion(self, timeout: float = 30.0) -> None:
        """Espera a que el cliente haya oído todo el audio encolado."""
        fin = asyncio.get_running_loop().time() + timeout
        while self.reproduciendo and not self.colgada.is_set():
            if asyncio.get_running_loop().time() > fin:
                break
            await asyncio.sleep(0.02)

    async def iniciar_grabacion(self, nombre: str) -> Optional[str]:
        """Empieza a grabar. Devuelve la ruta o el identificador de la grabación."""
        return None

    async def detener_grabacion(self, borrar: bool = False) -> None:
        """Detiene la grabación; con `borrar=True` la elimina (el cliente se opuso)."""
        return None


class Telefonia(ABC):
    """Proveedor de telefonía capaz de originar llamadas."""

    async def iniciar(self) -> None:
        """Conecta con el proveedor (eventos, autenticación...)."""

    async def detener(self) -> None:
        """Cierra conexiones."""

    @abstractmethod
    async def originar(self, numero: str, caller_id: str = "", timeout_s: int = 30) -> ResultadoMarcacion:
        """Llama a `numero` (formato E.164) y espera a que conteste o falle."""
