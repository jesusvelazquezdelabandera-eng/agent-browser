"""Conversión de formatos de audio: G.711 μ-law, PCM16 y cambio de frecuencia.

`audioop` ya no existe en Python 3.13, así que el μ-law se implementa con numpy.
Todo el audio interno es PCM de 16 bits, mono, little-endian.
"""

from __future__ import annotations

import numpy as np

_SESGO = 0x84
_MAXIMO = 32635


def _tabla_decodificacion() -> np.ndarray:
    u = ~np.arange(256, dtype=np.uint8)
    signo = u & 0x80
    exponente = (u >> 4) & 0x07
    mantisa = u & 0x0F
    muestra = ((mantisa.astype(np.int32) << 3) + _SESGO) << exponente
    muestra = muestra - _SESGO
    return np.where(signo != 0, -muestra, muestra).astype(np.int16)


_DECODIFICAR = _tabla_decodificacion()


def ulaw_a_pcm16(datos: bytes) -> bytes:
    """Decodifica G.711 μ-law a PCM16."""
    return _DECODIFICAR[np.frombuffer(datos, dtype=np.uint8)].tobytes()


def pcm16_a_ulaw(datos: bytes) -> bytes:
    """Codifica PCM16 a G.711 μ-law."""
    x = np.frombuffer(datos, dtype=np.int16).astype(np.int32)
    signo = np.where(x < 0, 0x80, 0)
    x = np.minimum(np.abs(x), _MAXIMO) + _SESGO
    exponente = np.floor(np.log2(x)).astype(np.int32) - 7
    exponente = np.clip(exponente, 0, 7)
    mantisa = (x >> (exponente + 3)) & 0x0F
    return (~(signo | (exponente << 4) | mantisa) & 0xFF).astype(np.uint8).tobytes()


def _filtro_paso_bajo(x: np.ndarray, corte: float, taps: int = 63) -> np.ndarray:
    """FIR de ventana de Hamming. `corte` es la fracción de la frecuencia de Nyquist."""
    n = np.arange(taps) - (taps - 1) / 2
    h = corte * np.sinc(corte * n) * np.hamming(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same")


def remuestrear(datos: bytes, origen: int, destino: int) -> bytes:
    """Cambia la frecuencia de muestreo de un bloque PCM16 completo.

    Pensado para bloques enteros (una frase, un turno del cliente), no para
    tramas sueltas de 20 ms, donde se notarían los bordes.
    """
    if origen == destino or not datos:
        return datos
    x = np.frombuffer(datos, dtype=np.int16).astype(np.float32)
    if destino < origen:
        # Antes de bajar la frecuencia se filtra para evitar aliasing.
        x = _filtro_paso_bajo(x, corte=0.9 * destino / origen)
    n_salida = int(round(len(x) * destino / origen))
    t_entrada = np.arange(len(x)) / origen
    t_salida = np.arange(n_salida) / destino
    y = np.interp(t_salida, t_entrada, x)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16).tobytes()


def silencio(ms: int, sample_rate: int) -> bytes:
    """Bloque de silencio PCM16."""
    return b"\x00\x00" * int(sample_rate * ms / 1000)


def rms(datos: bytes) -> float:
    """Energía media (RMS) de un bloque PCM16."""
    if not datos:
        return 0.0
    x = np.frombuffer(datos, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))


def duracion_ms(datos: bytes, sample_rate: int) -> int:
    return int(len(datos) / 2 / sample_rate * 1000)
