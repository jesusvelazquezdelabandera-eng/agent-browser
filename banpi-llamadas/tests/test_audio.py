"""Pruebas de conversión de audio y del VAD."""

from __future__ import annotations

import numpy as np

from app.audio.resample import pcm16_a_ulaw, remuestrear, silencio, ulaw_a_pcm16
from app.audio.vad import DetectorVoz, FinVoz, InicioVoz


def _tono(ms: int, sr: int, amplitud: int = 6000, hz: int = 300) -> bytes:
    t = np.arange(int(sr * ms / 1000)) / sr
    return (amplitud * np.sin(2 * np.pi * hz * t)).astype(np.int16).tobytes()


def test_ulaw_ida_y_vuelta() -> None:
    original = np.array([0, 100, -100, 1000, -1000, 8000, -8000, 32000, -32000], dtype=np.int16)
    vuelta = np.frombuffer(ulaw_a_pcm16(pcm16_a_ulaw(original.tobytes())), dtype=np.int16)
    # μ-law tiene error relativo pequeño (~3 %) y absoluto pequeño cerca de cero.
    assert np.all(np.abs(vuelta.astype(int) - original) <= np.maximum(8, np.abs(original) * 0.04))


def test_ulaw_valores_conocidos() -> None:
    assert pcm16_a_ulaw(np.array([0], dtype=np.int16).tobytes()) == b"\xff"
    assert ulaw_a_pcm16(b"\xff") == np.array([0], dtype=np.int16).tobytes()


def test_remuestrear_longitud() -> None:
    audio = _tono(1000, 22050)
    assert len(remuestrear(audio, 22050, 8000)) == 8000 * 2
    assert len(remuestrear(_tono(1000, 8000), 8000, 16000)) == 16000 * 2


def test_remuestrear_conserva_tono() -> None:
    salida = np.frombuffer(remuestrear(_tono(1000, 22050, hz=440), 22050, 8000), dtype=np.int16)
    espectro = np.abs(np.fft.rfft(salida))
    frecuencia = np.argmax(espectro) * 8000 / len(salida)
    assert abs(frecuencia - 440) < 5


def _detector() -> DetectorVoz:
    return DetectorVoz(8000, backend="energia", inicio_ms=200, silencio_fin_ms=400, min_voz_ms=200)


def test_vad_detecta_turno() -> None:
    vad = _detector()
    audio = silencio(300, 8000) + _tono(800, 8000) + silencio(600, 8000)
    eventos = vad.procesar(audio)
    assert isinstance(eventos[0], InicioVoz)
    assert isinstance(eventos[1], FinVoz)
    # Incluye la voz completa más algo de margen previo y el silencio final.
    assert 800 <= eventos[1].duracion_ms <= 800 + 300 + 400 + 40


def test_vad_ignora_ruido_corto() -> None:
    vad = _detector()
    eventos = vad.procesar(silencio(200, 8000) + _tono(60, 8000) + silencio(600, 8000))
    assert eventos == []


def test_vad_tramas_de_tamano_irregular() -> None:
    vad = _detector()
    audio = _tono(600, 8000) + silencio(600, 8000)
    eventos = []
    for i in range(0, len(audio), 222):  # trozos que no coinciden con 20 ms
        eventos += vad.procesar(audio[i : i + 222])
    assert [type(e) for e in eventos] == [InicioVoz, FinVoz]


def test_vad_webrtc_carga() -> None:
    vad = DetectorVoz(8000, backend="webrtc")
    assert vad.procesar(silencio(500, 8000)) == []
