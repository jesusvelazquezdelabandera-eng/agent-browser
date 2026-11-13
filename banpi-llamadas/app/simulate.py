"""Modo simulación por consola, sin telefonía.

Uso:
    python -m app.simulate                      # conversación escribiendo (texto)
    python -m app.simulate --modo microfono     # hablando con el micrófono (usa auriculares)
    python -m app.simulate --modo guion --guion data/guion_ejemplo.txt
        # el "cliente" dice las frases del archivo con voz sintética y se usa el
        # pipeline completo (VAD, Whisper, LLM, Piper); guarda el audio del agente

Opciones comunes:
    --llm plantillas   no usa Ollama (respuestas de respaldo); útil para probar sin modelo
    --nombre "María"   nombre del cliente para el saludo
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from app.agent import prompts
from app.agent.dialog import Accion, Dialogo
from app.agent.llm import Conversacion, crear_llm, respuesta_en_frases
from app.agent.pipeline import SesionLlamada
from app.config import RAIZ_PROYECTO, Settings, get_settings


class OyenteConsola:
    """Muestra la conversación en la consola."""

    def __init__(self) -> None:
        self.latencias: list[int] = []

    def al_hablar(self, texto: str, estado: str, latencia_ms: Optional[int]) -> None:
        extra = f"  ({latencia_ms} ms)" if latencia_ms is not None else ""
        if latencia_ms is not None:
            self.latencias.append(latencia_ms)
        print(f"ÁLEX    [{estado}] {texto}{extra}", flush=True)

    def al_escuchar(self, texto: str, estado: str) -> None:
        print(f"CLIENTE [{estado}] {texto}", flush=True)

    def al_accion(self, accion: Accion, dialogo: Dialogo) -> None:
        print(f"  → acción: {accion.value}", flush=True)


async def modo_texto(s: Settings, nombre: str) -> None:
    """Conversación escrita: tú haces de cliente."""
    llm = crear_llm(s)
    await llm.calentar()
    dialogo = Dialogo(nombre, settings=s)
    conversacion = Conversacion(prompts.prompt_sistema(s))
    print("Simulación por texto. Escribe lo que diría el cliente (Ctrl+C para salir).\n")

    turno = dialogo.inicio()
    while True:
        dicho: list[str] = []
        t0 = time.monotonic()
        primera: Optional[float] = None
        for frase in prompts.dividir_frases(turno.texto):
            dicho.append(frase)
        if turno.instruccion:
            mensajes = conversacion.mensajes(turno.instruccion, texto_previo=turno.texto)
            async for frase in respuesta_en_frases(llm, mensajes, turno.respaldo, max_frases=turno.max_frases, sin_preguntas=bool(turno.texto_final)):
                primera = primera or time.monotonic()
                dicho.append(frase)
        dicho.extend(prompts.dividir_frases(turno.texto_final))
        texto = " ".join(dicho)
        conversacion.agente(texto)
        latencia = f"  (LLM: {int((primera - t0) * 1000)} ms hasta la 1.ª frase)" if primera else ""
        print(f"\nÁLEX [{turno.estado.value}]: {texto}{latencia}")
        for accion in turno.acciones:
            print(f"  → acción: {accion.value}")
        if turno.terminar:
            break
        try:
            entrada = await asyncio.to_thread(input, "TÚ: ")
        except EOFError:
            break
        conversacion.cliente(entrada)
        turno = dialogo.procesar(entrada)

    d = dialogo.datos
    print("\n--- Resumen ---")
    print(f"Resultado: {d.resultado.value if d.resultado else '-'}")
    print(f"Datos: {d.resumen()}")
    if d.cita_texto:
        print(f"Cita ({d.cita_tipo}): {d.cita_texto}")
    if d.rellamar_texto:
        print(f"Volver a llamar: {d.rellamar_texto}")
    print(f"Grabación permitida: {'sí' if d.grabacion_permitida else 'no'}")


async def modo_voz(s: Settings, nombre: str, guion: Optional[Path], salida_wav: Path) -> None:
    """Usa el pipeline completo con micrófono o con un guion sintetizado."""
    from app.audio.stt import WhisperSTT
    from app.audio.tts import PiperTTS

    llm = crear_llm(s)
    stt = WhisperSTT(s)
    print("Cargando modelos (la primera vez puede tardar en descargarlos)...", flush=True)

    if guion is None:
        from app.telephony.simulada import LlamadaMicrofono

        sr = 16000
        tts = PiperTTS(sr, s)
        await asyncio.gather(stt.calentar(), llm.calentar(), tts.sintetizar("Hola."))
        llamada = LlamadaMicrofono(sr)
        print("Habla cuando quieras (usa auriculares). Ctrl+C para colgar.\n")
    else:
        from app.telephony.simulada import FraseCliente, LlamadaGuionada

        sr = 8000  # como en una llamada real
        tts = PiperTTS(sr, s)
        await asyncio.gather(stt.calentar(), llm.calentar())
        # La voz del "cliente" se genera con Piper un poco más lenta para diferenciarla.
        frases = [
            linea.strip()
            for linea in guion.read_text(encoding="utf-8").splitlines()
            if linea.strip() and not linea.startswith("#")
        ]
        guion_audio = []
        for frase in frases:
            interrumpir = None
            if frase.startswith("!"):  # "!texto" = el cliente interrumpe al agente
                frase, interrumpir = frase[1:].strip(), 1500
            guion_audio.append(
                FraseCliente(await tts.sintetizar(frase), texto=frase, interrumpir_tras_ms=interrumpir)
            )
        llamada = LlamadaGuionada(guion_audio, sample_rate=sr)
        print(f"Guion con {len(frases)} intervenciones del cliente.\n")

    oyente = OyenteConsola()
    sesion = SesionLlamada(llamada, Dialogo(nombre, settings=s), llm, stt, tts, s, oyente)
    try:
        resultado = await sesion.ejecutar()
    except (KeyboardInterrupt, asyncio.CancelledError):
        resultado = None
        await llamada.colgar()
    print("\n--- Resumen ---")
    print(f"Resultado: {resultado.value if resultado else '-'}")
    if oyente.latencias:
        print(
            f"Latencia percibida: media {sum(oyente.latencias) // len(oyente.latencias)} ms, "
            f"máxima {max(oyente.latencias)} ms (objetivo < 1500 ms)"
        )
    if sesion.metricas.stt_ms:
        stt = sesion.metricas.stt_ms
        print(f"Transcripción (Whisper): media {sum(stt) // len(stt)} ms por turno")
    print(f"Interrupciones: {sesion.metricas.interrupciones}")
    if guion is not None:
        llamada.guardar_wav(salida_wav)  # type: ignore[attr-defined]
        print(f"Audio del agente guardado en {salida_wav}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Simulación del agente de llamadas de Banpi")
    parser.add_argument("--modo", choices=["texto", "microfono", "guion"], default="texto")
    parser.add_argument("--guion", type=Path, default=RAIZ_PROYECTO / "data" / "guion_ejemplo.txt")
    parser.add_argument("--salida", type=Path, default=RAIZ_PROYECTO / "data" / "simulacion_agente.wav")
    parser.add_argument("--llm", choices=["ollama", "plantillas"], default=None)
    parser.add_argument("--nombre", default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    s = get_settings()
    if args.llm:
        from dataclasses import replace

        s = replace(s, llm_backend=args.llm)
    try:
        if args.modo == "texto":
            asyncio.run(modo_texto(s, args.nombre))
        else:
            guion = args.guion if args.modo == "guion" else None
            asyncio.run(modo_voz(s, args.nombre, guion, args.salida))
    except KeyboardInterrupt:
        print("\nSimulación terminada.")
        sys.exit(0)


if __name__ == "__main__":
    main()
