"""Bucle de una llamada: audio del cliente -> VAD -> STT -> diálogo -> LLM -> TTS -> audio.

Puntos clave:
- **Latencia**: al detectar el fin del turno se transcribe al momento. La respuesta
  del LLM llega en streaming, se corta por frases y cada frase se sintetiza y se
  encola en cuanto está lista, sin esperar al resto. Se mide la latencia percibida
  de cada turno: silencio de fin de turno + transcripción + primera frase sintetizada.
- **Interrupción (barge-in)**: si el cliente empieza a hablar mientras el agente
  habla, se cancela la generación y se vacía el audio pendiente al instante.
  El saludo inicial no se puede interrumpir, porque contiene los avisos legales
  (IA, finalidad comercial, grabación y derecho de oposición); lo que diga el
  cliente durante el saludo se procesa en cuanto termina.
- **Acciones inmediatas**: "no me llaméis más" se registra en cuanto se detecta,
  antes incluso de despedirse.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from app.agent import prompts
from app.agent.dialog import Accion, Dialogo, Resultado, Turno
from app.agent.llm import LLM, Conversacion, respuesta_en_frases
from app.audio.stt import Transcriptor
from app.audio.tts import Sintetizador
from app.audio.vad import DetectorVoz, FinVoz, InicioVoz
from app.config import Settings, get_settings
from app.telephony.base import Llamada

log = logging.getLogger(__name__)


class Oyente(Protocol):
    """Recibe lo que ocurre en la llamada (para guardarlo en la base de datos)."""

    def al_hablar(self, texto: str, estado: str, latencia_ms: Optional[int]) -> None: ...

    def al_escuchar(self, texto: str, estado: str) -> None: ...

    def al_accion(self, accion: Accion, dialogo: Dialogo) -> None: ...


class OyenteNulo:
    def al_hablar(self, texto: str, estado: str, latencia_ms: Optional[int]) -> None:
        log.info("AGENTE [%s] %s (latencia %s ms)", estado, texto, latencia_ms)

    def al_escuchar(self, texto: str, estado: str) -> None:
        log.info("CLIENTE [%s] %s", estado, texto)

    def al_accion(self, accion: Accion, dialogo: Dialogo) -> None:
        log.info("ACCIÓN %s", accion.value)


def trocear_para_voz(frase: str, max_caracteres: int = 45) -> list[str]:
    """Parte una frase larga por su primera coma.

    La síntesis tarda en proporción a la longitud del texto: si el primer trozo es
    corto ("Perfecto,"), suena antes y la latencia percibida baja.
    """
    if len(frase) <= max_caracteres:
        return [frase]
    i = frase.find(", ")
    if 3 <= i <= max_caracteres:
        return [frase[: i + 1], frase[i + 2 :]]
    return [frase]


@dataclass
class Metricas:
    latencias_ms: list[int] = field(default_factory=list)
    stt_ms: list[int] = field(default_factory=list)
    interrupciones: int = 0

    @property
    def latencia_media_ms(self) -> Optional[int]:
        if not self.latencias_ms:
            return None
        return int(sum(self.latencias_ms) / len(self.latencias_ms))


class SesionLlamada:
    """Conduce una llamada contestada de principio a fin."""

    def __init__(
        self,
        llamada: Llamada,
        dialogo: Dialogo,
        llm: LLM,
        stt: Transcriptor,
        tts: Sintetizador,
        settings: Optional[Settings] = None,
        oyente: Optional[Oyente] = None,
        silencio_max_s: float = 8.0,
        timeout_llm_s: float = 4.0,
    ) -> None:
        self.s = settings or get_settings()
        self.llamada = llamada
        self.dialogo = dialogo
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.oyente: Oyente = oyente or OyenteNulo()
        self.silencio_max_s = silencio_max_s
        self.timeout_llm_s = timeout_llm_s
        self.metricas = Metricas()
        self.conversacion = Conversacion(prompts.prompt_sistema(self.s))
        self.vad = DetectorVoz(
            llamada.sample_rate,
            backend=self.s.vad_backend,
            agresividad=self.s.vad_agresividad,
            inicio_ms=self.s.barge_in_ms,
            silencio_fin_ms=self.s.vad_silencio_fin_ms,
            min_voz_ms=self.s.vad_min_voz_ms,
        )
        self._eventos: asyncio.Queue = asyncio.Queue()
        self._tarea_hablar: Optional[asyncio.Task] = None
        self._interrumpible = False
        self._ultima_actividad = time.monotonic()
        self.cliente_colgo = False

    # --- API ---------------------------------------------------------------

    async def ejecutar(self) -> Optional[Resultado]:
        """Ejecuta la conversación y devuelve el resultado."""
        receptor = asyncio.create_task(self._recibir())
        try:
            await asyncio.wait_for(self._bucle(), timeout=self.s.max_duracion_s)
        except asyncio.TimeoutError:
            log.info("Duración máxima alcanzada; se cierra la llamada")
            await self._cancelar_habla()
            if not self.llamada.colgada.is_set():
                await self._decir(
                    Turno(self.dialogo.estado, texto="Le tengo que dejar ya. Muchas gracias por su tiempo."),
                    interrumpible=False,
                )
                await self._tarea_hablar  # type: ignore[misc]
            if self.dialogo.datos.resultado is None:
                self.dialogo.datos.resultado = Resultado.SIN_CITA
        finally:
            await self._cancelar_habla(vaciar=False)
            if not self.llamada.colgada.is_set():
                await self.llamada.esperar_fin_reproduccion(timeout=15)
                await asyncio.sleep(0.3)  # margen para que el último audio llegue al teléfono
            receptor.cancel()
            await asyncio.gather(receptor, return_exceptions=True)
            await self.llamada.colgar()
        return self.dialogo.datos.resultado

    # --- recepción de audio --------------------------------------------------

    async def _recibir(self) -> None:
        try:
            async for trama in self.llamada.frames():
                for evento in self.vad.procesar(trama):
                    await self._eventos.put(evento)
        finally:
            await self._eventos.put(None)  # el cliente ha colgado

    # --- bucle principal -----------------------------------------------------

    def _hablando(self) -> bool:
        return (self._tarea_hablar is not None and not self._tarea_hablar.done()) or self.llamada.reproduciendo

    async def _bucle(self) -> None:
        await self._decir(self.dialogo.inicio(), interrumpible=False)

        while True:
            if self.dialogo.terminado and (self._tarea_hablar is None or self._tarea_hablar.done()):
                return
            try:
                evento = await asyncio.wait_for(self._eventos.get(), timeout=0.2)
            except asyncio.TimeoutError:
                if self._hablando() or self.vad.hablando:
                    self._ultima_actividad = time.monotonic()
                elif time.monotonic() - self._ultima_actividad > self.silencio_max_s:
                    self._ultima_actividad = time.monotonic()
                    await self._decir(self.dialogo.silencio())
                continue

            self._ultima_actividad = time.monotonic()
            if evento is None:
                self.cliente_colgo = True
                await self._cancelar_habla()
                return
            if isinstance(evento, InicioVoz):
                if self._hablando() and self._interrumpible and not self.dialogo.terminado:
                    await self._interrumpir()
                continue
            if isinstance(evento, FinVoz):
                await self._turno_cliente(evento)

    async def _turno_cliente(self, evento: FinVoz) -> None:
        if self.dialogo.terminado:
            return  # ya se está despidiendo
        if self._hablando() and self._interrumpible:
            await self._interrumpir()
        t_fin = time.monotonic()
        texto = (await self.stt.transcribir(evento.audio, self.llamada.sample_rate)).strip()
        self.metricas.stt_ms.append(int((time.monotonic() - t_fin) * 1000))
        if not texto:
            return
        # Si el cliente habló durante el saludo (no interrumpible), se espera a que
        # termine de sonar entero antes de responder.
        if not self._interrumpible and self._hablando():
            if self._tarea_hablar is not None:
                await asyncio.gather(self._tarea_hablar, return_exceptions=True)
            await self.llamada.esperar_fin_reproduccion()
            t_fin = time.monotonic()
        estado = self.dialogo.estado.value
        self.oyente.al_escuchar(texto, estado)
        self.conversacion.cliente(texto)
        turno = self.dialogo.procesar(texto)
        for accion in turno.acciones:
            try:
                self.oyente.al_accion(accion, self.dialogo)
            except Exception:  # pragma: no cover - no debe tumbar la llamada
                log.exception("Error al registrar la acción %s", accion)
            if accion is Accion.DESACTIVAR_GRABACION:
                await self.llamada.detener_grabacion(borrar=True)
        await self._decir(turno, t_fin=t_fin)

    # --- habla ---------------------------------------------------------------

    async def _decir(self, turno: Turno, t_fin: Optional[float] = None, interrumpible: bool = True) -> None:
        await self._cancelar_habla()
        self._interrumpible = interrumpible
        self._tarea_hablar = asyncio.create_task(self._hablar(turno, t_fin))

    async def _hablar(self, turno: Turno, t_fin: Optional[float]) -> None:
        dicho: list[str] = []
        latencia: Optional[int] = None

        async def emitir(frase: str) -> None:
            nonlocal latencia
            # Solo la primera frase se trocea: es la que determina la latencia.
            for trozo in trocear_para_voz(frase) if latencia is None else [frase]:
                audio = await self.tts.sintetizar(trozo)
                if latencia is None and t_fin is not None:
                    # Latencia percibida: el silencio que esperó el VAD más el procesamiento.
                    latencia = int((time.monotonic() - t_fin) * 1000) + self.s.vad_silencio_fin_ms
                    self.metricas.latencias_ms.append(latencia)
                await self.llamada.enviar_audio(audio)
            dicho.append(frase)

        try:
            for frase in prompts.dividir_frases(turno.texto):
                await emitir(frase)
            if turno.instruccion:
                mensajes = self.conversacion.mensajes(turno.instruccion, texto_previo=turno.texto)
                async for frase in respuesta_en_frases(
                    self.llm, mensajes, turno.respaldo, timeout_primera_s=self.timeout_llm_s
                ):
                    await emitir(frase)
        finally:
            texto = " ".join(dicho)
            if texto:
                self.conversacion.agente(texto)
                self.oyente.al_hablar(texto, turno.estado.value, latencia)

    async def _interrumpir(self) -> None:
        self.metricas.interrupciones += 1
        log.debug("Interrupción del cliente: se corta el audio")
        await self._cancelar_habla()

    async def _cancelar_habla(self, vaciar: bool = True) -> None:
        tarea = self._tarea_hablar
        if tarea is not None and not tarea.done():
            tarea.cancel()
            await asyncio.gather(tarea, return_exceptions=True)
        if vaciar:
            self.llamada.vaciar_salida()
