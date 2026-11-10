"""Telefonía con Asterisk mediante ARI (Asterisk REST Interface) y External Media.

Flujo de una llamada saliente:

1. `POST /channels` origina la llamada al troncal SIP (`PJSIP/<número>@troncal`)
   y la manda a nuestra aplicación Stasis cuando el cliente contesta.
2. Llega `StasisStart` (contestada) o `ChannelDestroyed` (no contesta, ocupado...).
3. Se crea un canal External Media (`POST /channels/externalMedia`, μ-law) que
   envía el audio por RTP a nuestro `EndpointRTP`, y un puente (bridge) de mezcla
   con el canal del cliente y el de External Media.
4. Durante la llamada, el audio entra y sale por RTP. Si el cliente cuelga llega
   `StasisEnd`/`ChannelDestroyed` y se marca la llamada como colgada.
5. Al terminar se borran los canales y el puente.

La grabación (si el cliente no se opone) se hace en el propio Asterisk sobre el
puente: `POST /bridges/{id}/record`. Los archivos quedan en
`/var/spool/asterisk/recording/` del servidor de Asterisk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, Optional
from urllib.parse import quote

import httpx

from app.compliance import normalizar_telefono
from app.config import Settings, get_settings
from app.telephony.base import EstadoMarcacion, Llamada, ResultadoMarcacion, Telefonia
from app.telephony.rtp import SAMPLE_RATE, AsignadorPuertos, EndpointRTP

log = logging.getLogger(__name__)

# Códigos de causa Q.850 más habituales al no completarse una llamada.
CAUSAS = {
    17: EstadoMarcacion.OCUPADO,  # User busy
    18: EstadoMarcacion.NO_CONTESTA,  # No user responding
    19: EstadoMarcacion.NO_CONTESTA,  # No answer
    16: EstadoMarcacion.NO_CONTESTA,  # Normal clearing antes de contestar
    21: EstadoMarcacion.RECHAZADA,  # Call rejected
    1: EstadoMarcacion.FALLO,  # Unallocated number
    34: EstadoMarcacion.FALLO,  # No circuit available
    38: EstadoMarcacion.FALLO,  # Network out of order
}


def formatear_numero(numero: str, formato: str) -> str:
    """Adapta el número E.164 al formato que espera el proveedor SIP."""
    e164 = normalizar_telefono(numero)
    if formato == "e164":
        return e164
    if formato == "internacional":
        return "00" + e164[1:]
    if formato == "sin_mas":
        return e164[1:]
    # "nacional": los números españoles sin prefijo; el resto, internacional.
    if e164.startswith("+34"):
        return e164[3:]
    return "00" + e164[1:]


def causa_a_estado(causa: Optional[int]) -> EstadoMarcacion:
    return CAUSAS.get(causa or 0, EstadoMarcacion.FALLO)


class AsteriskARI(Telefonia):
    """Cliente ARI: REST para las órdenes y WebSocket para los eventos."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.s = settings or get_settings()
        self.http = httpx.AsyncClient(
            base_url=self.s.ari_url.rstrip("/") + "/ari",
            auth=(self.s.ari_usuario, self.s.ari_password),
            timeout=10.0,
        )
        self.puertos = AsignadorPuertos(self.s.rtp_puerto_inicio, self.s.rtp_puerto_fin)
        self._colas: dict[str, asyncio.Queue[dict]] = {}
        self._tarea_eventos: Optional[asyncio.Task] = None
        self._conectado = asyncio.Event()

    # --- conexión y eventos -----------------------------------------------

    def _url_eventos(self) -> str:
        base = self.s.ari_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
        clave = quote(f"{self.s.ari_usuario}:{self.s.ari_password}", safe="")
        return f"{base}/ari/events?app={quote(self.s.ari_app)}&api_key={clave}"

    async def iniciar(self) -> None:
        if self._tarea_eventos is None:
            self._tarea_eventos = asyncio.create_task(self._escuchar_eventos())
        await asyncio.wait_for(self._conectado.wait(), timeout=15)
        info = (await self.http.get("/asterisk/info")).json()
        log.info("Conectado a Asterisk %s", info.get("system", {}).get("version", "?"))

    async def detener(self) -> None:
        if self._tarea_eventos:
            self._tarea_eventos.cancel()
        await self.http.aclose()

    async def _escuchar_eventos(self) -> None:
        import websockets

        espera = 1.0
        while True:
            try:
                async with websockets.connect(self._url_eventos(), ping_interval=20) as ws:
                    self._conectado.set()
                    espera = 1.0
                    async for mensaje in ws:
                        self._despachar(json.loads(mensaje))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # reconexión automática
                self._conectado.clear()
                log.warning("WebSocket de ARI desconectado (%s); reintento en %.0f s", e, espera)
                await asyncio.sleep(espera)
                espera = min(espera * 2, 30)

    def _despachar(self, evento: dict) -> None:
        canal = (evento.get("channel") or {}).get("id")
        if canal and canal in self._colas:
            self._colas[canal].put_nowait(evento)

    # --- originar ------------------------------------------------------------

    async def originar(self, numero: str, caller_id: str = "", timeout_s: int = 30) -> ResultadoMarcacion:
        canal_id = f"banpi-{uuid.uuid4().hex[:12]}"
        cola: asyncio.Queue[dict] = asyncio.Queue()
        self._colas[canal_id] = cola
        destino = formatear_numero(numero, self.s.sip_formato_numero)
        params = {
            "endpoint": self.s.sip_endpoint.format(numero=destino),
            "app": self.s.ari_app,
            "appArgs": "saliente",
            "timeout": str(timeout_s),
            "channelId": canal_id,
        }
        if caller_id or self.s.caller_id:
            params["callerId"] = caller_id or self.s.caller_id
        try:
            r = await self.http.post("/channels", params=params)
            r.raise_for_status()
        except httpx.HTTPError as e:
            self._colas.pop(canal_id, None)
            return ResultadoMarcacion(EstadoMarcacion.FALLO, detalle=f"Error al originar: {e}")

        try:
            while True:
                evento = await asyncio.wait_for(cola.get(), timeout=timeout_s + 15)
                tipo = evento.get("type")
                if tipo == "StasisStart":
                    break
                if tipo in ("ChannelDestroyed", "ChannelHangupRequest"):
                    self._colas.pop(canal_id, None)
                    estado = causa_a_estado(evento.get("cause"))
                    return ResultadoMarcacion(estado, detalle=evento.get("cause_txt", ""))
        except asyncio.TimeoutError:
            self._colas.pop(canal_id, None)
            await self._borrar(f"/channels/{canal_id}")
            return ResultadoMarcacion(EstadoMarcacion.NO_CONTESTA, detalle="Tiempo de espera agotado")

        try:
            llamada = await self._conectar_audio(canal_id, cola)
        except Exception as e:
            log.exception("No se pudo conectar el audio")
            self._colas.pop(canal_id, None)
            await self._borrar(f"/channels/{canal_id}")
            return ResultadoMarcacion(EstadoMarcacion.FALLO, detalle=f"Error de audio: {e}")
        return ResultadoMarcacion(EstadoMarcacion.CONTESTADA, llamada=llamada)

    async def _conectar_audio(self, canal_id: str, cola: asyncio.Queue[dict]) -> "LlamadaAsterisk":
        puerto = await self.puertos.tomar()
        rtp = await EndpointRTP.crear(self.s.rtp_host_local, puerto)
        media_id = f"{canal_id}-media"
        puente_id = f"{canal_id}-puente"
        r = await self.http.post(
            "/channels/externalMedia",
            params={
                "app": self.s.ari_app,
                "channelId": media_id,
                "external_host": f"{self.s.rtp_host_local}:{puerto}",
                "format": "ulaw",
                "encapsulation": "rtp",
                "transport": "udp",
                "direction": "both",
            },
        )
        r.raise_for_status()
        (await self.http.post("/bridges", params={"type": "mixing", "bridgeId": puente_id})).raise_for_status()
        (
            await self.http.post(f"/bridges/{puente_id}/addChannel", params={"channel": f"{canal_id},{media_id}"})
        ).raise_for_status()
        llamada = LlamadaAsterisk(self, canal_id, media_id, puente_id, rtp, puerto)
        asyncio.create_task(llamada._vigilar(cola))
        return llamada

    async def _borrar(self, ruta: str) -> None:
        try:
            await self.http.delete(ruta)
        except httpx.HTTPError:
            pass


class LlamadaAsterisk(Llamada):
    """Llamada contestada a través de Asterisk."""

    sample_rate = SAMPLE_RATE

    def __init__(
        self, ari: AsteriskARI, canal_id: str, media_id: str, puente_id: str, rtp: EndpointRTP, puerto: int
    ) -> None:
        super().__init__()
        self.ari = ari
        self.id = canal_id
        self.media_id = media_id
        self.puente_id = puente_id
        self.rtp = rtp
        self.puerto = puerto
        self.grabacion: Optional[str] = None
        self._cerrada = False

    async def _vigilar(self, cola: asyncio.Queue[dict]) -> None:
        """Marca la llamada como colgada cuando Asterisk avisa de ello."""
        while not self.colgada.is_set():
            evento = await cola.get()
            if evento.get("type") in ("StasisEnd", "ChannelDestroyed", "ChannelHangupRequest"):
                self.colgada.set()

    async def frames(self) -> AsyncIterator[bytes]:
        """Audio del cliente. Rellena con silencio cuando no llegan paquetes.

        Asterisk (y algunos proveedores con supresión de silencio) no envía RTP
        mientras el otro extremo calla. Sin esas tramas el VAD no podría medir el
        silencio que marca el fin del turno, así que se generan aquí.
        """
        silencio_trama = b"\x00\x00" * 160
        while not self.colgada.is_set():
            try:
                yield await asyncio.wait_for(self.rtp.entrada.get(), timeout=0.06)
            except asyncio.TimeoutError:
                for _ in range(3):  # 60 ms de silencio
                    yield silencio_trama

    async def enviar_audio(self, pcm: bytes) -> None:
        self.rtp.salida.escribir(pcm)

    def vaciar_salida(self) -> None:
        self.rtp.salida.vaciar()

    @property
    def reproduciendo(self) -> bool:
        return self.rtp.salida.pendiente > 0

    async def iniciar_grabacion(self, nombre: str) -> Optional[str]:
        r = await self.ari.http.post(
            f"/bridges/{self.puente_id}/record",
            params={"name": nombre, "format": "wav", "ifExists": "overwrite"},
        )
        if r.is_success:
            self.grabacion = nombre
            return f"/var/spool/asterisk/recording/{nombre}.wav"
        log.warning("No se pudo iniciar la grabación: %s", r.text)
        return None

    async def detener_grabacion(self, borrar: bool = False) -> None:
        if not self.grabacion:
            return
        nombre, self.grabacion = self.grabacion, None
        await self.ari.http.post(f"/recordings/live/{nombre}/stop")
        if borrar:
            await self.ari._borrar(f"/recordings/stored/{nombre}")

    async def colgar(self) -> None:
        if self._cerrada:
            return
        self._cerrada = True
        if self.grabacion:
            await self.detener_grabacion()
        self.colgada.set()
        for ruta in (f"/channels/{self.id}", f"/channels/{self.media_id}", f"/bridges/{self.puente_id}"):
            await self.ari._borrar(ruta)
        self.rtp.cerrar()
        self.ari._colas.pop(self.id, None)
        await self.ari.puertos.devolver(self.puerto)
