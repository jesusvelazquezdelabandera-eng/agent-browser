"""Pruebas de RTP y de la integración con ARI usando un Asterisk simulado."""

from __future__ import annotations

import asyncio
import socket

import httpx
import numpy as np
import pytest

from app.audio.resample import pcm16_a_ulaw
from app.config import Settings
from app.telephony.asterisk_ari import AsteriskARI, LlamadaAsterisk, causa_a_estado, formatear_numero
from app.telephony.base import BufferSalida, EstadoMarcacion
from app.telephony.rtp import PT_ULAW, EndpointRTP, PaqueteRTP


def test_paquete_rtp_ida_y_vuelta() -> None:
    p = PaqueteRTP(PT_ULAW, 65535, 123456, 0xDEADBEEF, b"\xff" * 160, marcador=True)
    q = PaqueteRTP.desde_bytes(p.a_bytes())
    assert q == p


def test_paquete_rtp_con_csrc_y_extension() -> None:
    cabecera = bytes([0x91, 0x00]) + (1).to_bytes(2, "big") + (2).to_bytes(4, "big") + (3).to_bytes(4, "big")
    csrc = (7).to_bytes(4, "big")
    extension = b"\xbe\xde" + (1).to_bytes(2, "big") + b"\x00" * 4
    q = PaqueteRTP.desde_bytes(cabecera + csrc + extension + b"abc")
    assert q is not None and q.payload == b"abc"


def test_paquete_invalido() -> None:
    assert PaqueteRTP.desde_bytes(b"\x00" * 5) is None
    assert PaqueteRTP.desde_bytes(b"\x40" + b"\x00" * 20) is None  # versión 1


@pytest.mark.parametrize(
    "formato,esperado",
    [("nacional", "600112233"), ("e164", "+34600112233"), ("internacional", "0034600112233"), ("sin_mas", "34600112233")],
)
def test_formatear_numero(formato: str, esperado: str) -> None:
    assert formatear_numero("600 11 22 33", formato) == esperado


def test_causas() -> None:
    assert causa_a_estado(17) is EstadoMarcacion.OCUPADO
    assert causa_a_estado(19) is EstadoMarcacion.NO_CONTESTA
    assert causa_a_estado(999) is EstadoMarcacion.FALLO


def test_buffer_salida() -> None:
    b = BufferSalida()
    b.escribir(b"\x01\x02\x03")
    assert b.leer(4) == b"\x01\x02\x03\x00"
    b.escribir(b"\x01" * 10)
    assert b.vaciar() == 10 and b.pendiente == 0


async def _rtp_loopback() -> None:
    puerto = _puerto_libre()
    rtp = await EndpointRTP.crear("127.0.0.1", puerto)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    loop = asyncio.get_running_loop()
    try:
        # "Asterisk" envía audio del cliente.
        pcm = (3000 * np.sin(np.arange(160) / 5)).astype(np.int16).tobytes()
        paquete = PaqueteRTP(PT_ULAW, 1, 160, 42, pcm16_a_ulaw(pcm))
        sock.sendto(paquete.a_bytes(), ("127.0.0.1", puerto))
        recibido = await asyncio.wait_for(rtp.entrada.get(), timeout=2)
        assert len(recibido) == 320
        assert rtp.remoto == sock.getsockname()

        # El agente habla: deben llegar paquetes RTP μ-law de 160 bytes cada 20 ms.
        rtp.salida.escribir(b"\x10\x00" * 1600)  # 200 ms
        paquetes = []
        while len(paquetes) < 5:
            datos = await asyncio.wait_for(loop.sock_recv(sock, 2048), timeout=2)
            paquetes.append(PaqueteRTP.desde_bytes(datos))
        assert all(p is not None and p.payload_type == PT_ULAW and len(p.payload) == 160 for p in paquetes)
        secuencias = [p.secuencia for p in paquetes]  # type: ignore[union-attr]
        assert all((b - a) & 0xFFFF == 1 for a, b in zip(secuencias, secuencias[1:]))
    finally:
        rtp.cerrar()
        sock.close()


def test_rtp_loopback() -> None:
    asyncio.run(_rtp_loopback())


def _puerto_libre() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    puerto = s.getsockname()[1]
    s.close()
    return puerto


class AsteriskFalso:
    """Simula la API REST de ARI y registra las peticiones."""

    def __init__(self) -> None:
        self.peticiones: list[tuple[str, str, dict]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.peticiones.append((req.method, req.url.path, dict(req.url.params)))
        return httpx.Response(200, json={})


def _ari(falso: AsteriskFalso) -> AsteriskARI:
    puerto = _puerto_libre()
    s = Settings(rtp_puerto_inicio=puerto, rtp_puerto_fin=puerto + 2, caller_id="954000000")
    ari = AsteriskARI(s)
    ari.http = httpx.AsyncClient(transport=httpx.MockTransport(falso), base_url="http://ari/ari")
    return ari


async def _llamada_contestada() -> None:
    falso = AsteriskFalso()
    ari = _ari(falso)
    tarea = asyncio.create_task(ari.originar("+34600112233", timeout_s=5))
    await asyncio.sleep(0.05)
    canal = next(p[2]["channelId"] for p in falso.peticiones if p[1] == "/ari/channels")
    ari._despachar({"type": "StasisStart", "channel": {"id": canal}})
    resultado = await asyncio.wait_for(tarea, timeout=3)

    assert resultado.estado is EstadoMarcacion.CONTESTADA
    llamada = resultado.llamada
    assert isinstance(llamada, LlamadaAsterisk)
    rutas = [p[1] for p in falso.peticiones]
    assert "/ari/channels/externalMedia" in rutas and "/ari/bridges" in rutas
    origen = next(p[2] for p in falso.peticiones if p[1] == "/ari/channels")
    assert origen["endpoint"] == "PJSIP/600112233@troncal" and origen["callerId"] == "954000000"
    media = next(p[2] for p in falso.peticiones if p[1] == "/ari/channels/externalMedia")
    assert media["format"] == "ulaw"

    # El cliente cuelga.
    ari._despachar({"type": "ChannelDestroyed", "channel": {"id": canal}, "cause": 16})
    await asyncio.wait_for(llamada.colgada.wait(), timeout=2)
    await llamada.colgar()
    assert ("DELETE", f"/ari/bridges/{canal}-puente", {}) in falso.peticiones


def test_ari_llamada_contestada() -> None:
    asyncio.run(_llamada_contestada())


async def _llamada_ocupado() -> None:
    falso = AsteriskFalso()
    ari = _ari(falso)
    tarea = asyncio.create_task(ari.originar("600112233", timeout_s=5))
    await asyncio.sleep(0.05)
    canal = next(p[2]["channelId"] for p in falso.peticiones if p[1] == "/ari/channels")
    ari._despachar({"type": "ChannelDestroyed", "channel": {"id": canal}, "cause": 17, "cause_txt": "User busy"})
    resultado = await asyncio.wait_for(tarea, timeout=3)
    assert resultado.estado is EstadoMarcacion.OCUPADO and resultado.llamada is None


def test_ari_ocupado() -> None:
    asyncio.run(_llamada_ocupado())


def test_url_eventos_escapa_credenciales() -> None:
    ari = AsteriskARI(Settings(ari_url="http://127.0.0.1:8088", ari_usuario="u", ari_password="a&b"))
    assert ari._url_eventos() == "ws://127.0.0.1:8088/ari/events?app=banpi&api_key=u%3Aa%26b"
