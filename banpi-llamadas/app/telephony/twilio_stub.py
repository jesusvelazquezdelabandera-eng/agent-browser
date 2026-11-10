"""Esqueleto para cambiar a Twilio más adelante.

Cómo se implementaría (sin cambiar nada más de la aplicación):

1. `originar()` crea la llamada con la API REST de Twilio (`calls.create`) pasando
   un TwiML con `<Connect><Stream url="wss://tu-servidor/twilio"/></Connect>`.
2. Twilio abre un WebSocket (Media Streams) hacia ese servidor y envía mensajes
   `media` con audio μ-law 8 kHz en base64, en tramas de 20 ms: es el mismo formato
   que ya usamos con Asterisk, así que `ulaw_a_pcm16` y `pcm16_a_ulaw` sirven igual.
3. `LlamadaTwilio.frames()` decodifica esos mensajes; `enviar_audio()` manda
   mensajes `media` de vuelta; `vaciar_salida()` envía un mensaje `clear`, que
   descarta el audio pendiente en Twilio (interrupción).
4. El estado de la llamada (no contesta, ocupado...) llega por el `StatusCallback`.

Ojo: Twilio cobra por minuto y por uso de Media Streams, y necesita un servidor
accesible desde internet con HTTPS. Revisa también el prefijo 400 si aplica.
"""

from __future__ import annotations

from app.telephony.base import ResultadoMarcacion, Telefonia


class TwilioTelefonia(Telefonia):
    """Pendiente de implementar. Ver la explicación del módulo."""

    async def originar(self, numero: str, caller_id: str = "", timeout_s: int = 30) -> ResultadoMarcacion:
        raise NotImplementedError(
            "La telefonía con Twilio aún no está implementada; usa TELEFONIA_BACKEND=asterisk"
        )


def crear_telefonia(backend: str) -> Telefonia:
    """Crea la implementación de telefonía indicada por `TELEFONIA_BACKEND`."""
    if backend == "twilio":
        return TwilioTelefonia()
    from app.telephony.asterisk_ari import AsteriskARI

    return AsteriskARI()
