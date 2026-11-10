"""Cola de llamadas de las campañas, respetando horarios, intentos y consentimiento.

Cada ciclo:
1. Busca contactos pendientes de campañas activas cuyo próximo intento ya ha llegado.
2. Pasa cada uno por `ReglasCumplimiento` justo antes de llamar (las listas Robinson
   e interna se recargan en cada ciclo) y registra la decisión en la auditoría.
3. Si se permite, lanza la llamada respetando el máximo de llamadas simultáneas.
4. Al terminar, guarda el resultado y programa el siguiente intento si hace falta.

Un intento cuenta desde que se marca el número (conteste o no), para el límite de
3 intentos en 7 días.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agent.dialog import Accion, DatosLlamada, Dialogo, Resultado
from app.compliance import Motivo, ReglasCumplimiento, normalizar_telefono, registrar_auditoria
from app.config import Settings, get_settings
from app.db.models import (
    Campana,
    CampanaContacto,
    Cita,
    Contacto,
    ListaInterna,
    Llamada,
    Transcripcion,
)
from app.db.session import como_utc

log = logging.getLogger(__name__)

# Resultados de un intento en el que no se llegó a hablar con el cliente.
SIN_CONTACTO = {"no_contesta", "ocupado", "fallo", "rechazada", "sin_respuesta"}

Ejecutor = Callable[[int], Awaitable[None]]


def ahora_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Acciones y resultados sobre la base de datos
# ---------------------------------------------------------------------------


def marcar_no_llamar(s: Session, contacto: Contacto, motivo: str, llamada_id: Optional[int] = None) -> None:
    """Marca un contacto como 'no contactar' y lo añade a la lista interna."""
    contacto.no_llamar = True
    contacto.estado = "no_llamar"
    telefono = normalizar_telefono(contacto.telefono)
    if s.scalar(select(ListaInterna).where(ListaInterna.telefono == telefono)) is None:
        s.add(ListaInterna(telefono=telefono, motivo=motivo))
    for cc in s.scalars(select(CampanaContacto).where(CampanaContacto.contacto_id == contacto.id)):
        if cc.estado in ("pendiente", "reintentar"):
            cc.estado, cc.motivo = "excluido", "no_llamar"
    registrar_auditoria(
        s, "no_llamar", contacto_id=contacto.id, llamada_id=llamada_id, telefono=contacto.telefono, motivo=motivo
    )


def aplicar_resultado(
    s: Session,
    cc: CampanaContacto,
    llamada: Llamada,
    resultado: str,
    datos: Optional[DatosLlamada],
    settings: Settings,
    ahora: datetime,
) -> None:
    """Actualiza contacto, campaña y llamada según el resultado del intento."""
    contacto = cc.contacto
    llamada.resultado = resultado
    llamada.fin = ahora
    cc.motivo = resultado

    if resultado in SIN_CONTACTO:
        fallidos = s.scalars(
            select(Llamada.resultado).where(
                Llamada.contacto_id == contacto.id, Llamada.campana_id == cc.campana_id
            )
        ).all()
        if sum(1 for r in fallidos if r in SIN_CONTACTO) >= settings.max_intentos:
            cc.estado = "completado"
            cc.motivo = "sin_contacto_tras_intentos"
        else:
            cc.estado = "reintentar"
            cc.proximo_intento = ahora + timedelta(minutes=settings.reintento_minutos)
    elif resultado == Resultado.REPROGRAMAR.value:
        contacto.estado = "contactado"
        cc.estado = "reintentar"
        momento = datos.rellamar.momento if datos and datos.rellamar else None
        cc.proximo_intento = como_utc(momento) if momento else ahora + timedelta(days=1)
        llamada.rellamar_en = cc.proximo_intento
        llamada.rellamar_texto = datos.rellamar_texto if datos else ""
    elif resultado == Resultado.CITA.value:
        contacto.estado = "cita"
        cc.estado = "completado"
    elif resultado == Resultado.NO_INTERESA.value:
        contacto.estado = "no_interesa"
        cc.estado = "completado"
    elif resultado in (Resultado.NO_LLAMAR.value, Resultado.EQUIVOCADO.value):
        if not contacto.no_llamar:
            marcar_no_llamar(s, contacto, resultado, llamada.id)
        cc.estado = "completado"
    else:  # sin_cita, colgada...
        contacto.estado = "contactado"
        cc.estado = "completado"

    if datos is not None:
        llamada.estado_final = resultado
        llamada.datos = {
            "inmueble": datos.inmueble,
            "intereses": datos.intereses,
            "preocupaciones": datos.preocupaciones,
            "cita_tipo": datos.cita_tipo,
            "cita_texto": datos.cita_texto,
        }
    registrar_auditoria(
        s,
        "llamada_finalizada",
        contacto_id=contacto.id,
        llamada_id=llamada.id,
        telefono=contacto.telefono,
        resultado=resultado,
        proximo_intento=cc.proximo_intento if cc.estado == "reintentar" else None,
    )


class OyenteBD:
    """Guarda la conversación y aplica las acciones en la base de datos al momento."""

    def __init__(self, fabrica: sessionmaker[Session], llamada_id: int, contacto_id: int) -> None:
        self.fabrica = fabrica
        self.llamada_id = llamada_id
        self.contacto_id = contacto_id

    def _guardar(self, rol: str, texto: str, estado: str, latencia: Optional[int] = None) -> None:
        with self.fabrica.begin() as s:
            s.add(Transcripcion(llamada_id=self.llamada_id, rol=rol, texto=texto, estado=estado, latencia_ms=latencia))

    def al_hablar(self, texto: str, estado: str, latencia_ms: Optional[int]) -> None:
        self._guardar("agente", texto, estado, latencia_ms)

    def al_escuchar(self, texto: str, estado: str) -> None:
        self._guardar("cliente", texto, estado)

    def al_accion(self, accion: Accion, dialogo: Dialogo) -> None:
        with self.fabrica.begin() as s:
            contacto = s.get(Contacto, self.contacto_id)
            llamada = s.get(Llamada, self.llamada_id)
            assert contacto is not None and llamada is not None
            if accion is Accion.MARCAR_NO_LLAMAR:
                marcar_no_llamar(s, contacto, "pedido_en_llamada", self.llamada_id)
            elif accion is Accion.DESACTIVAR_GRABACION:
                llamada.grabacion_permitida = False
                llamada.grabacion_ruta = ""
                registrar_auditoria(
                    s, "oposicion_grabacion", contacto_id=contacto.id, llamada_id=llamada.id,
                    telefono=contacto.telefono,
                )  # fmt: skip
            elif accion is Accion.GUARDAR_CITA:
                d = dialogo.datos
                s.add(
                    Cita(
                        contacto_id=contacto.id,
                        llamada_id=llamada.id,
                        tipo=d.cita_tipo or "visita",
                        cuando_texto=d.cita_texto,
                        cuando=como_utc(d.cita.momento) if d.cita and d.cita.momento else None,
                        resumen=d.resumen(),
                    )
                )
                registrar_auditoria(
                    s, "cita", contacto_id=contacto.id, llamada_id=llamada.id, telefono=contacto.telefono,
                    tipo=d.cita_tipo, cuando=d.cita_texto,
                )  # fmt: skip


# ---------------------------------------------------------------------------
# Trabajador
# ---------------------------------------------------------------------------


class TrabajadorCampanas:
    """Selecciona a quién llamar y lanza las llamadas."""

    def __init__(
        self,
        fabrica: sessionmaker[Session],
        ejecutor: Ejecutor,
        settings: Optional[Settings] = None,
        reloj: Callable[[], datetime] = ahora_utc,
    ) -> None:
        self.fabrica = fabrica
        self.ejecutor = ejecutor
        self.s = settings or get_settings()
        self.reloj = reloj
        self._sem = asyncio.Semaphore(self.s.llamadas_simultaneas)
        self._tareas: set[asyncio.Task] = set()
        self._contactos_en_curso: set[int] = set()

    def recuperar_tras_reinicio(self) -> None:
        """Las llamadas que quedaron 'en curso' por un cierre inesperado se reintentan."""
        with self.fabrica.begin() as s:
            for cc in s.scalars(select(CampanaContacto).where(CampanaContacto.estado == "en_curso")):
                cc.estado = "reintentar"

    def _reglas(self, s: Session) -> ReglasCumplimiento:
        internos = s.scalars(select(ListaInterna.telefono)).all()
        return ReglasCumplimiento.desde_settings(self.s, internos)

    async def ciclo(self) -> int:
        """Revisa la cola una vez. Devuelve cuántas llamadas ha lanzado."""
        ahora = self.reloj()
        libres = self.s.llamadas_simultaneas - len(self._contactos_en_curso)
        if libres <= 0:
            return 0
        lanzar: list[int] = []
        with self.fabrica.begin() as s:
            reglas = self._reglas(s)
            candidatos = s.scalars(
                select(CampanaContacto)
                .join(Campana)
                .where(Campana.estado == "activa", CampanaContacto.estado.in_(("pendiente", "reintentar")))
                .order_by(CampanaContacto.proximo_intento.is_not(None), CampanaContacto.proximo_intento, CampanaContacto.id)
            ).all()
            for cc in candidatos:
                if len(lanzar) >= libres:
                    break
                proximo = como_utc(cc.proximo_intento)
                if proximo and proximo > ahora:
                    continue
                if cc.contacto_id in self._contactos_en_curso:
                    continue
                contacto = cc.contacto
                intentos = [como_utc(i) for i in s.scalars(select(Llamada.inicio).where(Llamada.contacto_id == contacto.id))]
                decision = reglas.comprobar(contacto, intentos, ahora)  # type: ignore[arg-type]
                registrar_auditoria(
                    s, "decision_cumplimiento", contacto_id=contacto.id, telefono=contacto.telefono,
                    permitido=decision.permitido, motivo=decision.motivo, detalle=decision.detalle,
                )  # fmt: skip
                if decision.permitido:
                    cc.estado = "en_curso"
                    lanzar.append(cc.id)
                    self._contactos_en_curso.add(contacto.id)
                elif decision.motivo is Motivo.REQUIERE_HUMANO:
                    cc.estado, cc.motivo = "cola_humana", decision.motivo.value
                    contacto.estado = "cola_humana"
                elif decision.permanente:
                    cc.estado, cc.motivo = "excluido", decision.motivo.value
                else:
                    cc.estado, cc.motivo = "reintentar", decision.motivo.value
                    cc.proximo_intento = decision.reintentar_desde
            self._cerrar_campanas_terminadas(s)

        ids_contacto = {}
        with self.fabrica() as s:
            for cc_id in lanzar:
                ids_contacto[cc_id] = s.get(CampanaContacto, cc_id).contacto_id  # type: ignore[union-attr]
        for cc_id in lanzar:
            tarea = asyncio.create_task(self._lanzar(cc_id, ids_contacto[cc_id]))
            self._tareas.add(tarea)
            tarea.add_done_callback(self._tareas.discard)
        return len(lanzar)

    async def _lanzar(self, cc_id: int, contacto_id: int) -> None:
        async with self._sem:
            try:
                await self.ejecutor(cc_id)
            except Exception:
                log.exception("Error en la llamada del contacto %s", contacto_id)
                with self.fabrica.begin() as s:
                    cc = s.get(CampanaContacto, cc_id)
                    if cc and cc.estado == "en_curso":
                        cc.estado = "reintentar"
                        cc.proximo_intento = self.reloj() + timedelta(minutes=self.s.reintento_minutos)
            finally:
                self._contactos_en_curso.discard(contacto_id)

    def _cerrar_campanas_terminadas(self, s: Session) -> None:
        for campana in s.scalars(select(Campana).where(Campana.estado == "activa")):
            abiertos = [cc for cc in campana.contactos if cc.estado in ("pendiente", "reintentar", "en_curso")]
            if campana.contactos and not abiertos:
                campana.estado = "terminada"

    async def esperar_llamadas(self) -> None:
        if self._tareas:
            await asyncio.gather(*self._tareas, return_exceptions=True)

    async def bucle(self, parar: asyncio.Event) -> None:
        """Bucle principal: un ciclo cada `worker_intervalo_s` segundos."""
        self.recuperar_tras_reinicio()
        while not parar.is_set():
            try:
                await self.ciclo()
            except Exception:
                log.exception("Error en el ciclo de campañas")
            try:
                await asyncio.wait_for(parar.wait(), timeout=self.s.worker_intervalo_s)
            except asyncio.TimeoutError:
                pass
        await self.esperar_llamadas()


# ---------------------------------------------------------------------------
# Ejecución real de una llamada
# ---------------------------------------------------------------------------


class EjecutorLlamadas:
    """Marca, conversa y guarda el resultado de una llamada de campaña."""

    def __init__(self, fabrica: sessionmaker[Session], telefonia: Any, llm: Any, stt: Any, tts: Any,
                 settings: Optional[Settings] = None) -> None:  # fmt: skip
        self.fabrica = fabrica
        self.telefonia = telefonia
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.s = settings or get_settings()

    async def __call__(self, cc_id: int) -> None:
        from app.agent.pipeline import SesionLlamada

        with self.fabrica.begin() as s:
            cc = s.get(CampanaContacto, cc_id)
            assert cc is not None
            contacto = cc.contacto
            llamada = Llamada(contacto_id=contacto.id, campana_id=cc.campana_id, inicio=ahora_utc())
            s.add(llamada)
            s.flush()
            llamada_id, contacto_id = llamada.id, contacto.id
            telefono, nombre = contacto.telefono, contacto.nombre
            registrar_auditoria(s, "marcacion", contacto_id=contacto_id, llamada_id=llamada_id, telefono=telefono)

        marcacion = await self.telefonia.originar(telefono, self.s.caller_id, self.s.timeout_respuesta_s)
        if marcacion.llamada is None:
            with self.fabrica.begin() as s:
                cc = s.get(CampanaContacto, cc_id)
                llamada = s.get(Llamada, llamada_id)
                assert cc is not None and llamada is not None
                llamada.error = marcacion.detalle
                aplicar_resultado(s, cc, llamada, marcacion.estado.value, None, self.s, ahora_utc())
            return

        tel = marcacion.llamada
        ruta_grabacion = ""
        if self.s.grabar_llamadas:
            ruta_grabacion = await tel.iniciar_grabacion(f"banpi-{llamada_id}") or ""
        with self.fabrica.begin() as s:
            fila = s.get(Llamada, llamada_id)
            assert fila is not None
            fila.grabacion_permitida = bool(ruta_grabacion)
            fila.grabacion_ruta = ruta_grabacion

        dialogo = Dialogo(nombre, settings=self.s)
        sesion = SesionLlamada(tel, dialogo, self.llm, self.stt, self.tts, self.s,
                               OyenteBD(self.fabrica, llamada_id, contacto_id))  # fmt: skip
        resultado = await sesion.ejecutar()
        texto_resultado = resultado.value if resultado else "colgada"

        with self.fabrica.begin() as s:
            cc = s.get(CampanaContacto, cc_id)
            fila = s.get(Llamada, llamada_id)
            assert cc is not None and fila is not None
            fila.latencia_media_ms = sesion.metricas.latencia_media_ms
            aplicar_resultado(s, cc, fila, texto_resultado, dialogo.datos, self.s, ahora_utc())
