"""Arranque de la aplicación: panel web (FastAPI) y worker de campañas.

    python -m app.main            # panel en http://127.0.0.1:8000 y worker activo

El panel permite cargar contactos (CSV), crear y lanzar campañas y ver llamadas,
transcripciones, citas y la cola para comerciales humanos. Contiene datos
personales: por defecto solo escucha en 127.0.0.1; si lo expones, define
PANEL_USUARIO y PANEL_PASSWORD (autenticación HTTP básica) y ponlo detrás de HTTPS.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Iterator, Optional
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.campaigns.worker import EjecutorLlamadas, TrabajadorCampanas, marcar_no_llamar
from app.compliance import normalizar_telefono, registrar_auditoria, telefono_valido
from app.config import RAIZ_PROYECTO, get_settings
from app.db.models import Auditoria, Campana, CampanaContacto, Cita, Contacto, Llamada
from app.db.session import get_sessionmaker, init_db

log = logging.getLogger("banpi")
plantillas = Jinja2Templates(directory=str(RAIZ_PROYECTO / "app" / "web" / "templates"))
_basic = HTTPBasic(auto_error=False)


def _zona() -> ZoneInfo:
    return ZoneInfo(get_settings().zona_horaria)


def _local(dt: Optional[datetime]) -> str:
    """Formatea una fecha UTC de la base de datos en hora de Madrid."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_zona()).strftime("%d/%m/%Y %H:%M")


plantillas.env.filters["local"] = _local


def autenticar(credenciales: Optional[HTTPBasicCredentials] = Depends(_basic)) -> None:
    """Autenticación básica opcional (solo si PANEL_USUARIO y PANEL_PASSWORD están definidos)."""
    s = get_settings()
    if not s.panel_usuario:
        return
    ok = credenciales is not None and secrets.compare_digest(
        credenciales.username.encode(), s.panel_usuario.encode()
    ) and secrets.compare_digest(credenciales.password.encode(), s.panel_password.encode())
    if not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})


def db() -> Iterator[Session]:
    s = get_sessionmaker()()
    try:
        yield s
        s.commit()
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Arranque del worker
# ---------------------------------------------------------------------------


async def _arrancar_worker(app: FastAPI) -> None:
    """Carga los modelos, conecta con la telefonía y arranca el worker."""
    from app.agent.llm import crear_llm
    from app.audio.stt import WhisperSTT
    from app.audio.tts import PiperTTS
    from app.telephony.twilio_stub import crear_telefonia

    s = get_settings()
    telefonia = crear_telefonia(s.telefonia_backend)
    try:
        await telefonia.iniciar()
    except Exception as e:
        log.error("No se pudo conectar con la telefonía (%s). El panel funciona, pero no se llamará.", e)
        app.state.error_worker = f"Sin conexión con la telefonía: {e}"
        return
    llm = crear_llm(s)
    stt = WhisperSTT(s)
    tts = PiperTTS(8000, s)
    # Precarga de modelos y de las frases fijas para que la primera llamada no espere.
    fijas = [
        "¿Tiene un minuto?",
        "¿Me oye bien?",
        "Sin problema. ¿Cuándo le viene mejor que le llamemos?",
        "De acuerdo, no se preocupe, gracias por atenderme.",
        "Que tenga muy buen día.",
        "Entendido, le quito ahora mismo de nuestra lista y no volveremos a llamarle.",
        "Disculpe las molestias y que tenga buen día.",
    ]
    await asyncio.gather(llm.calentar(), stt.calentar(), tts.precalentar(fijas))
    ejecutor = EjecutorLlamadas(get_sessionmaker(), telefonia, llm, stt, tts, s)
    trabajador = TrabajadorCampanas(get_sessionmaker(), ejecutor, s)
    app.state.telefonia = telefonia
    await trabajador.bucle(app.state.parar)


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    app.state.parar = asyncio.Event()
    app.state.error_worker = ""
    tarea = None
    if get_settings().worker_activo:
        tarea = asyncio.create_task(_arrancar_worker(app))
    else:
        app.state.error_worker = "Worker desactivado (WORKER_ACTIVO=false)"
    yield
    app.state.parar.set()
    if tarea:
        try:
            await asyncio.wait_for(tarea, timeout=30)
        except (asyncio.TimeoutError, Exception):
            tarea.cancel()
    tel = getattr(app.state, "telefonia", None)
    if tel:
        await tel.detener()


app = FastAPI(title="Banpi Domótica: agente de llamadas", lifespan=ciclo_de_vida, dependencies=[Depends(autenticar)])


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------


def _estadisticas(s: Session) -> dict[str, Any]:
    por_estado = dict(s.execute(select(Contacto.estado, func.count()).group_by(Contacto.estado)).all())
    por_resultado = dict(s.execute(select(Llamada.resultado, func.count()).group_by(Llamada.resultado)).all())
    latencia = s.scalar(select(func.avg(Llamada.latencia_media_ms)))
    return {
        "contactos": sum(por_estado.values()),
        "por_estado": por_estado,
        "llamadas": sum(por_resultado.values()),
        "por_resultado": por_resultado,
        "citas": s.scalar(select(func.count()).select_from(Cita)) or 0,
        "latencia_media_ms": int(latencia) if latencia else None,
    }


@app.get("/", response_class=HTMLResponse)
def inicio(request: Request, llamada: Optional[int] = None, msg: str = "", s: Session = Depends(db)) -> HTMLResponse:
    detalle = s.get(Llamada, llamada) if llamada else None
    campanas = s.scalars(select(Campana).order_by(Campana.id.desc())).all()
    resumen_campanas = []
    for c in campanas:
        cuenta = dict(
            s.execute(
                select(CampanaContacto.estado, func.count()).where(CampanaContacto.campana_id == c.id).group_by(CampanaContacto.estado)
            ).all()
        )
        resumen_campanas.append((c, cuenta))
    return plantillas.TemplateResponse(
        request,
        "index.html",
        {
            "s": get_settings(),
            "msg": msg,
            "error_worker": getattr(request.app.state, "error_worker", ""),
            "stats": _estadisticas(s),
            "campanas": resumen_campanas,
            "contactos": s.scalars(select(Contacto).order_by(Contacto.id.desc()).limit(200)).all(),
            "llamadas": s.scalars(select(Llamada).order_by(Llamada.id.desc()).limit(50)).all(),
            "citas": s.scalars(select(Cita).order_by(Cita.id.desc()).limit(100)).all(),
            "cola_humana": s.scalars(select(Contacto).where(Contacto.estado == "cola_humana")).all(),
            "detalle": detalle,
        },
    )


def _redirigir(msg: str) -> RedirectResponse:
    from urllib.parse import quote

    return RedirectResponse(f"/?msg={quote(msg)}", status_code=303)


def _si(valor: str) -> bool:
    return (valor or "").strip().lower() in {"1", "si", "sí", "true", "x", "yes", "s"}


def _fecha(valor: str) -> Optional[datetime]:
    valor = (valor or "").strip()
    if not valor:
        return None
    for formato in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(valor, formato).replace(tzinfo=_zona()).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


@app.post("/contactos/importar")
async def importar_contactos(archivo: UploadFile = File(...), s: Session = Depends(db)) -> RedirectResponse:
    """Importa un CSV con columnas: nombre, telefono, email, consentimiento,
    consentimiento_fecha, consentimiento_origen, base_legal."""
    contenido = (await archivo.read()).decode("utf-8-sig", errors="replace")
    try:
        dialecto = csv.Sniffer().sniff(contenido.splitlines()[0], delimiters=",;\t")
    except (csv.Error, IndexError):
        dialecto = csv.excel
    filas = csv.DictReader(io.StringIO(contenido), dialect=dialecto)
    nuevos = actualizados = invalidos = 0
    for fila in filas:
        fila = {(k or "").strip().lower(): (v or "").strip() for k, v in fila.items()}
        telefono = normalizar_telefono(fila.get("telefono", ""))
        if not telefono_valido(telefono):
            invalidos += 1
            continue
        consentimiento = _si(fila.get("consentimiento", ""))
        base_legal = fila.get("base_legal", "") or ("consentimiento" if consentimiento else "")
        contacto = s.scalar(select(Contacto).where(Contacto.telefono == telefono))
        if contacto is None:
            contacto = Contacto(telefono=telefono)
            s.add(contacto)
            nuevos += 1
        else:
            actualizados += 1
        contacto.nombre = fila.get("nombre", contacto.nombre)
        contacto.email = fila.get("email", contacto.email)
        contacto.consentimiento = consentimiento
        contacto.consentimiento_fecha = _fecha(fila.get("consentimiento_fecha", ""))
        contacto.consentimiento_origen = fila.get("consentimiento_origen", "")
        contacto.base_legal = base_legal
        s.flush()
        registrar_auditoria(
            s, "contacto_importado", contacto_id=contacto.id, telefono=telefono,
            consentimiento=consentimiento, base_legal=base_legal, origen=contacto.consentimiento_origen,
        )  # fmt: skip
    return _redirigir(f"Importados {nuevos} nuevos, {actualizados} actualizados, {invalidos} con teléfono no válido.")


@app.post("/contactos/{contacto_id}/no-llamar")
def contacto_no_llamar(contacto_id: int, s: Session = Depends(db)) -> RedirectResponse:
    contacto = s.get(Contacto, contacto_id)
    if contacto is None:
        raise HTTPException(404)
    marcar_no_llamar(s, contacto, "marcado_en_panel")
    return _redirigir(f"{contacto.telefono} marcado como no contactar.")


@app.post("/campanas")
def crear_campana(nombre: str = Form(...), s: Session = Depends(db)) -> RedirectResponse:
    """Crea una campaña con todos los contactos que no estén marcados como no contactar.

    El cumplimiento (consentimiento, Robinson, horario...) se comprueba justo antes
    de cada llamada, así que aquí no se filtra más.
    """
    campana = Campana(nombre=nombre.strip() or "Campaña")
    s.add(campana)
    s.flush()
    contactos = s.scalars(select(Contacto).where(Contacto.no_llamar.is_(False))).all()
    for c in contactos:
        s.add(CampanaContacto(campana_id=campana.id, contacto_id=c.id))
    registrar_auditoria(s, "campana_creada", campana_id=campana.id, contactos=len(contactos))
    return _redirigir(f"Campaña '{campana.nombre}' creada con {len(contactos)} contactos.")


@app.post("/campanas/{campana_id}/{accion}")
def cambiar_campana(campana_id: int, accion: str, s: Session = Depends(db)) -> RedirectResponse:
    campana = s.get(Campana, campana_id)
    if campana is None or accion not in ("iniciar", "pausar"):
        raise HTTPException(404)
    campana.estado = "activa" if accion == "iniciar" else "pausada"
    registrar_auditoria(s, f"campana_{accion}", campana_id=campana_id)
    return _redirigir(f"Campaña '{campana.nombre}': {campana.estado}.")


# ---------------------------------------------------------------------------
# API y exportaciones
# ---------------------------------------------------------------------------


@app.get("/api/estado")
def api_estado(s: Session = Depends(db)) -> dict[str, Any]:
    return _estadisticas(s)


@app.get("/api/llamadas/{llamada_id}")
def api_llamada(llamada_id: int, s: Session = Depends(db)) -> dict[str, Any]:
    llamada = s.get(Llamada, llamada_id)
    if llamada is None:
        raise HTTPException(404)
    return {
        "id": llamada.id,
        "contacto": llamada.contacto.telefono,
        "resultado": llamada.resultado,
        "latencia_media_ms": llamada.latencia_media_ms,
        "datos": llamada.datos,
        "transcripcion": [
            {"rol": t.rol, "texto": t.texto, "estado": t.estado, "latencia_ms": t.latencia_ms}
            for t in llamada.transcripciones
        ],
    }


def _csv(nombre: str, cabecera: list[str], filas: list[list[Any]]) -> StreamingResponse:
    salida = io.StringIO()
    escritor = csv.writer(salida, delimiter=";")
    escritor.writerow(cabecera)
    escritor.writerows(filas)
    return StreamingResponse(
        iter(["﻿" + salida.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{nombre}"'},
    )


@app.get("/exportar/citas.csv")
def exportar_citas(s: Session = Depends(db)) -> StreamingResponse:
    filas = [
        [c.id, c.contacto.nombre, c.contacto.telefono, c.tipo, c.cuando_texto, _local(c.cuando), c.resumen, _local(c.creada)]
        for c in s.scalars(select(Cita).order_by(Cita.id))
    ]
    return _csv("citas.csv", ["id", "nombre", "telefono", "tipo", "cuando", "fecha", "resumen", "creada"], filas)


@app.get("/exportar/cola_humana.csv")
def exportar_cola_humana(s: Session = Depends(db)) -> StreamingResponse:
    filas = [
        [c.id, c.nombre, c.telefono, c.email, c.base_legal]
        for c in s.scalars(select(Contacto).where(Contacto.estado == "cola_humana"))
    ]
    return _csv("cola_humana.csv", ["id", "nombre", "telefono", "email", "base_legal"], filas)


@app.get("/exportar/auditoria.csv")
def exportar_auditoria(s: Session = Depends(db)) -> StreamingResponse:
    filas = [
        [a.id, _local(a.momento), a.evento, a.contacto_id, a.llamada_id, a.telefono, a.detalle]
        for a in s.scalars(select(Auditoria).order_by(Auditoria.id))
    ]
    return _csv("auditoria.csv", ["id", "momento", "evento", "contacto", "llamada", "telefono", "detalle"], filas)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    uvicorn.run(app, host=s.panel_host, port=s.panel_puerto)


if __name__ == "__main__":
    main()
