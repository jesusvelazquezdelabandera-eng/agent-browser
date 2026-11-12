"""Máquina de estados de la conversación de Álex.

Flujo principal:

    SALUDO -> CONFIRMAR_MOMENTO -> DESCUBRIMIENTO -> PROPUESTA -> CIERRE -> FIN

con ramas a REPROGRAMAR (no es buen momento) y salidas directas a FIN cuando el
cliente no está interesado o pide que no le llamen más.

Las intenciones críticas (no llamar más, no interesa, oposición a la grabación,
mal momento) se detectan con reglas deterministas, nunca con el LLM, para que el
cumplimiento no dependa de lo que genere el modelo. Cada turno devuelve un
`Turno` con el texto fijo que hay que decir y/o la instrucción para que el LLM
redacte la respuesta.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from app.agent import prompts
from app.agent.fechas import FechaHablada, describir, extraer_fecha_hora
from app.compliance import aviso_inicial, se_opone_a_grabacion
from app.config import Settings, get_settings


class Estado(str, Enum):
    SALUDO = "saludo"
    CONFIRMAR_MOMENTO = "confirmar_momento"
    DESCUBRIMIENTO = "descubrimiento"
    PROPUESTA = "propuesta"
    CIERRE = "cierre"
    REPROGRAMAR = "reprogramar"
    FIN = "fin"


class Intencion(str, Enum):
    NO_LLAMAR = "no_llamar"
    NO_INTERESA = "no_interesa"
    EQUIVOCADO = "equivocado"
    MAL_MOMENTO = "mal_momento"
    OPONE_GRABACION = "opone_grabacion"
    PREGUNTA_IA = "pregunta_ia"
    PREGUNTA_PRECIO = "pregunta_precio"
    ORIGEN_DATOS = "origen_datos"
    PIDE_HUMANO = "pide_humano"
    AFIRMATIVO = "afirmativo"
    NEGATIVO = "negativo"
    OTRA = "otra"


class Accion(str, Enum):
    MARCAR_NO_LLAMAR = "marcar_no_llamar"
    DESACTIVAR_GRABACION = "desactivar_grabacion"
    GUARDAR_CITA = "guardar_cita"
    REPROGRAMAR = "reprogramar"


class Resultado(str, Enum):
    CITA = "cita"
    REPROGRAMAR = "reprogramar"
    NO_INTERESA = "no_interesa"
    NO_LLAMAR = "no_llamar"
    EQUIVOCADO = "equivocado"
    SIN_CITA = "sin_cita"
    SIN_RESPUESTA = "sin_respuesta"


@dataclass
class Turno:
    """Respuesta del agente para un turno.

    Se dice primero `texto` (fijo) y después, si hay `instruccion`, lo que genere
    el LLM. `respaldo` es lo que se dice si el LLM falla.
    """

    estado: Estado
    texto: str = ""
    instruccion: Optional[str] = None
    respaldo: str = ""
    terminar: bool = False
    acciones: list[Accion] = field(default_factory=list)
    intencion: Intencion = Intencion.OTRA


@dataclass
class DatosLlamada:
    """Información recogida durante la conversación."""

    inmueble: str = ""  # "vivienda" o "local"
    intereses: list[str] = field(default_factory=list)
    preocupaciones: list[str] = field(default_factory=list)
    cita_tipo: str = ""  # "visita" o "llamada"
    cita: Optional[FechaHablada] = None
    cita_texto: str = ""
    rellamar: Optional[FechaHablada] = None
    rellamar_texto: str = ""
    grabacion_permitida: bool = True
    resultado: Optional[Resultado] = None

    def resumen(self) -> str:
        partes = []
        if self.inmueble:
            partes.append(f"tiene un/a {self.inmueble}")
        if self.intereses:
            partes.append("le interesa: " + ", ".join(self.intereses))
        if self.preocupaciones:
            partes.append("le preocupa: " + ", ".join(self.preocupaciones))
        return "; ".join(partes) or "todavía no sabes nada"


# ---------------------------------------------------------------------------
# Detección de intenciones por reglas
# ---------------------------------------------------------------------------


def normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y sin signos de puntuación."""
    t = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    t = re.sub(r"[^\w\s:]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_PATRONES: list[tuple[Intencion, list[str]]] = [
    (
        Intencion.NO_LLAMAR,
        [
            r"no (me|nos) (vuelva[ns]?|volvais) a llamar",
            r"no (me|nos) (llame[ns]?|llameis|llamen|llames) mas",
            r"no (quiero|queremos) que (me|nos) (llame|vuelva)",
            r"(borre[n]?|borra|borrad|eliminen?|quite[n]?|quitad|saque[n]?) (me|mi|nos|el)? ?(de (la|su|vuestra) lista|mis datos|mi numero|el numero|de la base)",
            r"\b(borr|quit|sac|elimin)\w*(me|nos) de (la|su|sus|vuestra|vuestras|esa|esta) (lista|base)",
            r"\b(borr|quit|elimin)(e|a|ad|en|ame|eme)(me|nos)\b",  # "bórreme", "quítenme"
            r"(dar(me)?|darnos) de baja",
            r"\blista robinson\b",
            r"deje[n]? de llamar",
            r"\bno llamen?\b.*\bmas\b",
        ],
    ),
    (
        Intencion.EQUIVOCADO,
        [
            r"se (ha|han) equivocado",
            r"numero equivocado",
            r"aqui no (vive|hay) ningun",
            r"no (conozco|se quien es) a",
        ],
    ),
    (
        Intencion.MAL_MOMENTO,
        [
            r"no (es|es un) buen momento",
            r"\bahora no (puedo|me viene|es)",
            r"^ahora no$",
            r"estoy (ocupad|conduciendo|trabajando|en una reunion|en el medico|liad)",
            r"(llame|llamame|llamenme|llamadme|vuelva a llamar|vuelve a llamar|llamen)\w* (mas tarde|luego|otro dia|en otro momento|manana|el |la semana|esta tarde|a las)",
            r"(puede|podria|puedes) llamar(me)? (mas tarde|luego|otro dia|en otro momento|manana|el |a las)",
            r"\bmejor (mas tarde|luego|otro dia|en otro momento|manana)\b",
            r"no tengo tiempo",
            r"ahora mismo no",
        ],
    ),
    (
        Intencion.NO_INTERESA,
        [
            r"no (me|nos) interesa",
            r"no (estoy|estamos) interesad",
            r"^no gracias\b",
            r"\bno,? gracias$",
            r"no (quiero|queremos|necesito|necesitamos) nada",
            r"no (lo )?(necesito|necesitamos)\b",
            r"no me hace falta",
            r"ya (tengo|tenemos) (domotica|todo|alarma|empresa)",
            r"no (estoy|estamos) (buscando|pensando)",
            r"^no me interesa",
        ],
    ),
    (
        Intencion.ORIGEN_DATOS,
        [
            r"de donde (ha[ns]?|habeis|has|han) (sacado|conseguido|obtenido|cogido)",
            r"(como|por que|porque) (tiene[ns]?|teneis|tienes) (mi|este) (numero|telefono|dato)",
            r"quien (le|les|os|te) (ha|han) dado (mi|este) (numero|telefono)",
            r"de donde (sale|salen|viene) (mi|este|el) (numero|telefono)",
        ],
    ),
    (
        Intencion.PIDE_HUMANO,
        [
            r"hablar con (una persona|alguien|un humano|una persona real|un comercial|un tecnico|un agente)",
            r"(paseme|pasame|pongame|ponme) con (una persona|alguien|un comercial)",
            r"(prefiero|quiero) (una persona|un humano|hablar con)",
        ],
    ),
    (
        Intencion.PREGUNTA_IA,
        [
            r"\b(eres|es usted|es) (un |una )?(robot|maquina|ia|inteligencia artificial|bot|grabacion|contestador)\b",
            r"\bhablo con (un |una )?(robot|maquina|persona|ia|bot)\b",
            r"\bes una persona\b",
            r"\beres (una )?persona\b",
            r"\bes (usted )?real\b",
        ],
    ),
    (
        Intencion.PREGUNTA_PRECIO,
        [
            r"\bcuanto (cuesta|vale|sale|cobra|costaria|valdria|me costaria)",
            r"\bprecio",
            r"\bpresupuesto\b",
            r"\bcuanto dinero\b",
            r"\btarifa",
        ],
    ),
]

_AFIRMATIVO = re.compile(
    r"^(si|sii|claro|vale|venga|de acuerdo|adelante|bueno|perfecto|dime|digame|cuente|cuenteme|"
    r"ok|okey|vale vale|por supuesto|sin problema|como no|dale|un minuto si|tengo un minuto|"
    r"puedo|si puedo|si tengo|si dime|si digame|si claro|bien|muy bien|me parece bien|genial)\b"
)
_NEGATIVO = re.compile(r"^(no|nop|tampoco|para nada|ninguno|ninguna)\b")


def detectar_intencion(texto: str) -> Intencion:
    """Clasifica lo que ha dicho el cliente con reglas deterministas.

    El orden de `_PATRONES` define la prioridad: pedir no ser llamado gana a todo.
    """
    t = normalizar(texto)
    if not t:
        return Intencion.OTRA
    for intencion, patrones in _PATRONES:
        if any(re.search(p, t) for p in patrones):
            return intencion
    if se_opone_a_grabacion(texto):
        return Intencion.OPONE_GRABACION
    if _AFIRMATIVO.search(t):
        return Intencion.AFIRMATIVO
    if _NEGATIVO.search(t):
        return Intencion.NEGATIVO
    return Intencion.OTRA


# ---------------------------------------------------------------------------
# Extracción de datos
# ---------------------------------------------------------------------------

_INMUEBLE = {
    "vivienda": r"\b(casa|piso|vivienda|chalet|adosado|apartamento|atico|hogar|duplex|unifamiliar)\b",
    "local": r"\b(local|oficina|negocio|tienda|nave|empresa|despacho|bar|restaurante|clinica|comercio)\b",
}
_INTERESES = {
    "persianas": r"\b(persiana|toldo|estor|cortina)",
    "luces": r"\b(luz|luces|iluminacion|bombilla|lampara|enchufe)",
    "climatizacion": r"\b(clima|climatizacion|aire acondicionado|aire|calefaccion|termostato|temperatura|caldera|suelo radiante)",
    "alarma": r"\b(alarma|seguridad|robos?\b|ladron|intrus|okupa)",
    "camaras": r"\b(camara|videovigilancia|vigilancia|videoportero|portero)",
}
_PREOCUPACIONES = {
    "ahorro": r"\b(ahorr|factura|consumo|gasto|luz cara|energia|eficien)",
    "seguridad": r"\b(segur|robos?\b|ladron|okupa|tranquil|vigil)",
    "comodidad": r"\b(comod|facil|automatic|olvid|desde el movil|a distancia|vacaciones)",
    "mayores": r"\b(mayor|anciano|abuel|movilidad|dependien|accesib)",
}


def extraer_datos(texto: str, datos: DatosLlamada) -> None:
    """Rellena en `datos` lo que se pueda deducir de `texto` por palabras clave."""
    t = normalizar(texto)
    if not datos.inmueble:
        for tipo, patron in _INMUEBLE.items():
            if re.search(patron, t):
                datos.inmueble = tipo
                break
    for clave, patron in _INTERESES.items():
        if re.search(patron, t) and clave not in datos.intereses:
            datos.intereses.append(clave)
    for clave, patron in _PREOCUPACIONES.items():
        if re.search(patron, t) and clave not in datos.preocupaciones:
            datos.preocupaciones.append(clave)


def _tipo_cita(texto: str) -> str:
    t = normalizar(texto)
    if re.search(r"\b(llamada|llame|llamen|llamar|por telefono|comercial)\b", t) and not re.search(
        r"\bvisita|vengan|venga a|pasen|pase por\b", t
    ):
        return "llamada"
    if re.search(r"\b(visita|vengan|venga|pasen|pase|vernos|en persona|tecnico)\b", t):
        return "visita"
    return ""


# ---------------------------------------------------------------------------
# Máquina de estados
# ---------------------------------------------------------------------------


class Dialogo:
    """Gestiona el estado de una llamada y decide qué decir en cada turno."""

    MAX_TURNOS_DESCUBRIMIENTO = 3
    MAX_TURNOS_CIERRE = 3
    MAX_SILENCIOS = 2

    def __init__(
        self,
        nombre_cliente: str = "",
        settings: Optional[Settings] = None,
        origen_datos: str = "",
        reloj: Optional[callable] = None,  # type: ignore[valid-type]
    ) -> None:
        self.s = settings or get_settings()
        self.zona = ZoneInfo(self.s.zona_horaria)
        self.nombre_cliente = nombre_cliente
        # Dónde dio el cliente su consentimiento (para responder "¿de dónde tenéis mi número?").
        self.origen_datos = origen_datos.strip()
        self._ultimo_paso = ""
        self._reloj = reloj or (lambda: datetime.now(self.zona))
        self.estado = Estado.SALUDO
        self.datos = DatosLlamada(grabacion_permitida=self.s.grabar_llamadas)
        self.turnos_cliente = 0
        self._turnos_estado = 0
        self._silencios = 0
        self._aclaraciones = 0

    # --- utilidades ------------------------------------------------------

    @property
    def ahora(self) -> datetime:
        return self._reloj()

    @property
    def terminado(self) -> bool:
        return self.estado is Estado.FIN

    def _ir(self, estado: Estado) -> None:
        if estado is not self.estado:
            self.estado = estado
            self._turnos_estado = 0

    def _fin(self, resultado: Resultado, texto: str, acciones: Optional[list[Accion]] = None,
             intencion: Intencion = Intencion.OTRA) -> Turno:  # fmt: skip
        self.datos.resultado = resultado
        self._ir(Estado.FIN)
        return Turno(Estado.FIN, texto=texto, terminar=True, acciones=acciones or [], intencion=intencion)

    def _despedida(self) -> str:
        return "Muchas gracias por su tiempo y que tenga muy buen día."

    # --- API pública -------------------------------------------------------

    def inicio(self) -> Turno:
        """Primer turno: saludo con todos los avisos legales."""
        self._ir(Estado.CONFIRMAR_MOMENTO)
        texto = aviso_inicial(
            self.s.nombre_agente, self.s.nombre_empresa, self.s.ciudad_empresa, self.nombre_cliente
        )
        return Turno(Estado.CONFIRMAR_MOMENTO, texto=texto)

    def silencio(self) -> Turno:
        """El cliente no dice nada durante un rato."""
        self._silencios += 1
        if self._silencios > self.MAX_SILENCIOS:
            return self._fin(
                Resultado.SIN_RESPUESTA,
                "Parece que no le oigo bien. Le dejo, disculpe la molestia y que tenga buen día.",
            )
        return Turno(self.estado, texto="¿Me oye bien?")

    def procesar(self, texto_cliente: str) -> Turno:
        """Procesa lo que ha dicho el cliente y devuelve la respuesta del agente."""
        if self.terminado:
            return Turno(Estado.FIN, terminar=True)
        self._silencios = 0
        self.turnos_cliente += 1
        self._turnos_estado += 1
        intencion = detectar_intencion(texto_cliente)
        extraer_datos(texto_cliente, self.datos)

        turno = self._globales(intencion, texto_cliente)
        if turno is None:
            manejador = {
                Estado.CONFIRMAR_MOMENTO: self._confirmar_momento,
                Estado.DESCUBRIMIENTO: self._descubrimiento,
                Estado.PROPUESTA: self._cierre,
                Estado.CIERRE: self._cierre,
                Estado.REPROGRAMAR: self._reprogramar,
                Estado.SALUDO: self._confirmar_momento,
            }[self.estado]
            turno = manejador(intencion, texto_cliente)
        turno.intencion = intencion

        if not turno.terminar and self.turnos_cliente >= self.s.max_turnos:
            return self._fin(
                Resultado.SIN_CITA,
                "No le entretengo más. " + self._despedida(),
                intencion=intencion,
            )
        return turno

    # --- intenciones válidas en cualquier estado ---------------------------

    def _globales(self, intencion: Intencion, texto: str) -> Optional[Turno]:
        if intencion is Intencion.NO_LLAMAR:
            return self._fin(
                Resultado.NO_LLAMAR,
                "Entendido, le quito ahora mismo de nuestra lista y no volveremos a llamarle. "
                "Disculpe las molestias y que tenga buen día.",
                [Accion.MARCAR_NO_LLAMAR],
            )
        if intencion is Intencion.EQUIVOCADO:
            return self._fin(
                Resultado.EQUIVOCADO,
                "Vaya, disculpe, me habrán dado mal el número. No le molestamos más, que tenga buen día.",
                [Accion.MARCAR_NO_LLAMAR],
            )
        if intencion is Intencion.NO_INTERESA:
            return self._fin(
                Resultado.NO_INTERESA,
                "De acuerdo, no se preocupe, gracias por atenderme. Que tenga muy buen día.",
            )
        if intencion is Intencion.MAL_MOMENTO and self.estado is not Estado.REPROGRAMAR:
            # En el cierre, "que me llame el comercial el martes" es una cita, no un aplazamiento.
            if not (self.estado is Estado.CIERRE and _tipo_cita(texto)):
                return self._mal_momento(texto)
        if intencion is Intencion.OPONE_GRABACION:
            self.datos.grabacion_permitida = False
            return self._continuar("Sin problema, no grabamos la llamada.", [Accion.DESACTIVAR_GRABACION])
        if intencion is Intencion.PREGUNTA_IA:
            return self._continuar(
                f"Sí, soy un asistente virtual con inteligencia artificial de {self.s.nombre_empresa}; "
                "si lo prefiere, le puede llamar una persona del equipo."
            )
        if intencion is Intencion.ORIGEN_DATOS:
            # Derecho de información (RGPD, art. 14): de dónde salen los datos.
            if self.origen_datos:
                origen = f"Nos dio su consentimiento para que le llamáramos a través de {self.origen_datos}"
            else:
                origen = f"Su número está en nuestra base de datos de clientes interesados de {self.s.nombre_empresa}"
            return self._continuar(
                f"{origen}. Si lo prefiere, le borro ahora mismo y no le volvemos a llamar."
            )
        if intencion is Intencion.PIDE_HUMANO:
            self.datos.cita_tipo = "llamada"
            self._ir(Estado.CIERRE)
            fecha = extraer_fecha_hora(texto, self.ahora)
            if fecha.tiene_dia or fecha.tiene_hora:
                return self._confirmar_cita(fecha, texto)
            return Turno(
                Estado.CIERRE,
                texto="Claro, le pido a un compañero del equipo comercial que le llame. "
                "¿Qué día y a qué hora le viene bien?",
            )
        if intencion is Intencion.PREGUNTA_PRECIO:
            self._ir(Estado.CIERRE)
            return Turno(
                Estado.CIERRE,
                texto="El precio depende de cada casa, así que no le quiero dar una cifra al aire; "
                "se lo concreta un comercial sin compromiso. ¿Le parece que le agendemos una visita gratuita?",
            )
        return None

    def _continuar(self, prefijo: str, acciones: Optional[list[Accion]] = None) -> Turno:
        """Dice `prefijo` y repite la pregunta pendiente del estado actual."""
        pregunta = self._pregunta_pendiente()
        return Turno(self.estado, texto=f"{prefijo} {pregunta}".strip(), acciones=acciones or [])

    def _pregunta_pendiente(self) -> str:
        if self.estado is Estado.CONFIRMAR_MOMENTO:
            return "¿Tiene un minuto?"
        if self.estado is Estado.DESCUBRIMIENTO:
            return prompts.instruccion_descubrimiento(self._paso_descubrimiento(), "").respaldo
        if self.estado in (Estado.PROPUESTA, Estado.CIERRE):
            if self.datos.cita_tipo:
                return "¿Qué día y a qué hora le viene bien?"
            return "¿Le vendría bien una visita gratuita, o prefiere que le llame un comercial?"
        if self.estado is Estado.REPROGRAMAR:
            return "¿Cuándo le viene mejor que le llamemos?"
        return ""

    # --- estados -----------------------------------------------------------

    def _confirmar_momento(self, intencion: Intencion, texto: str) -> Turno:
        if intencion in (Intencion.NEGATIVO,):
            return self._mal_momento(texto)
        if intencion is Intencion.AFIRMATIVO:
            self._ir(Estado.DESCUBRIMIENTO)
            return self._turno_descubrimiento()
        # Respuesta ambigua ("¿de qué se trata?"): se explica y se avanza.
        self._ir(Estado.DESCUBRIMIENTO)
        if self.datos.inmueble:
            return self._turno_descubrimiento()
        ins = prompts.instruccion_aclarar_momento()
        return Turno(Estado.DESCUBRIMIENTO, instruccion=ins.objetivo, respaldo=ins.respaldo)

    def _mal_momento(self, texto: str) -> Turno:
        fecha = extraer_fecha_hora(texto, self.ahora)
        if fecha.encontrada:
            return self._guardar_rellamada(fecha, texto)
        self._ir(Estado.REPROGRAMAR)
        return Turno(
            Estado.REPROGRAMAR,
            texto="Sin problema. ¿Cuándo le viene mejor que le llamemos?",
        )

    def _reprogramar(self, intencion: Intencion, texto: str) -> Turno:
        fecha = extraer_fecha_hora(texto, self.ahora)
        if fecha.encontrada:
            return self._guardar_rellamada(fecha, texto)
        if intencion is Intencion.NEGATIVO or self._turnos_estado >= 2:
            # No da hora: se reintentará más adelante dentro de los límites de intentos.
            self.datos.rellamar_texto = "sin hora concreta"
            return self._fin(
                Resultado.REPROGRAMAR,
                "De acuerdo, le llamaremos otro día. " + self._despedida(),
                [Accion.REPROGRAMAR],
            )
        return Turno(Estado.REPROGRAMAR, texto="¿Le viene mejor por la mañana o por la tarde, y qué día?")

    def _guardar_rellamada(self, fecha: FechaHablada, texto: str) -> Turno:
        if fecha.momento is None:  # "luego", "otro día" sin concretar
            self.datos.rellamar = fecha
            self.datos.rellamar_texto = texto.strip()
            return self._fin(
                Resultado.REPROGRAMAR,
                "Perfecto, le llamamos en otro momento. " + self._despedida(),
                [Accion.REPROGRAMAR],
            )
        self.datos.rellamar = fecha
        self.datos.rellamar_texto = describir(fecha, self.ahora)
        return self._fin(
            Resultado.REPROGRAMAR,
            f"Perfecto, le llamo {self.datos.rellamar_texto}. Gracias y hasta luego.",
            [Accion.REPROGRAMAR],
        )

    def _paso_descubrimiento(self) -> str:
        if not self.datos.inmueble:
            return "inmueble"
        if not self.datos.intereses:
            return "intereses"
        return "preocupacion"

    def _turno_descubrimiento(self) -> Turno:
        paso = self._paso_descubrimiento()
        self._ultimo_paso = paso
        ins = prompts.instruccion_descubrimiento(paso, self.datos.resumen())
        return Turno(Estado.DESCUBRIMIENTO, instruccion=ins.objetivo, respaldo=ins.respaldo)

    def _descubrimiento(self, intencion: Intencion, texto: str) -> Turno:
        completo = bool(self.datos.inmueble and self.datos.intereses and self.datos.preocupaciones)
        if completo or self._turnos_estado >= self.MAX_TURNOS_DESCUBRIMIENTO or (
            self.datos.intereses and self._turnos_estado >= 2
        ):
            return self._turno_propuesta()
        if self._paso_descubrimiento() == self._ultimo_paso and intencion is not Intencion.PREGUNTA_IA:
            # El cliente no ha sabido contestar ("no sé"): no se repite la pregunta.
            if self._ultimo_paso == "intereses":
                self._ultimo_paso = "preocupacion"
                ins = prompts.instruccion_descubrimiento("preocupacion", self.datos.resumen())
                return Turno(Estado.DESCUBRIMIENTO, instruccion=ins.objetivo, respaldo=ins.respaldo)
            if self._ultimo_paso == "preocupacion":
                return self._turno_propuesta()
        return self._turno_descubrimiento()

    def _turno_propuesta(self) -> Turno:
        self._ir(Estado.CIERRE)
        ins = prompts.instruccion_propuesta(
            self.datos.resumen(), self.datos.intereses, self.s.nombre_empresa
        )
        return Turno(Estado.CIERRE, instruccion=ins.objetivo, respaldo=ins.respaldo)

    def _cierre(self, intencion: Intencion, texto: str) -> Turno:
        tipo = _tipo_cita(texto)
        if tipo:
            self.datos.cita_tipo = tipo
        fecha = extraer_fecha_hora(texto, self.ahora)

        if fecha.tiene_dia or fecha.tiene_hora:
            if not self.datos.cita_tipo:
                self.datos.cita_tipo = "visita"
            return self._confirmar_cita(fecha, texto)

        if intencion is Intencion.AFIRMATIVO or tipo:
            if not self.datos.cita_tipo:
                self.datos.cita_tipo = "visita"
            que = "la visita" if self.datos.cita_tipo == "visita" else "la llamada del comercial"
            return Turno(Estado.CIERRE, texto=f"Estupendo. ¿Qué día y a qué hora le viene bien {que}?")

        if intencion is Intencion.NEGATIVO:
            return self._fin(
                Resultado.SIN_CITA,
                "De acuerdo, sin problema. Si algún día lo necesita, aquí estamos. " + self._despedida(),
            )

        if self._turnos_estado > self.MAX_TURNOS_CIERRE:
            return self._fin(Resultado.SIN_CITA, "No le entretengo más. " + self._despedida())
        ins = prompts.instruccion_respuesta_libre(self.datos.resumen())
        return Turno(Estado.CIERRE, instruccion=ins.objetivo, respaldo=ins.respaldo)

    def _confirmar_cita(self, fecha: FechaHablada, texto: str) -> Turno:
        anterior = self.datos.cita
        # Combina "el martes" de un turno con "a las cinco" del siguiente.
        if anterior and anterior.momento and fecha.momento:
            if anterior.tiene_dia and not fecha.tiene_dia and fecha.tiene_hora:
                fecha = FechaHablada(
                    datetime.combine(anterior.momento.date(), fecha.momento.timetz()), True, True
                )
            elif anterior.tiene_hora and fecha.tiene_dia and not fecha.tiene_hora:
                fecha = FechaHablada(
                    datetime.combine(fecha.momento.date(), anterior.momento.timetz()), True, True
                )
        self.datos.cita = fecha

        if fecha.tiene_dia and not fecha.tiene_hora and self._turnos_estado <= self.MAX_TURNOS_CIERRE:
            return Turno(
                Estado.CIERRE,
                texto=f"Muy bien, {describir(fecha, self.ahora)}. ¿Y a qué hora le viene mejor?",
            )
        if fecha.momento and fecha.momento < self.ahora - timedelta(minutes=5):
            self.datos.cita = None
            return Turno(Estado.CIERRE, texto="Perdone, creo que no le he entendido bien el día. ¿Qué día le viene bien?")

        self.datos.cita_texto = describir(fecha, self.ahora)
        que = "la visita gratuita" if self.datos.cita_tipo == "visita" else "la llamada con un comercial"
        return self._fin(
            Resultado.CITA,
            f"Perfecto, le apunto {que} {self.datos.cita_texto}. "
            "Un compañero le confirmará la cita. Muchas gracias y que tenga muy buen día.",
            [Accion.GUARDAR_CITA],
        )
