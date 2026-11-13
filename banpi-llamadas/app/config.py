"""Configuración del agente por variables de entorno.

Todos los valores tienen un valor por defecto razonable para poder arrancar en
modo simulación sin configurar nada. Para llamadas reales hay que rellenar al
menos las variables de Asterisk (ver `.env.example`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Raíz del proyecto (carpeta que contiene `app/`).
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent


def _cargar_dotenv(ruta: Path) -> None:
    """Carga un archivo `.env` sencillo (CLAVE=valor) sin dependencias externas.

    Las variables ya definidas en el entorno tienen prioridad sobre el archivo.
    """
    if not ruta.is_file():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        clave = clave.strip()
        valor = valor.strip().strip('"').strip("'")
        os.environ.setdefault(clave, valor)


def _str(nombre: str, defecto: str) -> str:
    return os.environ.get(nombre, defecto)


def _int(nombre: str, defecto: int) -> int:
    valor = os.environ.get(nombre)
    return int(valor) if valor not in (None, "") else defecto


def _float(nombre: str, defecto: float) -> float:
    valor = os.environ.get(nombre)
    return float(valor) if valor not in (None, "") else defecto


def _bool(nombre: str, defecto: bool) -> bool:
    valor = os.environ.get(nombre)
    if valor in (None, ""):
        return defecto
    return valor.strip().lower() in {"1", "true", "si", "sí", "yes", "on"}


def _ruta(nombre: str, defecto: str) -> Path:
    """Devuelve una ruta; si es relativa, se interpreta desde la raíz del proyecto."""
    ruta = Path(os.environ.get(nombre, defecto)).expanduser()
    return ruta if ruta.is_absolute() else RAIZ_PROYECTO / ruta


@dataclass(frozen=True)
class Settings:
    """Configuración completa de la aplicación."""

    # Empresa y agente
    nombre_agente: str = "Álex"
    nombre_empresa: str = "Banpi Domótica"
    ciudad_empresa: str = "Sevilla"
    telefono_empresa: str = ""
    web_empresa: str = ""
    servicios: str = (
        "automatización de persianas y toldos, iluminación inteligente, "
        "climatización, alarmas y videovigilancia, y control desde el móvil"
    )
    zona_servicio: str = "Sevilla y provincia"

    # Base de datos
    database_url: str = "sqlite:///./banpi.db"

    # LLM (Ollama)
    llm_backend: str = "ollama"  # "ollama" o "plantillas" (sin modelo, para pruebas)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_temperatura: float = 0.4
    ollama_timeout_s: float = 20.0
    ollama_keep_alive: str = "30m"
    # Si el LLM no da su primera frase en este tiempo, Álex usa la frase de respaldo.
    llm_timeout_primera_s: float = 2.5

    # Reconocimiento de voz (faster-whisper)
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_dir: Path = field(default_factory=lambda: RAIZ_PROYECTO / "modelos" / "whisper")

    # Síntesis de voz (Piper)
    piper_voice: str = "es_ES-davefx-medium"
    piper_dir: Path = field(default_factory=lambda: RAIZ_PROYECTO / "modelos" / "piper")
    piper_cuda: bool = False
    # Duración relativa de la voz: < 1 habla más rápido (0.9 = un 10 % más rápido).
    piper_length_scale: float = 0.9
    tts_cache_dir: Path = field(default_factory=lambda: RAIZ_PROYECTO / "data" / "cache_tts")

    # Detección de voz (VAD)
    vad_backend: str = "webrtc"  # "webrtc" o "energia"
    vad_agresividad: int = 2
    vad_silencio_fin_ms: int = 600
    vad_min_voz_ms: int = 250
    barge_in_ms: int = 250

    # Telefonía
    telefonia_backend: str = "asterisk"  # "asterisk" o "twilio"
    ari_url: str = "http://127.0.0.1:8088"
    ari_usuario: str = "banpi"
    ari_password: str = "cambia-esta-clave"
    ari_app: str = "banpi"
    sip_endpoint: str = "PJSIP/{numero}@troncal"
    # Cómo espera el proveedor el número: "nacional" (600112233), "e164" (+34600112233),
    # "internacional" (0034600112233) o "sin_mas" (34600112233).
    sip_formato_numero: str = "nacional"
    caller_id: str = ""
    rtp_host_local: str = "127.0.0.1"
    rtp_puerto_inicio: int = 40000
    rtp_puerto_fin: int = 40100
    timeout_respuesta_s: int = 30
    grabaciones_dir: Path = field(default_factory=lambda: RAIZ_PROYECTO / "data" / "grabaciones")

    # Cumplimiento y campañas
    zona_horaria: str = "Europe/Madrid"
    hora_inicio: int = 9
    hora_fin: int = 21
    max_intentos: int = 3
    ventana_intentos_dias: int = 7
    reintento_minutos: int = 120
    llamadas_simultaneas: int = 1
    robinson_file: Path = field(default_factory=lambda: RAIZ_PROYECTO / "data" / "robinson.txt")
    festivos_extra_file: Path = field(
        default_factory=lambda: RAIZ_PROYECTO / "data" / "festivos_extra.txt"
    )
    grabar_llamadas: bool = True
    worker_activo: bool = True
    worker_intervalo_s: float = 10.0

    # Conversación
    max_turnos: int = 20
    max_duracion_s: int = 300

    # Panel web
    panel_host: str = "127.0.0.1"
    panel_puerto: int = 8000
    panel_usuario: str = ""
    panel_password: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Lee la configuración del entorno (y de `.env` si existe) una sola vez."""
    _cargar_dotenv(RAIZ_PROYECTO / ".env")
    return Settings(
        nombre_agente=_str("AGENTE_NOMBRE", "Álex"),
        nombre_empresa=_str("EMPRESA_NOMBRE", "Banpi Domótica"),
        ciudad_empresa=_str("EMPRESA_CIUDAD", "Sevilla"),
        telefono_empresa=_str("EMPRESA_TELEFONO", ""),
        web_empresa=_str("EMPRESA_WEB", ""),
        servicios=_str("EMPRESA_SERVICIOS", Settings.servicios),
        zona_servicio=_str("EMPRESA_ZONA", "Sevilla y provincia"),
        database_url=_str("DATABASE_URL", f"sqlite:///{RAIZ_PROYECTO / 'data' / 'banpi.db'}"),
        llm_backend=_str("LLM_BACKEND", "ollama"),
        ollama_url=_str("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=_str("OLLAMA_MODEL", "qwen2.5:7b"),
        ollama_temperatura=_float("OLLAMA_TEMPERATURA", 0.4),
        ollama_timeout_s=_float("OLLAMA_TIMEOUT_S", 20.0),
        ollama_keep_alive=_str("OLLAMA_KEEP_ALIVE", "30m"),
        llm_timeout_primera_s=_float("LLM_TIMEOUT_PRIMERA_S", 2.5),
        whisper_model=_str("WHISPER_MODEL", "small"),
        whisper_device=_str("WHISPER_DEVICE", "auto"),
        whisper_compute_type=_str("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_dir=_ruta("WHISPER_DIR", "modelos/whisper"),
        piper_voice=_str("PIPER_VOICE", "es_ES-davefx-medium"),
        piper_dir=_ruta("PIPER_DIR", "modelos/piper"),
        piper_cuda=_bool("PIPER_CUDA", False),
        piper_length_scale=_float("PIPER_LENGTH_SCALE", 0.9),
        tts_cache_dir=_ruta("TTS_CACHE_DIR", "data/cache_tts"),
        vad_backend=_str("VAD_BACKEND", "webrtc"),
        vad_agresividad=_int("VAD_AGRESIVIDAD", 2),
        vad_silencio_fin_ms=_int("VAD_SILENCIO_FIN_MS", 600),
        vad_min_voz_ms=_int("VAD_MIN_VOZ_MS", 250),
        barge_in_ms=_int("BARGE_IN_MS", 250),
        telefonia_backend=_str("TELEFONIA_BACKEND", "asterisk"),
        ari_url=_str("ARI_URL", "http://127.0.0.1:8088"),
        ari_usuario=_str("ARI_USUARIO", "banpi"),
        ari_password=_str("ARI_PASSWORD", "cambia-esta-clave"),
        ari_app=_str("ARI_APP", "banpi"),
        sip_endpoint=_str("SIP_ENDPOINT", "PJSIP/{numero}@troncal"),
        sip_formato_numero=_str("SIP_FORMATO_NUMERO", "nacional"),
        caller_id=_str("CALLER_ID", ""),
        rtp_host_local=_str("RTP_HOST_LOCAL", "127.0.0.1"),
        rtp_puerto_inicio=_int("RTP_PUERTO_INICIO", 40000),
        rtp_puerto_fin=_int("RTP_PUERTO_FIN", 40100),
        timeout_respuesta_s=_int("TIMEOUT_RESPUESTA_S", 30),
        grabaciones_dir=_ruta("GRABACIONES_DIR", "data/grabaciones"),
        zona_horaria=_str("ZONA_HORARIA", "Europe/Madrid"),
        hora_inicio=_int("HORA_INICIO", 9),
        hora_fin=_int("HORA_FIN", 21),
        max_intentos=_int("MAX_INTENTOS", 3),
        ventana_intentos_dias=_int("VENTANA_INTENTOS_DIAS", 7),
        reintento_minutos=_int("REINTENTO_MINUTOS", 120),
        llamadas_simultaneas=_int("LLAMADAS_SIMULTANEAS", 1),
        robinson_file=_ruta("ROBINSON_FILE", "data/robinson.txt"),
        festivos_extra_file=_ruta("FESTIVOS_EXTRA_FILE", "data/festivos_extra.txt"),
        grabar_llamadas=_bool("GRABAR_LLAMADAS", True),
        worker_activo=_bool("WORKER_ACTIVO", True),
        worker_intervalo_s=_float("WORKER_INTERVALO_S", 10.0),
        max_turnos=_int("MAX_TURNOS", 20),
        max_duracion_s=_int("MAX_DURACION_S", 300),
        panel_host=_str("PANEL_HOST", "127.0.0.1"),
        panel_puerto=_int("PANEL_PUERTO", 8000),
        panel_usuario=_str("PANEL_USUARIO", ""),
        panel_password=_str("PANEL_PASSWORD", ""),
    )
