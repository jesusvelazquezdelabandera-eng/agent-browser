"""Configuración común de las pruebas."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Permite `import app` al ejecutar pytest desde la carpeta del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Las pruebas nunca deben usar Ollama, Asterisk ni la base de datos real.
os.environ.setdefault("LLM_BACKEND", "plantillas")
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("WORKER_ACTIVO", "false")
os.environ.setdefault("VAD_BACKEND", "energia")
