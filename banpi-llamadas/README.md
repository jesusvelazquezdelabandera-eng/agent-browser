# Agente de llamadas salientes de Banpi Domótica

Álex es un agente de voz que llama a una lista de contactos, presenta los servicios de domótica de Banpi Domótica (Sevilla) y agenda una visita gratuita o una llamada con un comercial. Todo funciona en local con software libre: el único coste es el minuto de la línea SIP.

> **Aviso legal.** Este proyecto no sustituye al asesoramiento legal. Antes de llamar a nadie, revisa con un profesional el RGPD y la LOPDGDD, la Ley 11/2022 General de Telecomunicaciones (art. 66), la Circular 1/2023 de la AEPD, la Ley 10/2025 de servicios de atención a la clientela (prefijo 400), el texto refundido de la Ley General para la Defensa de los Consumidores y Usuarios y el Reglamento europeo de Inteligencia Artificial (art. 50). Consulta la sección [Cumplimiento legal](#cumplimiento-legal).

## Cómo funciona

```
Cliente ⇄ proveedor SIP ⇄ Asterisk ──ARI (órdenes y eventos)──▶ app Python
                              └──External Media (RTP μ-law 8 kHz)──▶ VAD → Whisper → diálogo → Ollama → Piper ─┐
                              ◀───────────────────────────────────────────────────────────────── audio del agente ┘
```

<table>
<tr><th>Pieza</th><th>Herramienta</th><th>Dónde</th></tr>
<tr><td>Reconocimiento de voz</td><td>faster-whisper (<code>small</code> por defecto), en local</td><td><code>app/audio/stt.py</code></td></tr>
<tr><td>Modelo de lenguaje</td><td>Ollama con <code>qwen2.5:7b</code> (configurable)</td><td><code>app/agent/llm.py</code></td></tr>
<tr><td>Síntesis de voz</td><td>Piper con <code>es_ES-davefx-medium</code></td><td><code>app/audio/tts.py</code></td></tr>
<tr><td>Detección de voz</td><td>webrtcvad</td><td><code>app/audio/vad.py</code></td></tr>
<tr><td>Telefonía</td><td>Asterisk + ARI + External Media; interfaz preparada para Twilio</td><td><code>app/telephony/</code></td></tr>
<tr><td>Conversación</td><td>Máquina de estados con reglas + LLM para redactar</td><td><code>app/agent/dialog.py</code>, <code>prompts.py</code></td></tr>
<tr><td>Cumplimiento</td><td>Consentimiento, Robinson, horario, intentos, auditoría</td><td><code>app/compliance.py</code></td></tr>
<tr><td>Base de datos</td><td>SQLite + SQLAlchemy</td><td><code>app/db/</code></td></tr>
<tr><td>Panel y campañas</td><td>FastAPI + worker asíncrono</td><td><code>app/main.py</code>, <code>app/campaigns/worker.py</code></td></tr>
</table>

### La conversación

1. **Saludo**: Álex dice su nombre, que es un asistente de inteligencia artificial de Banpi Domótica en Sevilla, que llama para ofrecer domótica, que la llamada puede grabarse y que el cliente puede oponerse o pedir que no le llamen más. Este saludo no se puede interrumpir, porque contiene los avisos legales; lo que diga el cliente mientras tanto se responde al terminar.
2. **¿Es buen momento?** Si no lo es, pregunta cuándo llamar y guarda la hora ("mañana a las cinco", "el lunes por la mañana").
3. **Descubrimiento**: vivienda o local, qué le gustaría automatizar (persianas, luces, climatización, alarma, cámaras) y qué le preocupa (ahorro, seguridad, comodidad).
4. **Propuesta breve** adaptada a lo que ha contado, sin precios.
5. **Cierre**: visita gratuita sin compromiso o llamada con un comercial, con día y hora.
6. **Despedida.**

Las decisiones importantes las toman reglas deterministas, no el LLM: "no me llaméis más" (marca el contacto como no contactar al instante), "no me interesa" (se despide con educación), "ahora no puedo", la oposición a la grabación, "¿eres un robot?" (siempre reconoce que es una IA), las preguntas de precio (nunca da cifras) y "quiero hablar con una persona". El LLM solo redacta las respuestas, con un máximo de dos frases; si inventa un precio, la frase se sustituye. Si Ollama falla o tarda más de 4 segundos, Álex usa una frase de respaldo y la llamada sigue.

El cliente puede interrumpir a Álex: en cuanto el VAD detecta 250 ms de voz, se corta el audio pendiente y se cancela la generación.

### Latencia

El objetivo es que pasen menos de 1,5 s desde que el cliente deja de hablar hasta que Álex empieza a responder. Para conseguirlo, la respuesta del LLM llega en streaming, se corta por frases, cada frase se sintetiza en cuanto está lista, la primera frase se parte por la coma para que suene antes y las frases fijas se guardan en caché. La latencia de cada turno se guarda en la base de datos y se ve en el panel.

Medidas en una máquina de 4 núcleos **sin GPU**, con el modo guion (Whisper + Piper reales, sin LLM). La latencia incluye los 600 ms de silencio que espera el VAD para dar el turno por terminado:

<table>
<tr><th>Whisper</th><th>Transcripción por turno</th><th>Latencia percibida media</th></tr>
<tr><td><code>small</code></td><td>1255 ms</td><td>1964 ms</td></tr>
<tr><td><code>base</code></td><td>567 ms</td><td>1288 ms</td></tr>
</table>

Ollama suma el tiempo hasta su primera frase: con GPU suele ser de unos cientos de milisegundos, y en CPU un modelo de 7B puede tardar varios segundos. Sin GPU, usa `WHISPER_MODEL=base` y un modelo más pequeño (`OLLAMA_MODEL=qwen2.5:3b`). Con GPU NVIDIA, `small` o `medium` y `qwen2.5:7b` entran en el objetivo.

## Requisitos

- Linux (probado en Ubuntu 24.04) o Windows 10/11 con WSL2.
- Python 3.11 o superior.
- Docker con Docker Compose, para Asterisk y Ollama (o instálalos directamente).
- 16 GB de RAM recomendados; una GPU NVIDIA mejora mucho la latencia.
- Una cuenta en un proveedor SIP con un número español.

## Instalación en Linux

```bash
cd banpi-llamadas
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Para la simulación con micrófono, instala también PortAudio: `sudo apt install libportaudio2`.

## Instalación en Windows con WSL

1. En PowerShell como administrador: `wsl --install -d Ubuntu-24.04` y reinicia.
2. Instala [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) y activa la integración con WSL en **Settings > Resources > WSL integration**.
3. Abre Ubuntu y sigue los pasos de [Instalación en Linux](#instalación-en-linux). Trabaja dentro del sistema de archivos de Linux (por ejemplo, `~/banpi-llamadas`), no en `/mnt/c/`, porque es mucho más rápido.
4. Para la GPU NVIDIA en WSL, instala el driver normal de Windows; WSL2 lo usa directamente.
5. `network_mode: host` de Docker funciona en WSL2 con Docker Desktop reciente. Si Asterisk no recibe audio del proveedor, instala Asterisk dentro de Ubuntu con `sudo apt install asterisk` y copia los archivos de `asterisk/conf/` a `/etc/asterisk/`.

El micrófono desde WSL depende de WSLg; si no funciona, usa la simulación por texto o por guion.

## Descargar los modelos

**Ollama** (modelo de lenguaje):

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
```

**Piper** (voz) y **Whisper** (reconocimiento) se descargan solos la primera vez que se usan, en `modelos/`. Para descargarlos antes:

```bash
python -m piper.download_voices es_ES-davefx-medium --download-dir modelos/piper
python -c "from faster_whisper import WhisperModel; WhisperModel('small', download_root='modelos/whisper')"
```

## Probar en modo simulación

Prueba siempre en simulación antes de llamar a nadie.

**Por texto** (tú haces de cliente):

```bash
python -m app.simulate --nombre "María"
python -m app.simulate --llm plantillas      # sin Ollama, con frases de respaldo
```

**Por guion de voz**: el cliente dice las frases de `data/guion_ejemplo.txt` con voz sintética, y se usa el pipeline completo (VAD, Whisper, LLM y Piper) a 8 kHz, como en una llamada real. Muestra la latencia de cada turno y guarda lo que dice Álex en `data/simulacion_agente.wav`. Una línea que empiece por `!` interrumpe a Álex mientras habla.

```bash
python -m app.simulate --modo guion
```

**Con micrófono** (usa auriculares; con altavoces, Álex se oye a sí mismo y se interrumpe):

```bash
python -m app.simulate --modo microfono
```

## Configurar Asterisk con un proveedor SIP

1. Contrata una cuenta SIP con un número español y anota el servidor, el usuario y la contraseña.
2. Edita `asterisk/conf/pjsip.conf` y sustituye `SIP_SERVIDOR`, `SIP_USUARIO` y `SIP_PASSWORD` en todo el archivo. Si la máquina está detrás de un router, descomenta `external_media_address`, `external_signaling_address` y `local_net` y pon tu IP pública.
3. Cambia la contraseña de ARI en `asterisk/conf/ari.conf` y pon la misma en `ARI_PASSWORD` del `.env`.
4. Arranca Asterisk:

   ```bash
   mkdir -p data/grabaciones
   docker compose up -d asterisk
   docker compose exec asterisk asterisk -rx "pjsip show registrations"
   ```

   El registro debe aparecer como `Registered`.
5. En el `.env`, ajusta `SIP_FORMATO_NUMERO` al formato que pida tu proveedor (`nacional`, `e164`, `internacional` o `sin_mas`) y `CALLER_ID` al número que verá el cliente.
6. Si la máquina tiene firewall, abre UDP 5060 (SIP) y 10000 a 10200 (RTP de Asterisk) hacia tu proveedor.

El audio entre Asterisk y la app usa μ-law 8 kHz a propósito: en Asterisk 20 a 24, External Media sobre RTP descarta sin avisar el audio entrante en `slin16` ([asterisk#2268](https://github.com/asterisk/asterisk/issues/2268)).

Para comprobar ARI y External Media sin proveedor, con Asterisk en la misma máquina y Whisper y Piper disponibles:

```bash
BANPI_E2E_ASTERISK=1 python -m pytest tests/test_e2e_asterisk.py -s
```

Un canal local de Asterisk hace de cliente y dice "No me interesa, gracias"; la prueba comprueba que el audio llega, que Whisper lo entiende y que Álex se despide y cuelga.

## Lanzar una campaña de prueba

1. Arranca el panel y el worker:

   ```bash
   python -m app.main
   ```

2. Abre <http://127.0.0.1:8000>.
3. Edita `data/contactos_ejemplo.csv` y pon **tu propio móvil** en la fila "Cliente de prueba". Impórtalo en **1. Cargar contactos**.
4. Crea una campaña en **2. Crear campaña** y pulsa **Lanzar**.
5. Si es un día laborable entre las 9:00 y las 21:00, tu móvil sonará en unos segundos. Al terminar, en **Últimas llamadas** verás el resultado, la transcripción y la latencia; las citas aparecen en **Citas**.

El CSV usa estas columnas (separadas por `;` o `,`):

<table>
<tr><th>Columna</th><th>Contenido</th></tr>
<tr><td><code>nombre</code></td><td>Nombre del cliente; Álex usa solo el nombre de pila al saludar.</td></tr>
<tr><td><code>telefono</code></td><td>Teléfono; los de 9 cifras se tratan como españoles (+34).</td></tr>
<tr><td><code>email</code></td><td>Opcional.</td></tr>
<tr><td><code>consentimiento</code></td><td><code>sí</code> si el contacto aceptó expresamente recibir llamadas comerciales automatizadas.</td></tr>
<tr><td><code>consentimiento_fecha</code></td><td>Fecha del consentimiento (<code>AAAA-MM-DD</code> o <code>DD/MM/AAAA</code>).</td></tr>
<tr><td><code>consentimiento_origen</code></td><td>Dónde se obtuvo (formulario web, tienda...), para poder demostrarlo.</td></tr>
<tr><td><code>base_legal</code></td><td><code>consentimiento</code> o <code>interes_legitimo</code>.</td></tr>
</table>

El panel no tiene contraseña por defecto y solo escucha en `127.0.0.1`. Si lo vas a abrir a otros equipos, define `PANEL_USUARIO` y `PANEL_PASSWORD`, cambia `PANEL_HOST` y ponlo detrás de HTTPS.

## Cumplimiento legal

Antes de cada llamada, `app/compliance.py` comprueba, en este orden:

1. Que el teléfono es válido y el contacto no ha pedido que no le llamen.
2. Que no está en la lista interna de no llamar ni en `data/robinson.txt`. Como entidad adherida, descarga la exportación oficial de la Lista Robinson (ADigital) y sustituye ese archivo antes de cada campaña.
3. Que tiene **consentimiento expreso**. Álex es un sistema automático sin intervención humana, y el art. 66.1.a de la Ley 11/2022 exige consentimiento previo para ese tipo de llamadas comerciales; el interés legítimo que admite el art. 66.1.b no basta. Por eso los contactos con `base_legal = interes_legitimo` y sin consentimiento no los llama la IA: pasan a la **cola para comerciales humanos**, que se exporta desde el panel.
4. Que es lunes a viernes, de 9:00 a 21:00 hora de Madrid, y no es festivo nacional (Viernes Santo incluido). Añade los festivos de Andalucía y de Sevilla en `data/festivos_extra.txt`.
5. Que no se ha llamado ya 3 veces en los últimos 7 días. Además, tras 3 intentos sin hablar con el cliente, el contacto se da por cerrado en esa campaña.

Durante la llamada:

- El saludo informa de la identidad de la empresa, de que habla una IA (AI Act, art. 50, aplicable desde el 2 de agosto de 2026), de la finalidad comercial, del derecho a oponerse (Circular 1/2023, art. 6) y de la grabación.
- La llamada se graba en Asterisk solo si `GRABAR_LLAMADAS=true`; si el cliente se opone, la grabación se detiene y se borra.
- "No me llaméis más" marca el contacto como no contactar y lo añade a la lista interna en el mismo instante.

Cada decisión (importación, comprobación de cumplimiento, marcación, oposición, cita, resultado) queda en la tabla `auditoria`, que se exporta desde el panel.

**Prefijo 400.** Desde el 17 de octubre de 2026, las empresas dentro del ámbito de la Ley 10/2025 (grandes empresas y servicios básicos) deben hacer las llamadas comerciales desde un número 400. Según las fuentes consultadas, una pyme de domótica queda fuera, pero algunas fuentes lo interpretan de otra forma: confírmalo con tu asesor y, si te aplica, pon un número 400 en `CALLER_ID`.

## Configuración

Todas las opciones están en `.env.example`, con comentarios. Las más habituales:

<table>
<tr><th>Variable</th><th>Por defecto</th><th>Para qué sirve</th></tr>
<tr><td><code>OLLAMA_MODEL</code></td><td><code>qwen2.5:7b</code></td><td>Modelo de Ollama. Con CPU, prueba <code>qwen2.5:3b</code>.</td></tr>
<tr><td><code>LLM_BACKEND</code></td><td><code>ollama</code></td><td><code>plantillas</code> usa frases fijas sin modelo.</td></tr>
<tr><td><code>WHISPER_MODEL</code></td><td><code>small</code></td><td><code>base</code> en CPU, <code>small</code> o <code>medium</code> con GPU.</td></tr>
<tr><td><code>PIPER_LENGTH_SCALE</code></td><td><code>0.9</code></td><td>Velocidad de la voz; menor es más rápida.</td></tr>
<tr><td><code>VAD_SILENCIO_FIN_MS</code></td><td><code>600</code></td><td>Silencio que marca el fin del turno del cliente.</td></tr>
<tr><td><code>BARGE_IN_MS</code></td><td><code>250</code></td><td>Voz necesaria para interrumpir a Álex.</td></tr>
<tr><td><code>LLAMADAS_SIMULTANEAS</code></td><td><code>1</code></td><td>Llamadas a la vez; cada una carga CPU o GPU.</td></tr>
<tr><td><code>EMPRESA_SERVICIOS</code>, <code>EMPRESA_ZONA</code>, <code>EMPRESA_TELEFONO</code></td><td></td><td>Datos que Álex puede contar. Revísalos: el LLM solo usa lo que pongas aquí.</td></tr>
</table>

Para cambiar el tono o las frases, edita `app/agent/prompts.py` (prompt de sistema e instrucciones por estado) y `app/compliance.py` (texto del saludo).

## Pruebas

```bash
python -m pytest
```

Ejecuta las pruebas de cumplimiento, diálogo, fechas habladas, LLM (con un Ollama simulado), audio, telefonía (con un Asterisk simulado y RTP real por UDP), pipeline con interrupciones, worker y panel. No necesitan modelos ni Asterisk.

## Estructura

```
app/
  main.py               panel FastAPI y arranque del worker
  config.py             configuración por variables de entorno
  compliance.py         reglas legales y de consentimiento
  simulate.py           simulación por consola (texto, guion, micrófono)
  agent/                diálogo, prompts, LLM, fechas habladas y pipeline de la llamada
  audio/                VAD, Whisper, Piper y conversión de audio
  telephony/            interfaz abstracta, Asterisk ARI, RTP, simulación y esqueleto de Twilio
  campaigns/worker.py   cola de llamadas con horarios y reintentos
  db/                   modelos y sesión de SQLAlchemy
  web/templates/        panel HTML
asterisk/               imagen Docker y configuración de Asterisk
data/                   CSV de ejemplo, Robinson, festivos y guion de simulación
tests/                  pruebas automáticas
```

## Problemas frecuentes

- **El cliente no oye nada o Álex no oye al cliente.** Casi siempre es NAT. Revisa `external_media_address` y `local_net` en `pjsip.conf` y que los puertos RTP estén abiertos. Con `docker compose exec asterisk asterisk -rvvv` verás la llamada en directo.
- **Álex tarda mucho en contestar.** Mira la latencia por turno en el panel. Usa un Whisper más pequeño, un modelo de Ollama más pequeño o una GPU.
- **Álex se interrumpe solo en la simulación con micrófono.** Usa auriculares.
- **Error al originar la llamada.** Comprueba que el troncal está registrado y que `SIP_FORMATO_NUMERO` coincide con lo que espera tu proveedor.
- **No llama a nadie.** Mira en el panel el motivo de cada contacto en la campaña (fuera de horario, sin consentimiento...). Los detalles están en la auditoría.
