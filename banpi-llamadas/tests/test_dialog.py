"""Pruebas de la máquina de estados del diálogo (sin LLM)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.agent import prompts
from app.agent.dialog import (
    Accion,
    Dialogo,
    Estado,
    Intencion,
    Resultado,
    detectar_intencion,
)

MADRID = ZoneInfo("Europe/Madrid")
AHORA = datetime(2026, 11, 10, 11, 0, tzinfo=MADRID)


def nuevo(nombre: str = "María López") -> Dialogo:
    d = Dialogo(nombre, reloj=lambda: AHORA)
    d.inicio()
    return d


class TestIntenciones:
    @pytest.mark.parametrize(
        "texto,esperada",
        [
            ("No me llaméis más, por favor", Intencion.NO_LLAMAR),
            ("no me vuelvan a llamar", Intencion.NO_LLAMAR),
            ("bórrenme de su lista", Intencion.NO_LLAMAR),
            ("quiero darme de baja", Intencion.NO_LLAMAR),
            ("estoy en la lista Robinson", Intencion.NO_LLAMAR),
            ("No gracias, no me interesa y no me llaméis más", Intencion.NO_LLAMAR),
            ("No me interesa", Intencion.NO_INTERESA),
            ("no, gracias", Intencion.NO_INTERESA),
            ("ya tengo alarma", Intencion.NO_INTERESA),
            ("ahora no puedo, estoy conduciendo", Intencion.MAL_MOMENTO),
            ("llámame más tarde", Intencion.MAL_MOMENTO),
            ("no es buen momento", Intencion.MAL_MOMENTO),
            ("se ha equivocado de número", Intencion.EQUIVOCADO),
            ("no quiero que me grabe", Intencion.OPONE_GRABACION),
            ("¿eres un robot?", Intencion.PREGUNTA_IA),
            ("¿hablo con una persona?", Intencion.PREGUNTA_IA),
            ("¿cuánto cuesta?", Intencion.PREGUNTA_PRECIO),
            ("quiero hablar con una persona", Intencion.PIDE_HUMANO),
            ("Sí, dígame", Intencion.AFIRMATIVO),
            ("vale", Intencion.AFIRMATIVO),
            ("claro que sí", Intencion.AFIRMATIVO),
            ("no", Intencion.NEGATIVO),
            ("tengo un piso en Nervión", Intencion.OTRA),
            ("", Intencion.OTRA),
        ],
    )
    def test_detectar(self, texto: str, esperada: Intencion) -> None:
        assert detectar_intencion(texto) is esperada


class TestFlujo:
    def test_saludo_cumple_avisos(self) -> None:
        d = Dialogo("María López", reloj=lambda: AHORA)
        t = d.inicio()
        assert d.estado is Estado.CONFIRMAR_MOMENTO
        assert "Álex" in t.texto and "inteligencia artificial" in t.texto
        assert "Banpi Domótica" in t.texto and "Sevilla" in t.texto
        assert "grabarse" in t.texto and "no quiere más llamadas" in t.texto
        assert "María" in t.texto

    def test_camino_feliz_hasta_cita(self) -> None:
        d = nuevo()
        t = d.procesar("Sí, dígame")
        assert d.estado is Estado.DESCUBRIMIENTO and t.instruccion and "vivienda" in t.respaldo

        t = d.procesar("Tengo un piso en Triana")
        assert d.datos.inmueble == "vivienda"
        assert "persianas" in t.respaldo

        t = d.procesar("Pues sí, las persianas y la alarma, que me preocupa la seguridad")
        assert d.estado is Estado.CIERRE
        assert set(d.datos.intereses) >= {"persianas", "alarma"}
        assert "visita gratuita" in t.respaldo

        t = d.procesar("Vale, una visita")
        assert d.datos.cita_tipo == "visita" and not t.terminar
        assert "qué día" in t.texto.lower()

        t = d.procesar("El jueves a las cinco")
        assert t.terminar and Accion.GUARDAR_CITA in t.acciones
        assert d.datos.resultado is Resultado.CITA
        assert d.datos.cita_texto == "el jueves 12 a las cinco de la tarde"
        assert "jueves 12" in t.texto

    def test_cita_dia_y_luego_hora(self) -> None:
        d = nuevo()
        d.procesar("sí")
        d.procesar("tengo un local, me interesan las luces")
        d.procesar("el ahorro")
        t = d.procesar("El viernes")
        assert not t.terminar and "qué hora" in t.texto
        t = d.procesar("a las diez y media")
        assert t.terminar and d.datos.cita and d.datos.cita.momento
        assert d.datos.cita.momento.day == 13 and d.datos.cita.momento.hour == 10

    def test_llamada_con_comercial(self) -> None:
        d = nuevo()
        d.procesar("sí")
        d.procesar("casa, la climatización")
        d.procesar("comodidad")
        t = d.procesar("Mejor que me llame un comercial el lunes por la tarde")
        assert t.terminar and d.datos.cita_tipo == "llamada" and d.datos.resultado is Resultado.CITA

    def test_no_llamar_en_cualquier_estado(self) -> None:
        for pasos in ([], ["sí"], ["sí", "tengo un piso"]):
            d = nuevo()
            for p in pasos:
                d.procesar(p)
            t = d.procesar("No me llaméis más")
            assert t.terminar and Accion.MARCAR_NO_LLAMAR in t.acciones
            assert d.datos.resultado is Resultado.NO_LLAMAR

    def test_no_interesa_termina_con_educacion(self) -> None:
        d = nuevo()
        t = d.procesar("No me interesa, gracias")
        assert t.terminar and d.datos.resultado is Resultado.NO_INTERESA
        assert "gracias" in t.texto.lower()
        assert Accion.MARCAR_NO_LLAMAR not in t.acciones

    def test_mal_momento_con_hora(self) -> None:
        d = nuevo()
        t = d.procesar("Ahora no puedo, llámame mañana a las cinco")
        assert t.terminar and Accion.REPROGRAMAR in t.acciones
        assert d.datos.rellamar and d.datos.rellamar.momento
        assert d.datos.rellamar.momento.day == 11 and d.datos.rellamar.momento.hour == 17

    def test_mal_momento_sin_hora_pregunta_cuando(self) -> None:
        d = nuevo()
        t = d.procesar("No")
        assert d.estado is Estado.REPROGRAMAR and not t.terminar
        t = d.procesar("el lunes por la mañana")
        assert t.terminar and d.datos.resultado is Resultado.REPROGRAMAR
        assert d.datos.rellamar_texto == "el lunes 16 a las diez de la mañana"

    def test_reprogramar_sin_concretar(self) -> None:
        d = nuevo()
        d.procesar("no")
        d.procesar("pues no sé")
        t = d.procesar("no sé, ya veremos")
        assert t.terminar and d.datos.resultado is Resultado.REPROGRAMAR

    def test_oposicion_grabacion(self) -> None:
        d = nuevo()
        t = d.procesar("No quiero que se grabe la llamada")
        assert Accion.DESACTIVAR_GRABACION in t.acciones
        assert not d.datos.grabacion_permitida and not t.terminar
        assert "¿Tiene un minuto?" in t.texto

    def test_pregunta_si_es_ia(self) -> None:
        d = nuevo()
        d.procesar("sí")
        t = d.procesar("¿Pero tú eres un robot?")
        assert "inteligencia artificial" in t.texto and d.estado is Estado.DESCUBRIMIENTO

    def test_precio_no_inventa(self) -> None:
        d = nuevo()
        d.procesar("sí")
        t = d.procesar("¿Y cuánto cuesta eso?")
        assert "€" not in t.texto and "euro" not in t.texto.lower()
        assert "comercial" in t.texto and d.estado is Estado.CIERRE

    def test_pide_humano(self) -> None:
        d = nuevo()
        t = d.procesar("Prefiero hablar con una persona")
        assert d.datos.cita_tipo == "llamada" and "qué día" in t.texto.lower()

    def test_equivocado(self) -> None:
        d = nuevo()
        t = d.procesar("Se ha equivocado, aquí no vive ninguna María")
        assert t.terminar and Accion.MARCAR_NO_LLAMAR in t.acciones

    def test_respuesta_ambigua_avanza(self) -> None:
        d = nuevo()
        t = d.procesar("¿De qué se trata?")
        assert d.estado is Estado.DESCUBRIMIENTO and t.instruccion

    def test_negativo_en_cierre(self) -> None:
        d = nuevo()
        d.procesar("sí")
        d.procesar("un piso, las luces")
        d.procesar("ahorrar")
        t = d.procesar("No")
        assert t.terminar and d.datos.resultado is Resultado.SIN_CITA

    def test_silencios(self) -> None:
        d = nuevo()
        assert not d.silencio().terminar
        assert not d.silencio().terminar
        t = d.silencio()
        assert t.terminar and d.datos.resultado is Resultado.SIN_RESPUESTA

    def test_limite_de_turnos(self) -> None:
        d = nuevo()
        d.procesar("sí")
        t = None
        for _ in range(30):
            t = d.procesar("hmm")
            if t.terminar:
                break
        assert t is not None and t.terminar
        assert d.turnos_cliente <= d.s.max_turnos

    def test_tras_fin_no_hace_nada(self) -> None:
        d = nuevo()
        d.procesar("no me interesa")
        assert d.procesar("hola").terminar


class TestLimpiezaLLM:
    def test_recorta_a_dos_frases(self) -> None:
        texto = "Hola. Esto es una prueba. Y otra frase más. Y otra."
        assert prompts.recortar(texto) == "Hola. Esto es una prueba."

    def test_quita_formato_y_prefijos(self) -> None:
        assert prompts.limpiar_frase("Álex: **Claro**, le cuento.") == "Claro, le cuento."

    def test_sustituye_precios(self) -> None:
        for frase in ["Cuesta 300 euros.", "Desde 99€ al mes.", "Tiene un descuento del 20%."]:
            assert not prompts.contiene_precio(prompts.limpiar_frase(frase))

    def test_quita_emojis(self) -> None:
        assert prompts.limpiar_frase("Perfecto \U0001F600") == "Perfecto"
