"""Excepciones por modelo sobre capacidades_por_defecto.

Un catalogo descubierto no tiene por que ser homogeneo: grok publica 31 ids
de los que 25 hacen tool calls y 6 no (medido el 2026-08-18).
"""
import os

from llm_libre.catalogo import normalizar
from llm_libre.modelos import Capacidades
from llm_libre.proveedores import cargar

DEFECTO = Capacidades(tools=True, vision=False, contexto=128000, max_salida=8192)
CATALOGO = {"data": [{"id": "bueno"}, {"id": "sin-tools"}, {"id": "otro"}]}


def _por_id(rutas):
    return {r.modelo_id: r.capacidades for r in rutas}


def test_sin_excepciones_todos_heredan_el_defecto():
    caps = _por_id(normalizar("grok", CATALOGO, 1, DEFECTO))
    assert all(c.tools for c in caps.values())


def test_la_excepcion_pisa_solo_el_campo_declarado():
    caps = _por_id(normalizar("grok", CATALOGO, 1, DEFECTO,
                              {"sin-tools": {"tools": False}}))
    assert caps["sin-tools"].tools is False
    assert caps["bueno"].tools is True
    # lo NO declarado se hereda entero
    assert caps["sin-tools"].contexto == 128000
    assert caps["sin-tools"].max_salida == 8192
    assert caps["sin-tools"].vision is False


def test_una_excepcion_para_un_id_que_no_existe_no_molesta():
    caps = _por_id(normalizar("grok", CATALOGO, 1, DEFECTO,
                              {"fantasma": {"tools": False}}))
    assert len(caps) == 3 and all(c.tools for c in caps.values())


def test_puede_pisar_varios_campos_a_la_vez():
    caps = _por_id(normalizar("grok", CATALOGO, 1, DEFECTO,
                              {"otro": {"tools": False, "vision": True}}))
    assert caps["otro"].tools is False and caps["otro"].vision is True


def test_el_yaml_declara_las_6_excepciones_de_grok():
    provs = cargar("proveedores.yaml", dict(os.environ))
    grok = [p for p in provs if p.id == "grok"][0]
    assert grok.capacidades_por_defecto.tools is True
    # vision True: el proxy resuelve la imagen por su cuenta
    # (resolve_vision_model manda al VISION_POOL lo que no ve), y 30 de 31
    # rutas leyeron un codigo de 4 digitos en una imagen de 488x232.
    assert grok.capacidades_por_defecto.vision is True
    # Solo la familia imagine-agent-mode queda afuera: son agentes de
    # generacion de imagenes, 0/3 en tool_calls al repetir, y el propio
    # grok_backend documenta que no tienen vision.
    assert set(grok.excepciones) == {
        "imagine-agent-mode", "imagine-agent-mode-dev", "imagine-agent-mode-grok-4-5"}
    assert all(v == {"tools": False, "vision": False} for v in grok.excepciones.values())


def test_minimax_declara_vision_medida():
    from llm_libre.proveedores import rutas_fijas
    provs = cargar("proveedores.yaml", {"MINIMAX_API_KEY": "x"})
    mm = [r for p in provs if p.id == "minimax" for r in rutas_fijas(p)]
    assert mm and mm[0].capacidades.vision is True
