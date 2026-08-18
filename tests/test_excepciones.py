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
    # La correccion del 2026-08-18: NINGUNA ruta de grok ve imagenes.
    assert grok.capacidades_por_defecto.vision is False
    assert set(grok.excepciones) == {
        "claude-3-opus", "gpt-4o", "grok-4-1-thinking-1129",
        "imagine-agent-mode", "imagine-agent-mode-dev", "imagine-agent-mode-grok-4-5"}
    assert all(v == {"tools": False} for v in grok.excepciones.values())


def test_minimax_declara_vision_medida():
    from llm_libre.proveedores import rutas_fijas
    provs = cargar("proveedores.yaml", {"MINIMAX_API_KEY": "x"})
    mm = [r for p in provs if p.id == "minimax" for r in rutas_fijas(p)]
    assert mm and mm[0].capacidades.vision is True
