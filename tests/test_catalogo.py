import json
from pathlib import Path

from llm_libre.catalogo import normalizar

FIXTURES = Path(__file__).parent / "fixtures"


def _cargar(nombre):
    return json.loads((FIXTURES / nombre).read_text())


def test_descarta_los_modelos_de_pago():
    datos = {"data": [
        {"id": "caro/modelo", "pricing": {"prompt": "0.0000015"},
         "architecture": {"output_modalities": ["text"]}},
    ]}
    assert normalizar("kilo", datos) == []


def test_descarta_modelos_gratis_que_no_devuelven_solo_texto():
    # google/lyria-* tiene pricing 0 pero es un modelo de MUSICA:
    # output_modalities = ["text", "audio"]. Filtrar por precio no alcanza.
    datos = {"data": [
        {"id": "google/lyria-3-pro-preview", "pricing": {"prompt": "0"},
         "context_length": 1048576,
         "architecture": {"input_modalities": ["text", "image"],
                          "output_modalities": ["text", "audio"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    assert normalizar("kilo", datos) == []


def test_acepta_entrada_multimodal_mientras_la_salida_sea_texto():
    datos = {"data": [
        {"id": "nvidia/nemotron-omni:free", "pricing": {"prompt": "0"},
         "context_length": 256000,
         "architecture": {"input_modalities": ["text", "audio", "image", "video"],
                          "output_modalities": ["text"]},
         "supported_parameters": ["tools", "tool_choice"],
         "top_provider": {"max_completion_tokens": 8192}},
    ]}
    rutas = normalizar("kilo", datos)
    assert len(rutas) == 1
    assert rutas[0].capacidades.vision is True
    assert rutas[0].capacidades.tools is True
    assert rutas[0].capacidades.contexto == 256000
    assert rutas[0].capacidades.max_salida == 8192
    assert rutas[0].tier == "gratis"


def test_detecta_ausencia_de_tools():
    datos = {"data": [
        {"id": "sin/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens", "temperature"]},
    ]}
    assert normalizar("kilo", datos)[0].capacidades.tools is False


def test_precio_ilegible_se_trata_como_de_pago():
    datos = {"data": [{"id": "raro/modelo", "pricing": {},
                       "architecture": {"output_modalities": ["text"]}}]}
    assert normalizar("kilo", datos) == []


def test_contra_el_catalogo_real_de_kilo():
    rutas = normalizar("kilo", _cargar("kilo_models.json"))
    assert len(rutas) > 5
    assert all(r.tier == "gratis" for r in rutas)
    assert all(r.proveedor == "kilo" for r in rutas)
    assert not any("lyria" in r.modelo_id for r in rutas)
    assert any(r.capacidades.tools for r in rutas)


def test_contra_el_catalogo_real_de_openrouter():
    rutas = normalizar("openrouter", _cargar("openrouter_models.json"))
    assert len(rutas) > 5
    assert not any("lyria" in r.modelo_id for r in rutas)
