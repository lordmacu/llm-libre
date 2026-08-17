import json
import logging
from pathlib import Path

from llm_libre.catalogo import normalizar
from llm_libre.modelos import Capacidades

FIXTURES = Path(__file__).parent / "fixtures"

_DEFAULTS_CHATGPT = Capacidades(tools=False, vision=False, contexto=128000, max_salida=8192)


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


def test_normalizar_estampa_la_prioridad_del_proveedor():
    datos = {"data": [
        {"id": "sin/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    rutas = normalizar("kilo", datos, prioridad=1)
    assert rutas[0].prioridad == 1


def test_normalizar_sin_prioridad_declarada_usa_el_default_cien():
    datos = {"data": [
        {"id": "sin/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    assert normalizar("kilo", datos)[0].prioridad == 100


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


# --- Fix round 3, B2 (Blocking): en una instalacion nueva TODAS las rutas
#     arrancan con la calidad neutra, asi que el unico discriminador es el
#     ttft -- y lo mas rapido del pozo es un clasificador de seguridad que
#     contesta "User Safety: safe" a cualquier cosa. Se descarta por lo que el
#     PROPIO proveedor publica de el en /models (name/description), que el
#     normalizador tiraba a la basura. Sigue siendo descubrimiento: no hay ni
#     un id cableado. ---

def _modelo(mid, nombre="Un Modelo", descripcion="Un modelo de chat general."):
    return {"id": mid, "name": nombre, "description": descripcion,
            "pricing": {"prompt": "0"}, "context_length": 4096,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools"],
            "top_provider": {"max_completion_tokens": 4096}}


def test_descarta_un_guardrail_por_su_propia_descripcion():
    datos = {"data": [_modelo(
        "vendor/lo-que-sea:free", "Vendor: Lo Que Sea (free)",
        "A compact 4B-parameter multimodal guardrail model. It moderates both "
        "inputs to and responses from LLMs.")]}
    assert normalizar("kilo", datos) == []


def test_descarta_un_clasificador_y_un_reranker_por_su_descripcion():
    datos = {"data": [
        _modelo("v/a:free", "V: A", "A safety classifier for prompts."),
        _modelo("v/b:free", "V: B", "A reranker for retrieval pipelines."),
        _modelo("v/c:free", "V: C", "A text embeddings model."),
    ]}
    assert normalizar("kilo", datos) == []


def test_descarta_los_meta_routers_por_su_propia_descripcion():
    # No son un modelo: son un sorteo entre otros modelos. Medirlos en el
    # ranking mide una ruleta, y esconden que ruta sirvio de verdad.
    datos = {"data": [
        _modelo("kilo-auto/free", "Auto Free",
                "Rotates through available free models. Limited capability."),
        _modelo("openrouter/free", "Free Models Router",
                "The simplest way to get free inference. openrouter/free is a "
                "router that selects free models at random."),
    ]}
    assert normalizar("kilo", datos) == []


def test_no_descarta_un_modelo_de_chat_normal():
    datos = {"data": [_modelo(
        "nvidia/nemotron-3-super:free", "NVIDIA: Nemotron 3 Super (free)",
        "A 120B-parameter open hybrid MoE model, activating just 12B parameters "
        "for maximum compute efficiency and accuracy in complex multi-agent "
        "applications.")]}
    assert [r.modelo_id for r in normalizar("kilo", datos)] == ["nvidia/nemotron-3-super:free"]


def test_un_modelo_sin_name_ni_description_no_se_descarta():
    # Los campos son opcionales: un proveedor que no los publique no debe
    # perder todo su catalogo.
    datos = {"data": [{"id": "pelado:free", "pricing": {"prompt": "0"},
                       "context_length": 4096,
                       "architecture": {"output_modalities": ["text"]}}]}
    assert [r.modelo_id for r in normalizar("kilo", datos)] == ["pelado:free"]


def test_el_catalogo_real_de_kilo_ya_no_trae_el_clasificador_ni_los_meta_routers():
    ids = {r.modelo_id for r in normalizar("kilo", _cargar("kilo_models.json"))}
    assert "nvidia/nemotron-3.5-content-safety:free" not in ids
    assert "kilo-auto/free" not in ids
    assert "openrouter/free" not in ids
    # ...y los modelos de chat de verdad siguen ahi.
    assert "nvidia/nemotron-3-super-120b-a12b:free" in ids
    assert "poolside/laguna-s-2.1:free" in ids
    assert len(ids) == 11


def test_el_catalogo_real_de_openrouter_ya_no_trae_el_clasificador_ni_el_meta_router():
    ids = {r.modelo_id for r in normalizar("openrouter", _cargar("openrouter_models.json"))}
    assert "nvidia/nemotron-3.5-content-safety:free" not in ids
    assert "openrouter/free" not in ids
    assert "google/gemma-4-31b-it:free" in ids
    assert "openai/gpt-oss-20b:free" in ids
    assert len(ids) == 15


# --- Fix round 3, ALSO: `m["id"]` sin guard. Una entrada de catalogo sin id
#     reventaba con KeyError; sondeo.py se lo tragaba entero y el catalogo de
#     ESE proveedor quedaba congelado para siempre, sin un solo log. ---

def test_una_entrada_sin_id_se_omite_sin_reventar_y_se_loguea(caplog):
    datos = {"data": [
        {"pricing": {"prompt": "0"}, "architecture": {"output_modalities": ["text"]}},
        {"id": "bueno:free", "pricing": {"prompt": "0"},
         "architecture": {"output_modalities": ["text"]}},
    ]}
    with caplog.at_level(logging.WARNING, logger="llm_libre.catalogo"):
        rutas = normalizar("kilo", datos)
    assert [r.modelo_id for r in rutas] == ["bueno:free"]
    assert "sin 'id'" in caplog.text


def test_una_entrada_que_no_es_un_objeto_se_omite_sin_reventar(caplog):
    # Un error de auth servido con 200 deja a normalizar() iterando strings.
    with caplog.at_level(logging.WARNING, logger="llm_libre.catalogo"):
        assert normalizar("kilo", {"error": "unauthorized"}) == []
    assert caplog.records


# --- Follow-up de Task 13: chatgpt-proxy paso a ser un catalogo DESCUBIERTO
#     (su /v1/models ahora es dinamico, con TTL cache), pero sigue sin traer
#     metadatos de capacidad -- solo id/object/created/owned_by/description.
#     `capacidades_por_defecto` es el mecanismo GENERAL para ese patron:
#     ids descubiertos, capacidades declaradas una sola vez para todos. ---

def test_chatgpt_descubre_los_cinco_ids_reales_del_fixture():
    rutas = normalizar("chatgpt", _cargar("chatgpt_models.json"),
                       capacidades_por_defecto=_DEFAULTS_CHATGPT)
    assert len(rutas) == 5
    assert {r.modelo_id for r in rutas} == {
        "gpt-5-5", "gpt-5-6", "gpt-5-3-mini", "gpt-5-5-mini", "gpt-5-6-mini"}


def test_chatgpt_descarta_los_alias_legacy_por_su_propia_description():
    rutas = normalizar("chatgpt", _cargar("chatgpt_models.json"),
                       capacidades_por_defecto=_DEFAULTS_CHATGPT)
    ids = {r.modelo_id for r in rutas}
    assert not ids & {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"}


def test_chatgpt_descarta_auto_por_ser_un_id_reservado():
    rutas = normalizar("chatgpt", _cargar("chatgpt_models.json"),
                       capacidades_por_defecto=_DEFAULTS_CHATGPT)
    assert "auto" not in {r.modelo_id for r in rutas}


# --- INFO de la revision round 6: IDS_RESERVADOS solo excluia el id LITERAL
#     "auto", no los alias compuestos que api.py TAMBIEN trata como
#     reservados ("auto:rapido", "auto:potente", "auto:tools",
#     "auto:vision" -- ver ALIAS en api.py e interpretar_pedido, que resuelve
#     CUALQUIER id que empiece con "auto:" como alias, nunca como id
#     literal). Un proveedor que publicara un modelo real llamado, por
#     ejemplo, "auto:rapido" creaba una ruta PERMANENTEMENTE inalcanzable:
#     ningun pedido con `model: "auto:rapido"` llega nunca a
#     `pedido.modelo == "auto:rapido"`, porque interpretar_pedido lo
#     resuelve antes como perfil "rapido" con modelo=None. ---

def test_se_descartan_tambien_los_alias_compuestos_de_auto():
    rutas = normalizar("prov", [
        {"id": "auto:rapido", "name": "Choca con el alias compuesto"},
        {"id": "auto:potente", "name": "Choca con el alias compuesto"},
        {"id": "auto:tools", "name": "Choca con el alias compuesto"},
        {"id": "auto:vision", "name": "Choca con el alias compuesto"},
        {"id": "un-modelo-real", "name": "Modelo real"},
    ], capacidades_por_defecto=_DEFAULTS_CHATGPT)
    ids = {r.modelo_id for r in rutas}
    assert ids == {"un-modelo-real"}


def test_un_modelo_nuevo_del_proxy_aparece_sin_tocar_el_yaml():
    # El fixture "con_modelo_nuevo" simula que manana el backend real de
    # ChatGPT agrega un modelo (gpt-5-7) que hoy no existe: tiene que
    # aparecer solo, sin que nadie edite proveedores.yaml ni este test.
    rutas = normalizar("chatgpt", _cargar("chatgpt_models_con_modelo_nuevo.json"),
                       capacidades_por_defecto=_DEFAULTS_CHATGPT)
    ids = {r.modelo_id for r in rutas}
    assert "gpt-5-7" in ids
    assert len(ids) == 6


def test_las_capacidades_por_defecto_se_aplican_a_cada_id_descubierto():
    rutas = normalizar("chatgpt", _cargar("chatgpt_models.json"),
                       capacidades_por_defecto=_DEFAULTS_CHATGPT)
    assert len(rutas) == 5
    assert all(r.capacidades == _DEFAULTS_CHATGPT for r in rutas)
    assert all(r.capacidades.tools is False for r in rutas)   # obligatorio: tools -> HTTP 500


def test_capacidades_por_defecto_saltea_precio_y_modalidad_de_salida():
    # Un proveedor que declara defaults esta afirmando lo que su catalogo NO
    # puede decir: una entrada sin "pricing" (normalmente => "de pago", se
    # descarta) y con output_modalities de musica (normalmente se descarta)
    # igual se acepta, porque esos dos chequeos se saltean en este modo.
    datos = {"data": [
        {"id": "modelo-sin-metadatos", "description": "Un Modelo"},
    ]}
    rutas = normalizar("chatgpt", datos, capacidades_por_defecto=_DEFAULTS_CHATGPT)
    assert [r.modelo_id for r in rutas] == ["modelo-sin-metadatos"]
    assert rutas[0].capacidades == _DEFAULTS_CHATGPT


def test_capacidades_por_defecto_no_desactiva_el_filtro_de_especialidad():
    # Solo se saltean precio/modalidad -- lo que el brief pidio. El filtro de
    # especialidad (guardrails, clasificadores, meta-routers) sigue activo
    # como defensa: un proveedor con defaults tambien puede exponer algo asi.
    datos = {"data": [
        {"id": "guardrail-1", "description": "A content safety guardrail model."},
    ]}
    assert normalizar("chatgpt", datos, capacidades_por_defecto=_DEFAULTS_CHATGPT) == []


def test_un_proveedor_sin_capacidades_por_defecto_mantiene_el_comportamiento_de_siempre():
    # Kilo/OpenRouter no declaran defaults: normalizar() sin ese argumento
    # (o con None explicito) tiene que dar EXACTAMENTE lo mismo que antes de
    # este cambio -- precio, modalidad de salida y especialidad siguen
    # filtrando como siempre.
    datos = {"data": [
        {"id": "caro/modelo", "pricing": {"prompt": "0.0000015"},
         "architecture": {"output_modalities": ["text"]}},
    ]}
    assert normalizar("kilo", datos, capacidades_por_defecto=None) == []
    assert normalizar("kilo", datos) == []


def test_contra_el_catalogo_real_de_chatgpt_proxy():
    # El fixture completo, tal cual lo devuelve /v1/models hoy (verificado
    # 2026-08-16): 10 entradas -- 5 modelos reales, 4 alias legacy, "auto".
    rutas = normalizar("chatgpt", _cargar("chatgpt_models.json"),
                       capacidades_por_defecto=_DEFAULTS_CHATGPT)
    assert len(rutas) == 5
    assert all(r.proveedor == "chatgpt" for r in rutas)
    assert all(r.tier == "gratis" for r in rutas)
