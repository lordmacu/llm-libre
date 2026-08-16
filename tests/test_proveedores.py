from pathlib import Path

from llm_libre.proveedores import cargar, rutas_fijas

YAML = str(Path(__file__).resolve().parents[1] / "proveedores.yaml")


def test_carga_los_cuatro_proveedores():
    ps = cargar(YAML, {"OPENROUTER_API_KEY": "or", "MINIMAX_API_KEY": "mm"})
    assert [p.id for p in ps] == ["chatgpt", "kilo", "openrouter", "minimax"]


def test_kilo_sin_clave_queda_con_clave_vacia_y_sigue_siendo_valido():
    kilo = next(p for p in cargar(YAML, {}) if p.id == "kilo")
    assert kilo.clave == ""
    assert kilo.tier == "gratis"


def test_resuelve_las_claves_desde_el_entorno():
    ps = cargar(YAML, {"MINIMAX_API_KEY": "secreta"})
    assert next(p for p in ps if p.id == "minimax").clave == "secreta"


def test_las_cabeceras_extra_se_conservan():
    orouter = next(p for p in cargar(YAML, {}) if p.id == "openrouter")
    assert orouter.cabeceras_extra["X-Title"] == "llm-libre"


def test_los_modelos_fijos_se_vuelven_rutas_de_pago():
    minimax = next(p for p in cargar(YAML, {}) if p.id == "minimax")
    rutas = rutas_fijas(minimax)
    assert len(rutas) == 1
    assert rutas[0].clave == "minimax/MiniMax-M3"
    assert rutas[0].tier == "pago"
    assert rutas[0].capacidades.tools is True
    assert rutas[0].capacidades.vision is False
    assert rutas[0].capacidades.contexto == 128000
    assert rutas[0].capacidades.max_salida == 32768


def test_un_proveedor_gratis_no_tiene_modelos_fijos():
    kilo = next(p for p in cargar(YAML, {}) if p.id == "kilo")
    assert rutas_fijas(kilo) == []


def test_clave_de_solo_espacios_se_normaliza_a_vacia():
    ps = cargar(YAML, {"KILO_API_KEY": "   ", "OPENROUTER_API_KEY": "\t\n"})
    kilo = next(p for p in ps if p.id == "kilo")
    orouter = next(p for p in ps if p.id == "openrouter")
    assert kilo.clave == ""
    assert orouter.clave == ""


# --- Task 13: chatgpt-proxy, prioridad y base_url_env ---

def test_chatgpt_tiene_prioridad_cero_y_es_gratis():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.tier == "gratis"
    assert chatgpt.prioridad == 0


def test_kilo_y_openrouter_quedan_en_prioridad_uno():
    ps = cargar(YAML, {})
    kilo = next(p for p in ps if p.id == "kilo")
    orouter = next(p for p in ps if p.id == "openrouter")
    assert kilo.prioridad == 1
    assert orouter.prioridad == 1


def test_minimax_queda_en_prioridad_dos():
    minimax = next(p for p in cargar(YAML, {}) if p.id == "minimax")
    assert minimax.prioridad == 2


def test_un_proveedor_sin_prioridad_en_el_yaml_usa_el_default_cien(tmp_path):
    yaml_sin_prioridad = tmp_path / "sin_prioridad.yaml"
    yaml_sin_prioridad.write_text(
        "proveedores:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialecto: openai\n"
        "    base_url: https://suelto.test\n"
        "    modelos_path: /models\n")
    p = cargar(str(yaml_sin_prioridad), {})[0]
    assert p.prioridad == 100


def test_los_cinco_modelos_de_chatgpt_declaran_tools_false():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    rutas = rutas_fijas(chatgpt)
    assert len(rutas) == 5
    assert all(r.capacidades.tools is False for r in rutas)
    assert {r.modelo_id for r in rutas} == {
        "gpt-5-6", "gpt-5-5", "gpt-5-6-mini", "gpt-5-5-mini", "gpt-5-3-mini"}


def test_chatgpt_no_declara_el_modelo_auto_ni_los_alias_legacy():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    ids = {m["id"] for m in chatgpt.modelos_fijos}
    assert "auto" not in ids
    assert not ids & {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"}


def test_rutas_fijas_de_chatgpt_heredan_su_prioridad():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    rutas = rutas_fijas(chatgpt)
    assert all(r.prioridad == 0 for r in rutas)


def test_base_url_env_usa_la_variable_de_entorno_si_esta_definida():
    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888"


def test_base_url_env_cae_al_default_del_yaml_si_la_variable_falta():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.base_url == "http://127.0.0.1:8888"


def test_base_url_env_cae_al_default_si_la_variable_esta_en_blanco():
    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "   "})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "http://127.0.0.1:8888"


def test_un_proveedor_sin_base_url_env_no_se_afecta(tmp_path):
    # kilo no declara base_url_env: una variable de entorno que por
    # casualidad tenga su id no debe pisarle la url.
    kilo = next(p for p in cargar(YAML, {"KILO_URL": "https://otra.test"}) if p.id == "kilo")
    assert kilo.base_url == "https://api.kilo.ai/api/gateway"
