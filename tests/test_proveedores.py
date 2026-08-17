from pathlib import Path

from llm_libre.modelos import Capacidades
from llm_libre.proveedores import cargar, rutas_fijas

YAML = str(Path(__file__).resolve().parents[1] / "proveedores.yaml")


def test_carga_los_proveedores_registrados():
    # openrouter se saco del registro (decision del operador, 2026-08-17):
    # nunca tuvo OPENROUTER_API_KEY configurada, asi que sus 16 rutas
    # 401-eaban siempre -- casi la mitad del catalogo, gastando cupo de
    # sonda y espacio en /v1/ranking solo para demostrar que seguian
    # muertas. Sigue documentado en docs/providers.md como ejemplo del
    # patron "todo descubierto" con clave opcional.
    # perplexity entro el 2026-08-17 con UNA ruta declarada (`turbo`): su
    # /v1/models publica 124 modelos pero en el flujo anonimo todos caen a
    # turbo, asi que declarar el catalogo seria medir 124 clones.
    ps = cargar(YAML, {"MINIMAX_API_KEY": "mm"})
    assert [p.id for p in ps] == ["chatgpt", "perplexity", "deepseek", "kilo", "minimax"]


def test_perplexity_declara_una_sola_ruta_sin_tools():
    # Verificado contra el proxy real: no hay function calling, asi que una
    # peticion con tools NUNCA debe rutearse aca -- lo garantiza esta
    # declaracion, no el proveedor.
    pplx = next(p for p in cargar(YAML, {}) if p.id == "perplexity")
    rutas = rutas_fijas(pplx)
    assert [r.clave for r in rutas] == ["perplexity/turbo"]
    assert rutas[0].capacidades.tools is False
    assert rutas[0].tier == "gratis"
    assert pplx.base_url.endswith("/v1")   # sin el /v1 todo da 404


def test_kilo_sin_clave_queda_con_clave_vacia_y_sigue_siendo_valido():
    kilo = next(p for p in cargar(YAML, {}) if p.id == "kilo")
    assert kilo.clave == ""
    assert kilo.tier == "gratis"


def test_resuelve_las_claves_desde_el_entorno():
    ps = cargar(YAML, {"MINIMAX_API_KEY": "secreta"})
    assert next(p for p in ps if p.id == "minimax").clave == "secreta"


def test_las_cabeceras_extra_se_conservan(tmp_path):
    # Migrado a un YAML sintetico (revision post-Task-14): antes afirmaba
    # sobre `openrouter`, el unico proveedor real que declaraba
    # cabeceras_extra -- pero se saco del registro (ver
    # test_carga_los_tres_proveedores) y este mecanismo (cargar() propaga
    # cabeceras_extra tal cual) no depende de que ningun proveedor puntual
    # lo use hoy.
    yaml_con_cabeceras = tmp_path / "con_cabeceras.yaml"
    yaml_con_cabeceras.write_text(
        "proveedores:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialecto: openai\n"
        "    base_url: https://suelto.test\n"
        "    modelos_path: /models\n"
        "    cabeceras_extra:\n"
        "      X-Title: llm-libre\n")
    p = cargar(str(yaml_con_cabeceras), {})[0]
    assert p.cabeceras_extra["X-Title"] == "llm-libre"


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
    assert rutas[0].prioridad == 2   # la de minimax en el YAML, no una constante


def test_un_proveedor_gratis_no_tiene_modelos_fijos():
    kilo = next(p for p in cargar(YAML, {}) if p.id == "kilo")
    assert rutas_fijas(kilo) == []


def test_clave_de_solo_espacios_se_normaliza_a_vacia():
    ps = cargar(YAML, {"KILO_API_KEY": "   ", "MINIMAX_API_KEY": "\t\n"})
    kilo = next(p for p in ps if p.id == "kilo")
    minimax = next(p for p in ps if p.id == "minimax")
    assert kilo.clave == ""
    assert minimax.clave == ""


# --- Task 13: chatgpt-proxy, prioridad y base_url_env ---

def test_chatgpt_tiene_prioridad_cero_y_es_gratis():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.tier == "gratis"
    assert chatgpt.prioridad == 0


def test_kilo_queda_en_prioridad_uno():
    kilo = next(p for p in cargar(YAML, {}) if p.id == "kilo")
    assert kilo.prioridad == 1


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


def test_base_url_env_usa_la_variable_de_entorno_si_esta_definida():
    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v1"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1"


def test_base_url_env_cae_al_default_del_yaml_si_la_variable_falta():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.base_url == "http://127.0.0.1:8888/v1"


def test_base_url_env_cae_al_default_si_la_variable_esta_en_blanco():
    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "   "})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "http://127.0.0.1:8888/v1"


def test_un_proveedor_sin_base_url_env_no_se_afecta(tmp_path):
    # kilo no declara base_url_env: una variable de entorno que por
    # casualidad tenga su id no debe pisarle la url.
    kilo = next(p for p in cargar(YAML, {"KILO_URL": "https://otra.test"}) if p.id == "kilo")
    assert kilo.base_url == "https://api.kilo.ai/api/gateway"


# --- Follow-up de Task 13: chatgpt paso de modelos_fijos a DESCUBIERTO
#     (su /v1/models ahora es dinamico), pero sigue sin traer metadatos de
#     capacidad -- por eso declara `capacidades_por_defecto` en vez de
#     `modelos_fijos`. Es un mecanismo GENERAL: cualquier proveedor cuyo
#     catalogo sea igual de desnudo lo puede usar, no es especial de chatgpt. ---

def test_chatgpt_se_descubre_por_modelos_path_no_por_modelos_fijos():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.modelos_path == "/models"
    assert chatgpt.modelos_fijos == []


def test_chatgpt_declara_capacidades_por_defecto_con_tools_false():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.capacidades_por_defecto == Capacidades(
        tools=False, vision=False, contexto=128000, max_salida=8192)


def test_kilo_no_declara_capacidades_por_defecto():
    kilo = next(p for p in cargar(YAML, {}) if p.id == "kilo")
    assert kilo.capacidades_por_defecto is None


def test_minimax_tampoco_declara_capacidades_por_defecto():
    # Sigue siendo el patron viejo: ids Y capacidades declaradas a mano
    # (rutas_fijas), no descubrimiento con defaults.
    minimax = next(p for p in cargar(YAML, {}) if p.id == "minimax")
    assert minimax.capacidades_por_defecto is None
    assert len(rutas_fijas(minimax)) == 1


def test_un_proveedor_sin_capacidades_por_defecto_en_el_yaml_queda_en_none(tmp_path):
    yaml_sin_defaults = tmp_path / "sin_defaults.yaml"
    yaml_sin_defaults.write_text(
        "proveedores:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialecto: openai\n"
        "    base_url: https://suelto.test\n"
        "    modelos_path: /models\n")
    p = cargar(str(yaml_sin_defaults), {})[0]
    assert p.capacidades_por_defecto is None


# --- Revision del follow-up: rungs sin pinnear ---
#
# `rutas_fijas` estampando `prioridad=p.prioridad` no tenia NINGUN test que
# distinguiera "toma la prioridad real del proveedor" de "siempre pone la
# misma constante" -- inerte hoy porque minimax es el unico modelos_fijos y
# es de pago, pero el registro invita a futuros proveedores gratis
# declarados. Se prueba con un YAML sintetico y una prioridad bien
# distintiva (77) para que ninguna coincidencia con un default (100) o con
# el minimax real (2) pueda disfrazar una constante hardcodeada.

# --- Hallazgo 1 de la revision: el desenvuelto de canvas era GLOBAL, y
#     ':::nota{...}' tambien es sintaxis Docusaurus/MDX estandar -- se
#     verificaba en vivo que una ruta de Kilo perdia esas marcas de
#     documentacion legitima. Pasa a ser una declaracion POR PROVEEDOR, misma
#     forma que capacidades_por_defecto: apagada por defecto, prendida solo
#     para chatgpt. ---

def test_chatgpt_declara_desenvuelve_canvas():
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.desenvuelve_canvas is True


def test_kilo_no_desenvuelve_canvas():
    kilo = next(p for p in cargar(YAML, {}) if p.id == "kilo")
    assert kilo.desenvuelve_canvas is False


def test_un_proveedor_sin_desenvuelve_canvas_en_el_yaml_usa_el_default_falso(tmp_path):
    yaml_sin_canvas = tmp_path / "sin_canvas.yaml"
    yaml_sin_canvas.write_text(
        "proveedores:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialecto: openai\n"
        "    base_url: https://suelto.test\n"
        "    modelos_path: /models\n")
    p = cargar(str(yaml_sin_canvas), {})[0]
    assert p.desenvuelve_canvas is False


# --- Hallazgo 2 de la revision (timeout por proveedor): agregado limpio, sin
#     complicar el diseno -- default None significa "usar el TIMEOUT_S global
#     de proxy.py", igual que hoy para todo el que no lo declare. ---

def test_un_proveedor_sin_timeout_declarado_queda_en_none(tmp_path):
    # kilo sigue sin declarar timeout_s -- Task 14 solo le puso uno a
    # chatgpt (ver el test de abajo), a proposito: no se toca el timeout de
    # ningun otro proveedor.
    kilo = next(p for p in cargar(YAML, {}) if p.id == "kilo")
    assert kilo.timeout_s is None


def test_chatgpt_declara_su_propio_timeout_en_el_yaml_real():
    # Task 14: chatgpt tiene prioridad:0 (se prueba primero en CADA pedido) y
    # corre en `blog`, una maquina saturada -- sin timeout_s propio, un
    # cuelgue ahi costaba hasta TIMEOUT_S=90s completo por intento. 45s
    # (ver el comentario en proveedores.yaml para la medicion que lo
    # justifica) acota ese peor caso a la mitad sin bajarle el timeout a
    # nadie mas.
    chatgpt = next(p for p in cargar(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.timeout_s == 45.0


def test_un_proveedor_puede_declarar_su_propio_timeout(tmp_path):
    yaml_con_timeout = tmp_path / "con_timeout.yaml"
    yaml_con_timeout.write_text(
        "proveedores:\n"
        "  - id: lento\n"
        "    tier: gratis\n"
        "    dialecto: openai\n"
        "    base_url: https://lento.test\n"
        "    modelos_path: /models\n"
        "    timeout_s: 20\n")
    p = cargar(str(yaml_con_timeout), {})[0]
    assert p.timeout_s == 20.0


# --- Hallazgo 6 de la revision: CHATGPT_PROXY_URL reemplaza TODO base_url,
#     asi que el operador tiene que acordarse de poner el /v1 el mismo --
#     mismo footgun que ya se arreglo del lado del YAML, sobreviviendo del
#     lado del entorno. Eleccion: NORMALIZAR (agregar el sufijo de ruta que
#     el YAML ya declara como default, si la variable no lo trae) en vez de
#     reventar al arrancar -- hay una unica interpretacion correcta (el
#     sufijo que el propio YAML ya declara), asi que auto-corregir es mas
#     util que tirar abajo un despliegue que ya esta corriendo por un typo
#     recuperable. Se loguea igual, para que quede visible en produccion. ---

def test_base_url_env_agrega_el_sufijo_si_la_variable_no_lo_trae():
    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1"


def test_base_url_env_con_query_string_no_le_astilla_el_sufijo_adentro():
    # LOW de la revision: el sufijo se agregaba por concatenacion de texto,
    # asi que una URL con path vacio pero CON query string terminaba con el
    # sufijo pegado DENTRO del valor de la query
    # ("...:8888?token=abc" -> "...:8888?token=abc/v1"). Hay que parsear la
    # URL y reconstruirla, no concatenar strings.
    chatgpt = next(
        p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888?token=abc"})
        if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1?token=abc"


def test_base_url_env_no_duplica_el_sufijo_si_ya_lo_trae():
    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v1"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1"


def test_base_url_env_con_barra_final_no_duplica_el_sufijo():
    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1"


def test_base_url_env_loguea_cuando_normaliza(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="llm_libre.proveedores"):
        cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888"})
    assert "chatgpt" in caplog.text
    assert "/v1" in caplog.text


def test_base_url_env_sin_sufijo_en_el_default_no_agrega_nada(tmp_path):
    # kilo (sin base_url_env hoy) no se ve afectado por este mecanismo; un
    # proveedor CON base_url_env pero cuyo default no tiene ruta (solo
    # host) tampoco debe agregar nada de la nada.
    yaml_sin_sufijo = tmp_path / "sin_sufijo.yaml"
    yaml_sin_sufijo.write_text(
        "proveedores:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialecto: openai\n"
        "    base_url_env: SUELTO_URL\n"
        "    base_url: https://suelto.test\n"
        "    modelos_path: /models\n")
    p = cargar(str(yaml_sin_sufijo), {"SUELTO_URL": "https://otra.test"})[0]
    assert p.base_url == "https://otra.test"


# --- Re-revision: la regla de normalizacion era DEMASIADO ansiosa -- agregaba
#     el sufijo sin condicion, asi que "...:8888/v2" (una ruta PROPIA del
#     operador, p.ej. un mount de reverse proxy) terminaba en
#     ".../v2/v1/chat/completions". Se aprieta: solo se agrega el sufijo
#     cuando la URL del entorno NO trae ruta propia (vacia o "/"); si ya trae
#     una, se usa TAL CUAL -- "dijeron lo que quisieron decir" -- con un aviso
#     por si fue sin querer, no una correccion silenciosa. ---

def test_base_url_env_con_ruta_propia_no_se_le_pisa_el_sufijo():
    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v2"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v2"


def test_base_url_env_con_ruta_propia_loguea_advertencia_sin_modificar(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="llm_libre.proveedores"):
        chatgpt = next(
            p for p in cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v2"})
            if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v2"
    assert "chatgpt" in caplog.text
    assert "/v2" in caplog.text


def test_base_url_env_con_la_ruta_correcta_no_loguea_nada(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="llm_libre.proveedores"):
        cargar(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v1"})
    assert caplog.text == ""


def test_rutas_fijas_usa_la_prioridad_real_del_proveedor_no_una_constante(tmp_path):
    yaml_prioridad_rara = tmp_path / "prioridad_rara.yaml"
    yaml_prioridad_rara.write_text(
        "proveedores:\n"
        "  - id: pago_futuro\n"
        "    tier: pago\n"
        "    prioridad: 77\n"
        "    dialecto: openai\n"
        "    base_url: https://pago-futuro.test\n"
        "    modelos_fijos:\n"
        "      - id: modelo-x\n"
        "        tools: true\n"
        "        vision: false\n"
        "        contexto: 1000\n"
        "        max_salida: 100\n")
    p = cargar(str(yaml_prioridad_rara), {})[0]
    rutas = rutas_fijas(p)
    assert len(rutas) == 1
    assert rutas[0].prioridad == 77


def test_deepseek_declara_dos_rutas_sin_tools():
    # Verificado contra el proxy real (2026-08-17): manda tools sin reventar pero
    # nunca devuelve tool_calls -- contesta en prosa. Un cliente agentico recibiria
    # texto donde espera una llamada estructurada, asi que esta declaracion es lo
    # unico que impide que el router le mande un pedido con herramientas.
    # deepseek-vision NO se declara: no se verifico entrada de imagenes.
    ds = next(p for p in cargar(YAML, {}) if p.id == "deepseek")
    rutas = rutas_fijas(ds)
    assert [r.modelo_id for r in rutas] == ["deepseek-chat", "deepseek-reasoner"]
    assert all(r.capacidades.tools is False for r in rutas)
    assert all(r.capacidades.vision is False for r in rutas)
    assert ds.base_url.endswith("/v1")
    assert ds.timeout_s == 60.0   # el proof-of-work en WASM suma tiempo variable
