from llm_libre.modelos import Capacidades, Metricas, Pedido, Ruta
from llm_libre.router import order_routes


def r(modelo, proveedor="kilo", tier="gratis", tools=True, vision=False, contexto=100000):
    return Ruta(proveedor, modelo, tier,
                Capacidades(tools=tools, vision=vision, contexto=contexto, max_salida=4096))


def m(calidad=0.8, confiabilidad=0.9, ttft=500, cooldown=0.0, medida_en=1000.0):
    return Metricas(calidad, confiabilidad, ttft, cooldown, medida_en)


def test_descarta_las_rutas_que_no_soportan_tools_cuando_se_piden():
    rutas = [r("con:free", tools=True), r("sin:free", tools=False)]
    salida = order_routes(rutas, {}, Pedido(requiere_tools=True), now=0.0)
    assert [x.modelo_id for x in salida] == ["con:free"]


def test_descarta_las_rutas_sin_vision_cuando_se_pide():
    rutas = [r("ve:free", vision=True), r("ciego:free", vision=False)]
    salida = order_routes(rutas, {}, Pedido(requiere_vision=True), now=0.0)
    assert [x.modelo_id for x in salida] == ["ve:free"]


def test_descarta_las_rutas_con_contexto_insuficiente():
    rutas = [r("grande:free", contexto=200000), r("chico:free", contexto=8000)]
    salida = order_routes(rutas, {}, Pedido(min_contexto=100000), now=0.0)
    assert [x.modelo_id for x in salida] == ["grande:free"]


def test_ordena_por_puntaje_descendente():
    rutas = [r("malo:free"), r("bueno:free")]
    metricas = {"kilo/malo:free": m(calidad=0.3), "kilo/bueno:free": m(calidad=0.95)}
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert [x.modelo_id for x in salida] == ["bueno:free", "malo:free"]


def test_las_rutas_de_pago_van_siempre_al_final_aunque_puntuen_mejor():
    rutas = [r("MiniMax-M3", proveedor="minimax", tier="pago"), r("flojo:free")]
    metricas = {"minimax/MiniMax-M3": m(calidad=1.0, confiabilidad=1.0, ttft=100),
                "kilo/flojo:free": m(calidad=0.2, confiabilidad=0.3, ttft=5000)}
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert [x.tier for x in salida] == ["gratis", "pago"]


def test_permitir_pago_falso_saca_las_rutas_de_pago():
    rutas = [r("MiniMax-M3", proveedor="minimax", tier="pago"), r("g:free")]
    salida = order_routes(rutas, {}, Pedido(permitir_pago=False), now=0.0)
    assert [x.tier for x in salida] == ["gratis"]


def test_excluye_las_rutas_en_cooldown_pero_no_las_vencidas():
    rutas = [r("castigada:free"), r("vencida:free")]
    metricas = {"kilo/castigada:free": m(cooldown=500.0),
                "kilo/vencida:free": m(cooldown=50.0)}
    salida = order_routes(rutas, metricas, Pedido(), now=100.0)
    assert [x.modelo_id for x in salida] == ["vencida:free"]


def test_un_modelo_explicito_filtra_pero_conserva_los_dos_proveedores():
    rutas = [r("comun:free", proveedor="kilo"),
             r("comun:free", proveedor="openrouter"),
             r("otro:free", proveedor="kilo")]
    salida = order_routes(rutas, {}, Pedido(modelo="comun:free"), now=0.0)
    assert len(salida) == 2
    assert {x.proveedor for x in salida} == {"kilo", "openrouter"}


def test_sin_candidatas_devuelve_lista_vacia():
    salida = order_routes([r("sin:free", tools=False)], {}, Pedido(requiere_tools=True), now=0.0)
    assert salida == []


def test_una_ruta_sin_metricas_usa_las_neutras_y_no_ceros():
    # kilo/conocida:free puntua 0.3*0.5*factor_latencia(1500) = 0.075 en balanceado.
    # kilo/nueva:free (sin metricas) puntua con METRICAS_NEUTRAS (0.6, 0.8, 1500):
    # 0.6*0.8*factor_latencia(1500) = 0.24, asi que le gana a la conocida y queda primera.
    # Si el fallback fuera Metricas(0,0,0,0) puntuaria 0 y quedaria de ULTIMA: el orden
    # se invertiria y este assert fallaria, que es justo lo que este test debe detectar.
    #
    # `medida_en=None` en la conocida es deliberado (fix round 3, B2b): las dos
    # rutas quedan igual de "no medidas" para que lo que decida el orden sea el
    # puntaje, que es lo unico que este test quiere proteger. El criterio de
    # medida-antes-que-supuesta tiene sus propios tests, mas abajo.
    rutas = [r("conocida:free"), r("nueva:free")]
    metricas = {"kilo/conocida:free": m(calidad=0.3, confiabilidad=0.5, ttft=1500,
                                        medida_en=None)}
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert [x.modelo_id for x in salida] == ["nueva:free", "conocida:free"]


# --- Fix round 3, B2 (Blocking), mitad (b): una ruta que nunca paso por la
#     bateria de calidad carga un supuesto neutro (0.6), no una medicion. No
#     puede preferirse por encima de una ruta cuya calidad SI se midio: eso es
#     lo que dejaba a un modelo recien aparecido -- rapido y sin evaluar --
#     arriba de todo hasta 25 h. Tiene que seguir siendo alcanzable, eso si, o
#     nunca se mediria. ---

def test_una_ruta_nunca_sondeada_va_despues_de_una_con_calidad_medida():
    rutas = [r("nueva:free"), r("medida:free")]
    metricas = {
        # La medida puntua PEOR en balanceado (0.35*0.9*f(500) = 0.24) que la
        # nueva con los neutros (0.6*0.9*f(200) = 0.48): si el orden fuera solo
        # por puntaje, la nueva ganaria. Debe perder igual.
        "kilo/medida:free": m(calidad=0.35, ttft=500, medida_en=1000.0),
        "kilo/nueva:free": m(calidad=0.6, ttft=200, medida_en=None),
    }
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert [x.modelo_id for x in salida] == ["medida:free", "nueva:free"]


def test_una_ruta_nunca_sondeada_sigue_en_la_cadena_para_poder_medirse():
    rutas = [r("nueva:free"), r("medida:free")]
    metricas = {"kilo/medida:free": m(calidad=0.9, medida_en=1000.0)}
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert "nueva:free" in [x.modelo_id for x in salida]


def test_entre_dos_nunca_sondeadas_sigue_mandando_el_puntaje():
    rutas = [x for x in (r("lenta:free"), r("rapida:free"))]
    metricas = {"kilo/lenta:free": m(ttft=5000, medida_en=None),
                "kilo/rapida:free": m(ttft=100, medida_en=None)}
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert [x.modelo_id for x in salida] == ["rapida:free", "lenta:free"]


# --- Task 13: `prioridad`, un concepto DISTINTO de `tier` y de `perfil`. ---

def _rp(modelo, prioridad, proveedor="kilo", tier="gratis", tools=True):
    return Ruta(proveedor, modelo, tier,
                Capacidades(tools=tools, vision=False, contexto=100000, max_salida=4096),
                prioridad=prioridad)


def test_la_prioridad_ordena_dentro_del_mismo_tier_por_encima_del_puntaje():
    # gpt-5 (prioridad 0) puntua PEOR que el gratis de siempre (prioridad 1,
    # el default) y aun asi tiene que ganarle: la prioridad decide antes que
    # el puntaje dentro del mismo tier.
    rutas = [_rp("chatgpt:free", 0, proveedor="chatgpt"), _rp("normal:free", 1)]
    metricas = {"chatgpt/chatgpt:free": m(calidad=0.3, confiabilidad=0.3, ttft=3000),
                "kilo/normal:free": m(calidad=0.99, confiabilidad=0.99, ttft=50)}
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert [x.modelo_id for x in salida] == ["chatgpt:free", "normal:free"]


def test_la_prioridad_no_rompe_el_invariante_de_pago_al_final():
    # EL CASO QUE IMPORTA: una ruta de PAGO con prioridad 0 (la mas alta
    # posible) y un puntaje perfecto contra una ruta gratis mediocre con la
    # prioridad default. La plata es la razon: pago va al final SIEMPRE, la
    # prioridad no puede comprar ese lugar.
    rutas = [_rp("MiniMax-M3", 0, proveedor="minimax", tier="pago"),
             _rp("mediocre:free", 100)]
    metricas = {"minimax/MiniMax-M3": m(calidad=1.0, confiabilidad=1.0, ttft=50),
                "kilo/mediocre:free": m(calidad=0.2, confiabilidad=0.3, ttft=5000)}
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert [x.tier for x in salida] == ["gratis", "pago"]
    assert [x.modelo_id for x in salida] == ["mediocre:free", "MiniMax-M3"]


# --- Hallazgo 2 de la revision de Task 13: `prioridad` no tenia escape para
#     una ruta persistentemente rota -- con chatgpt en prioridad 0, order_routes()
#     la seguia poniendo primero SIEMPRE, sin mirar su salud, mientras el
#     cooldown (round 8: lo dispara un 429 de inmediato o una SONDA que
#     confirma la ruta rota, ver proxy.Proxy._sospechar) sea la UNICA salida
#     de esa trampa: se filtra ANTES de que la prioridad importe. ---

# --- Hallazgo 5 de la revision: `prioridad` y "nunca medida" en la clave de
#     orden se podian intercambiar de posicion sin que ningun test existente
#     lo notara -- justo el rung que decide el rollout real: en un deploy
#     nuevo, TODAS las rutas de chatgpt arrancan sin medir mientras Kilo ya
#     carga mediciones de la base de produccion. Si el orden fuera
#     (no-medida, prioridad) en vez de (prioridad, no-medida), Kilo (medido)
#     le ganaria a chatgpt (prioridad 0, sin medir) el dia del deploy -- lo
#     opuesto de lo que prioridad:0 promete. ---

def test_la_prioridad_decide_antes_que_medida_vs_no_medida():
    # chatgpt: prioridad 0 (la mejor) pero NUNCA MEDIDO (calidad_medida_en
    # None, el estado real el dia de un deploy). kilo: prioridad 1 (peor)
    # pero CON medicion real -- el estado real de una base de produccion ya
    # rodando. Si el orden fuera (no-medida, prioridad), kilo ganaria.
    rutas = [_rp("chatgpt:free", 0, proveedor="chatgpt"), _rp("normal:free", 1)]
    metricas = {
        "chatgpt/chatgpt:free": m(medida_en=None),
        "kilo/normal:free": m(medida_en=1000.0),
    }
    salida = order_routes(rutas, metricas, Pedido(), now=0.0)
    assert [x.modelo_id for x in salida] == ["chatgpt:free", "normal:free"]


def test_una_ruta_en_cooldown_se_salta_aunque_tenga_la_maxima_prioridad():
    rutas = [_rp("chatgpt:free", 0, proveedor="chatgpt"), _rp("normal:free", 1)]
    metricas = {
        # chatgpt tiene la mejor prioridad Y el mejor puntaje -- y aun asi no
        # puede ganar mientras este en cooldown.
        "chatgpt/chatgpt:free": m(calidad=0.99, confiabilidad=0.99, ttft=50,
                                  cooldown=500.0),
        "kilo/normal:free": m(calidad=0.2, confiabilidad=0.3, ttft=5000),
    }
    salida = order_routes(rutas, metricas, Pedido(), now=100.0)
    assert [x.modelo_id for x in salida] == ["normal:free"]


def test_un_modelo_explicito_se_sirve_aunque_no_sea_el_de_mayor_prioridad():
    # "Elegir modelo a mano ya funciona y debe seguir igual" (brief, punto 4):
    # pedir un id real evita el ordenamiento por completo, prioridad incluida.
    rutas = [_rp("prioritario:free", 0, proveedor="chatgpt"),
             _rp("elegido-a-mano:free", 1)]
    salida = order_routes(rutas, {}, Pedido(modelo="elegido-a-mano:free"), now=0.0)
    assert [x.modelo_id for x in salida] == ["elegido-a-mano:free"]


# --- Follow-up de Task 13: chatgpt paso a catalogo DESCUBIERTO con
#     capacidades declaradas (tools:false SIEMPRE). Se fija el comportamiento
#     que ya cubria el filtro generico de tools/id explicito, con una ruta
#     con la FORMA real de una ruta de chatgpt (prioridad 0, tools=False). ---

def test_una_peticion_con_tools_nunca_enruta_a_chatgpt():
    chatgpt = _rp("gpt-5-3-mini", 0, proveedor="chatgpt", tools=False)
    kilo = _rp("con-tools:free", 1, tools=True)
    salida = order_routes([chatgpt, kilo], {}, Pedido(requiere_tools=True), now=0.0)
    assert [x.modelo_id for x in salida] == ["con-tools:free"]


def test_una_peticion_sin_tools_prefiere_chatgpt_por_su_prioridad():
    # La contracara: sin exigir tools, chatgpt sigue yendo primero pese a
    # tools=False, porque nadie lo esta pidiendo.
    chatgpt = _rp("gpt-5-3-mini", 0, proveedor="chatgpt", tools=False)
    kilo = _rp("normal:free", 1, tools=True)
    salida = order_routes([chatgpt, kilo], {}, Pedido(), now=0.0)
    assert [x.modelo_id for x in salida] == ["gpt-5-3-mini", "normal:free"]


def test_un_id_de_chatgpt_elegido_a_mano_sigue_ruteando_directo():
    chatgpt = _rp("gpt-5-3-mini", 0, proveedor="chatgpt", tools=False)
    kilo = _rp("otro:free", 1)
    salida = order_routes([chatgpt, kilo], {}, Pedido(modelo="gpt-5-3-mini"), now=0.0)
    assert [x.modelo_id for x in salida] == ["gpt-5-3-mini"]


def test_una_ruta_de_chatgpt_descubierta_de_verdad_rutea_directo_al_pedirla():
    # Integra catalog.normalize (donde chatgpt DESCUBRE sus ids, con
    # capacidades_por_defecto) con router.order_routes: prueba el camino
    # completo, no rutas armadas a mano.
    from llm_libre.catalog import normalize
    defaults = Capacidades(tools=False, vision=False, contexto=128000, max_salida=8192)
    descubiertas = normalize(
        "chatgpt",
        {"data": [{"id": "gpt-5-3-mini", "description": "GPT-5.3 Mini"},
                  {"id": "gpt-5-5", "description": "GPT-5.5"}]},
        priority=0, default_capabilities=defaults)
    salida = order_routes(descubiertas, {}, Pedido(modelo="gpt-5-3-mini"), now=0.0)
    assert [x.modelo_id for x in salida] == ["gpt-5-3-mini"]
