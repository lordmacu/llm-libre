from llm_libre.modelos import Capacidades, Metricas, Pedido, Ruta
from llm_libre.router import ordenar


def r(modelo, proveedor="kilo", tier="gratis", tools=True, vision=False, contexto=100000):
    return Ruta(proveedor, modelo, tier,
                Capacidades(tools=tools, vision=vision, contexto=contexto, max_salida=4096))


def m(calidad=0.8, confiabilidad=0.9, ttft=500, cooldown=0.0, medida_en=1000.0):
    return Metricas(calidad, confiabilidad, ttft, cooldown, medida_en)


def test_descarta_las_rutas_que_no_soportan_tools_cuando_se_piden():
    rutas = [r("con:free", tools=True), r("sin:free", tools=False)]
    salida = ordenar(rutas, {}, Pedido(requiere_tools=True), ahora=0.0)
    assert [x.modelo_id for x in salida] == ["con:free"]


def test_descarta_las_rutas_sin_vision_cuando_se_pide():
    rutas = [r("ve:free", vision=True), r("ciego:free", vision=False)]
    salida = ordenar(rutas, {}, Pedido(requiere_vision=True), ahora=0.0)
    assert [x.modelo_id for x in salida] == ["ve:free"]


def test_descarta_las_rutas_con_contexto_insuficiente():
    rutas = [r("grande:free", contexto=200000), r("chico:free", contexto=8000)]
    salida = ordenar(rutas, {}, Pedido(min_contexto=100000), ahora=0.0)
    assert [x.modelo_id for x in salida] == ["grande:free"]


def test_ordena_por_puntaje_descendente():
    rutas = [r("malo:free"), r("bueno:free")]
    metricas = {"kilo/malo:free": m(calidad=0.3), "kilo/bueno:free": m(calidad=0.95)}
    salida = ordenar(rutas, metricas, Pedido(), ahora=0.0)
    assert [x.modelo_id for x in salida] == ["bueno:free", "malo:free"]


def test_las_rutas_de_pago_van_siempre_al_final_aunque_puntuen_mejor():
    rutas = [r("MiniMax-M3", proveedor="minimax", tier="pago"), r("flojo:free")]
    metricas = {"minimax/MiniMax-M3": m(calidad=1.0, confiabilidad=1.0, ttft=100),
                "kilo/flojo:free": m(calidad=0.2, confiabilidad=0.3, ttft=5000)}
    salida = ordenar(rutas, metricas, Pedido(), ahora=0.0)
    assert [x.tier for x in salida] == ["gratis", "pago"]


def test_permitir_pago_falso_saca_las_rutas_de_pago():
    rutas = [r("MiniMax-M3", proveedor="minimax", tier="pago"), r("g:free")]
    salida = ordenar(rutas, {}, Pedido(permitir_pago=False), ahora=0.0)
    assert [x.tier for x in salida] == ["gratis"]


def test_excluye_las_rutas_en_cooldown_pero_no_las_vencidas():
    rutas = [r("castigada:free"), r("vencida:free")]
    metricas = {"kilo/castigada:free": m(cooldown=500.0),
                "kilo/vencida:free": m(cooldown=50.0)}
    salida = ordenar(rutas, metricas, Pedido(), ahora=100.0)
    assert [x.modelo_id for x in salida] == ["vencida:free"]


def test_un_modelo_explicito_filtra_pero_conserva_los_dos_proveedores():
    rutas = [r("comun:free", proveedor="kilo"),
             r("comun:free", proveedor="openrouter"),
             r("otro:free", proveedor="kilo")]
    salida = ordenar(rutas, {}, Pedido(modelo="comun:free"), ahora=0.0)
    assert len(salida) == 2
    assert {x.proveedor for x in salida} == {"kilo", "openrouter"}


def test_sin_candidatas_devuelve_lista_vacia():
    salida = ordenar([r("sin:free", tools=False)], {}, Pedido(requiere_tools=True), ahora=0.0)
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
    salida = ordenar(rutas, metricas, Pedido(), ahora=0.0)
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
    salida = ordenar(rutas, metricas, Pedido(), ahora=0.0)
    assert [x.modelo_id for x in salida] == ["medida:free", "nueva:free"]


def test_una_ruta_nunca_sondeada_sigue_en_la_cadena_para_poder_medirse():
    rutas = [r("nueva:free"), r("medida:free")]
    metricas = {"kilo/medida:free": m(calidad=0.9, medida_en=1000.0)}
    salida = ordenar(rutas, metricas, Pedido(), ahora=0.0)
    assert "nueva:free" in [x.modelo_id for x in salida]


def test_entre_dos_nunca_sondeadas_sigue_mandando_el_puntaje():
    rutas = [x for x in (r("lenta:free"), r("rapida:free"))]
    metricas = {"kilo/lenta:free": m(ttft=5000, medida_en=None),
                "kilo/rapida:free": m(ttft=100, medida_en=None)}
    salida = ordenar(rutas, metricas, Pedido(), ahora=0.0)
    assert [x.modelo_id for x in salida] == ["rapida:free", "lenta:free"]
