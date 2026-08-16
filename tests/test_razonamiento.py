from llm_libre.razonamiento import (RecortadorCercaCanvas, RecortadorStream,
                                    RecortadorStreamCompuesto, quitar_cercas_canvas,
                                    recortar)


def test_recorta_un_bloque_completo():
    limpio, razon = recortar("<think>pienso mucho</think>hola")
    assert limpio == "hola"
    assert razon == "pienso mucho"


def test_texto_sin_etiquetas_pasa_intacto():
    assert recortar("hola mundo") == ("hola mundo", "")


def test_acepta_las_tres_etiquetas():
    for abre, cierra in (("<think>", "</think>"),
                         ("<thinking>", "</thinking>"),
                         ("<reasoning>", "</reasoning>")):
        assert recortar(f"{abre}x{cierra}listo")[0] == "listo"


def test_una_etiqueta_que_nunca_cierra_no_se_traga_nada_de_texto_util():
    # Todo lo posterior es razonamiento; el cliente recibe lo previo, no vacio ni colgado.
    limpio, razon = recortar("antes<think>y nunca cierro")
    assert limpio == "antes"
    assert razon == "y nunca cierro"


def test_varios_bloques_en_la_misma_respuesta():
    limpio, razon = recortar("a<think>1</think>b<think>2</think>c")
    assert limpio == "abc"
    assert razon == "12"


def test_streaming_con_la_etiqueta_partida_en_cada_posicion_posible():
    entrada = "hola<think>secreto</think>chau"
    esperado = "holachau"
    for corte in range(1, len(entrada)):
        rec = RecortadorStream()
        salida = rec.alimentar(entrada[:corte]) + rec.alimentar(entrada[corte:]) + rec.cerrar()
        assert salida == esperado, f"fallo cortando en {corte}"
        assert rec.razonamiento == "secreto", f"fallo cortando en {corte}"


def test_streaming_caracter_por_caracter():
    entrada = "ab<thinking>zzz</thinking>cd"
    rec = RecortadorStream()
    salida = "".join(rec.alimentar(c) for c in entrada) + rec.cerrar()
    assert salida == "abcd"
    assert rec.razonamiento == "zzz"


def test_streaming_no_retiene_texto_que_no_puede_ser_etiqueta():
    # Emitir cuanto antes: solo se retiene lo que podria ser prefijo de una etiqueta.
    rec = RecortadorStream()
    assert rec.alimentar("hola mundo") == "hola mundo"


def test_streaming_retiene_un_prefijo_ambiguo_hasta_resolverlo():
    rec = RecortadorStream()
    assert rec.alimentar("hola<thi") == "hola"
    assert rec.alimentar("nk>oculto</think>fin") == "fin"
    assert rec.razonamiento == "oculto"


def test_un_menor_que_suelto_no_se_pierde():
    rec = RecortadorStream()
    salida = rec.alimentar("2 < 3") + rec.cerrar()
    assert salida == "2 < 3"


# --- Task 13: cercas de canvas (':::palabra{...}' ... ':::'). A DIFERENCIA de
#     <think>, el contenido de ADENTRO es la respuesta: no se descarta, solo
#     se quitan las dos lineas de marca. ---

_CERCA = (':::writing{variant="document" id="58321" title="Bogota"}\n'
         'Bogota, ciudad de niebla y montana,\n'
         'jamas se cansa de sonar.\n'
         ':::')
_CONTENIDO = 'Bogota, ciudad de niebla y montana,\njamas se cansa de sonar.\n'


def test_quita_la_cerca_de_canvas_conservando_el_contenido():
    assert quitar_cercas_canvas(_CERCA) == _CONTENIDO


def test_recortar_tambien_desenvuelve_la_cerca_de_canvas():
    # recortar() (el punto de entrada del camino no-streaming) debe hacer las
    # dos cosas: seguir recortando <think> Y desenvolver la cerca de canvas.
    limpio, razon = recortar("<think>pienso</think>" + _CERCA)
    assert limpio == _CONTENIDO
    assert razon == "pienso"


def test_una_cerca_de_canvas_sin_nada_alrededor_queda_vacia_de_marcas():
    limpio, razon = recortar(_CERCA)
    assert limpio == _CONTENIDO
    assert razon == ""


def test_contenido_normal_sin_cerca_no_se_toca():
    assert quitar_cercas_canvas("hola mundo, todo normal.") == "hola mundo, todo normal."


def test_texto_con_triple_dos_puntos_que_no_es_cerca_no_se_toca():
    # ':::' aparece pero NO al inicio de una linea como marca real: no debe
    # tocarse ni una letra.
    entrada = "el separador de esta plantilla es ::: en este formato, nada mas."
    assert quitar_cercas_canvas(entrada) == entrada


def test_dos_puntos_triples_al_inicio_de_linea_sin_palabra_no_es_apertura():
    # ':::' seguido de un espacio (no de una palabra) no cumple la sintaxis
    # de apertura: se conserva tal cual.
    entrada = "::: esto no es una cerca\nsegunda linea"
    assert quitar_cercas_canvas(entrada) == entrada


def test_una_cerca_que_nunca_cierra_no_pierde_contenido():
    # A diferencia de <think> sin cerrar (que se traga todo como
    # razonamiento), una cerca de canvas sin cerrar NO descarta nada: solo se
    # quita la marca de apertura, que ya se identifico con certeza.
    entrada = ':::writing{title="x"}\nhola\nmundo sin cerrar'
    assert quitar_cercas_canvas(entrada) == "hola\nmundo sin cerrar"


def test_streaming_con_la_cerca_partida_en_cada_posicion_posible():
    entrada = _CERCA
    for corte in range(1, len(entrada)):
        rec = RecortadorCercaCanvas()
        salida = rec.alimentar(entrada[:corte]) + rec.alimentar(entrada[corte:]) + rec.cerrar()
        assert salida == _CONTENIDO, f"fallo cortando en {corte}"


def test_streaming_caracter_por_caracter_de_la_cerca():
    rec = RecortadorCercaCanvas()
    salida = "".join(rec.alimentar(c) for c in _CERCA) + rec.cerrar()
    assert salida == _CONTENIDO


def test_streaming_de_texto_con_triple_dos_puntos_incrustado_no_se_toca():
    entrada = "el separador es ::: y sigue el texto normal despues de eso."
    for corte in range(1, len(entrada)):
        rec = RecortadorCercaCanvas()
        salida = rec.alimentar(entrada[:corte]) + rec.alimentar(entrada[corte:]) + rec.cerrar()
        assert salida == entrada, f"fallo cortando en {corte}"


def test_streaming_de_una_cerca_que_nunca_cierra_no_pierde_contenido():
    entrada = ':::writing{title="x"}\nhola\nmundo sin cerrar'
    esperado = "hola\nmundo sin cerrar"
    for corte in range(1, len(entrada)):
        rec = RecortadorCercaCanvas()
        salida = rec.alimentar(entrada[:corte]) + rec.alimentar(entrada[corte:]) + rec.cerrar()
        assert salida == esperado, f"fallo cortando en {corte}"


def test_recortador_stream_compuesto_encadena_think_y_canvas_en_streaming():
    # El camino de streaming de verdad (proxy.py) usa un solo objeto: primero
    # se descarta el razonamiento, despues se desenvuelve la cerca sobre lo
    # que YA es contenido visible.
    entrada = "<think>mmm</think>" + _CERCA
    rec = RecortadorStreamCompuesto()
    salida = ""
    for corte in range(0, len(entrada), 3):
        salida += rec.alimentar(entrada[corte:corte + 3])
    salida += rec.cerrar()
    assert salida == _CONTENIDO
    assert rec.razonamiento == "mmm"
