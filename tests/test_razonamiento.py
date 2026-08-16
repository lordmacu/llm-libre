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


def test_recortar_desenvuelve_la_cerca_solo_si_el_llamador_lo_pide():
    # recortar() (el punto de entrada del camino no-streaming) debe hacer las
    # dos cosas cuando desenvolver_canvas=True: seguir recortando <think> Y
    # desenvolver la cerca de canvas.
    limpio, razon = recortar("<think>pienso</think>" + _CERCA, desenvolver_canvas=True)
    assert limpio == _CONTENIDO
    assert razon == "pienso"


def test_una_cerca_de_canvas_sin_nada_alrededor_queda_vacia_de_marcas():
    limpio, razon = recortar(_CERCA, desenvolver_canvas=True)
    assert limpio == _CONTENIDO
    assert razon == ""


def test_recortar_por_defecto_no_toca_las_cercas():
    # El default (desenvolver_canvas=False) es a proposito: ':::nota{...}'
    # tambien es sintaxis Docusaurus/MDX estandar. Un proveedor que no
    # declara desenvuelve_canvas (Kilo, OpenRouter) no debe perder esas
    # marcas -- <think> SI se sigue recortando siempre, es ortogonal.
    limpio, razon = recortar("<think>pienso</think>" + _CERCA)
    assert limpio == _CERCA
    assert razon == "pienso"


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


# --- Hallazgo 4 de la revision de Task 13: cuatro mutaciones del automata de
#     canvas sobrevivian la suite entera. Cada una es un cambio de
#     comportamiento real, verificado ejecutando la mutacion a mano contra
#     estos tests exactos (no una suposicion). ---

def test_cerrar_descarta_la_marca_de_cierre_no_la_emite():
    # Toda cerca de este archivo termina en ':::' SIN salto de linea final,
    # asi que la marca de cierre real solo la consume cerrar() (nunca el
    # bucle principal de alimentar(), que solo actua sobre lineas completas
    # con '\n'). Si cerrar() emitiera `resto` en vez de "" al confirmar el
    # cierre, la cerca quedaria con un ':::' colgando al final.
    salida = quitar_cercas_canvas(_CERCA)
    assert salida == _CONTENIDO
    assert not salida.endswith(":::")


def test_sin_seguimiento_de_inicio_de_linea_no_se_pierde_texto_de_respuesta():
    # ':::' incrustado a mitad de una linea (no al inicio) NUNCA es una
    # marca. Sin el seguimiento de "ya se descarto esta linea"
    # (_en_inicio_de_linea), un ':' aislado que llega justo despues de un
    # flush se re-evalua como si fuera un inicio de linea nuevo, y
    # "a:::b" tragaria su propio final como si fuera una apertura -- la
    # perdida silenciosa de texto que esta area existe para evitar.
    entrada = "el rango a:::b\nsigue\n"
    rec = RecortadorCercaCanvas()
    salida = "".join(rec.alimentar(c) for c in entrada) + rec.cerrar()
    assert salida == entrada


def test_el_patron_de_apertura_exige_que_toda_la_linea_sea_la_marca():
    # ":::nota" es un prefijo valido, pero "esto no es cerca" arruina el
    # resto de la linea: la marca de apertura exige la LINEA ENTERA, no un
    # prefijo. Si el patron se aflojara (p.ej. dejara de anclar el final),
    # esta linea se tragaria entera en vez de conservarse.
    entrada = ":::nota esto no es cerca\nsegunda linea\n"
    assert quitar_cercas_canvas(entrada) == entrada


def test_el_patron_de_cierre_exige_la_linea_exacta_no_cualquier_triple_dos_puntos():
    # Una linea ":::otro" DENTRO de una cerca abierta no es el cierre -- el
    # cierre es EXACTAMENTE ':::' sola. Si el patron de cierre se aflojara
    # (p.ej. aceptara cualquier linea que empiece con ':::'), esta cerca se
    # cortaria antes de tiempo y "resto real" quedaria AFUERA, tratado como
    # si nunca hubiera estado adentro (en este caso ademas se perderia,
    # porque quedaria como una linea de apertura invalida).
    entrada = (':::writing{title="x"}\n'
              'primera linea\n'
              ':::otro\n'
              'resto real\n'
              ':::')
    esperado = 'primera linea\n:::otro\nresto real\n'
    assert quitar_cercas_canvas(entrada) == esperado


def test_recortador_stream_compuesto_encadena_think_y_canvas_cuando_se_pide():
    # El camino de streaming de verdad (proxy.py) usa un solo objeto: primero
    # se descarta el razonamiento, despues se desenvuelve la cerca sobre lo
    # que YA es contenido visible -- pero solo si el proveedor de la ruta lo
    # declara (desenvolver_canvas=True).
    entrada = "<think>mmm</think>" + _CERCA
    rec = RecortadorStreamCompuesto(desenvolver_canvas=True)
    salida = ""
    for corte in range(0, len(entrada), 3):
        salida += rec.alimentar(entrada[corte:corte + 3])
    salida += rec.cerrar()
    assert salida == _CONTENIDO
    assert rec.razonamiento == "mmm"


def test_recortador_stream_compuesto_por_defecto_no_toca_las_cercas():
    entrada = "<think>mmm</think>" + _CERCA
    rec = RecortadorStreamCompuesto()
    salida = ""
    for corte in range(0, len(entrada), 3):
        salida += rec.alimentar(entrada[corte:corte + 3])
    salida += rec.cerrar()
    assert salida == _CERCA
    assert rec.razonamiento == "mmm"
