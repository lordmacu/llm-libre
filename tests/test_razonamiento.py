from llm_libre.razonamiento import RecortadorStream, recortar


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
