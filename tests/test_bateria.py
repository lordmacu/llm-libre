from llm_libre.bateria import CASOS, evaluar


def _respuesta(texto):
    return {"choices": [{"message": {"role": "assistant", "content": texto}}]}


def test_hay_al_menos_cinco_casos():
    assert len(CASOS) >= 5


def test_todos_los_casos_piden_pocos_tokens():
    # Sondear consume la misma cuota gratis que el servicio necesita para trabajar.
    for c in CASOS:
        assert c.cuerpo["max_tokens"] <= 256, c.nombre


def test_el_caso_de_aritmetica_acepta_la_respuesta_correcta():
    caso = next(c for c in CASOS if c.nombre == "aritmetica")
    assert caso.verificar(_respuesta("12")) is True
    assert caso.verificar(_respuesta("13")) is False


def test_el_caso_de_formato_castiga_el_preambulo_de_razonamiento():
    # nvidia/nemotron responde "Here's a thinking process:..." a un pedido de una palabra.
    caso = next(c for c in CASOS if c.nombre == "formato")
    assert caso.verificar(_respuesta("hola")) is True
    assert caso.verificar(_respuesta("Here's a thinking process: ... hola")) is False


def test_el_caso_de_json_valida_contra_el_schema():
    caso = next(c for c in CASOS if c.nombre == "json")
    assert caso.verificar(_respuesta('{"ciudad":"Bogota","pais":"Colombia"}')) is True
    assert caso.verificar(_respuesta('{"ciudad":"Bogota"}')) is False
    assert caso.verificar(_respuesta("no es json")) is False


def test_el_caso_de_tools_exige_la_llamada_correcta():
    caso = next(c for c in CASOS if c.nombre == "tools")
    bueno = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city":"Bogota"}'}}]}}]}
    assert caso.verificar(bueno) is True
    assert caso.verificar(_respuesta("el clima esta lindo")) is False


def test_evaluar_cuenta_los_casos_pasados():
    assert evaluar([True, False, True]) == (2, 3)
