from llm_libre.cliente import armar_peticion
from llm_libre.proveedores import Proveedor


def _prov(clave="", extra=None):
    return Proveedor(id="kilo", tier="gratis", dialecto="openai",
                     base_url="https://api.kilo.ai/api/gateway", clave=clave,
                     modelos_path="/models", cabeceras_extra=extra or {}, modelos_fijos=[])


def test_sin_clave_no_manda_authorization():
    # El tier anonimo de Kilo depende de esto: mandar un Bearer vacio lo rompe.
    url, cabeceras, _ = armar_peticion(_prov(), {"model": "x"})
    assert url == "https://api.kilo.ai/api/gateway/chat/completions"
    assert "Authorization" not in cabeceras


def test_clave_de_solo_espacios_no_manda_authorization():
    # La normalizacion en cargar() convierte espacios a ""; aqui verificamos
    # que armar_peticion tampoco manda Authorization para claves vacías.
    _, cabeceras, _ = armar_peticion(_prov(clave="   "), {"model": "x"})
    assert "Authorization" not in cabeceras


def test_con_clave_manda_bearer():
    _, cabeceras, _ = armar_peticion(_prov(clave="abc"), {"model": "x"})
    assert cabeceras["Authorization"] == "Bearer abc"


def test_incluye_las_cabeceras_extra():
    _, cabeceras, _ = armar_peticion(_prov(extra={"X-Title": "llm-libre"}), {"model": "x"})
    assert cabeceras["X-Title"] == "llm-libre"


def test_reescribe_el_modelo_al_id_real_del_proveedor():
    _, _, cuerpo = armar_peticion(_prov(), {"model": "auto"}, modelo_real="poolside/x:free")
    assert cuerpo["model"] == "poolside/x:free"


def test_cuerpo_es_copia_somera_no_muta_original():
    """Verifica que el cuerpo devuelto es una copia somera del original.

    Las claves de nivel superior (como 'model') son independientes después de
    armar_peticion, pero las estructuras anidadas (como 'messages') se comparten
    intencionalmente con el original y permanecen inmutables durante failover.
    """
    messages = [{"role": "user", "content": "hola"}]
    original = {"model": "auto", "messages": messages}
    url, cabeceras, devuelto = armar_peticion(_prov(), original, modelo_real="real")

    # (a) Las claves de nivel superior son independientes: modelo_real reescribió el devuelto
    assert devuelto["model"] == "real"
    assert original["model"] == "auto"

    # (b) Las estructuras anidadas se comparten intencionalmente (identidad)
    assert devuelto["messages"] is original["messages"]
    assert devuelto["messages"] is messages
