import os

import pytest

# `llm_libre.principal` ejecuta `estado = crear_estado()` (I/O real: lee
# proveedores.yaml, abre la DB) apenas se importa -- es lo que permite que
# `uvicorn llm_libre.principal:app` funcione como entrypoint. Para que ESE
# primer import no dependa del entorno ambiental de quien corre la suite (que
# puede o no tener un .env cargado), se fija un entorno minimo y sin tocar
# disco ANTES del import: RUTA_DB=":memory:" evita el default de produccion
# (/datos/llm-libre.sqlite3, no escribible fuera de Docker) y una llave
# cualquiera evita el propio fail-fast que este archivo prueba mas abajo.
os.environ["RUTA_DB"] = ":memory:"
os.environ.setdefault("LLM_LIBRE_API_KEYS", "llave-de-arranque-para-tests")

from llm_libre import principal  # noqa: E402


def test_crear_estado_sin_llaves_falla_alto_y_claro(monkeypatch):
    """Sin LLM_LIBRE_API_KEYS el proceso no debe arrancar en silencio: mejor
    un fallo inmediato y explicito que un contenedor que parece sano
    (/health) y rechaza el 100% de las peticiones con 401."""
    monkeypatch.delenv("LLM_LIBRE_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="LLM_LIBRE_API_KEYS"):
        principal.crear_estado()


def test_crear_estado_con_llaves_vacias_o_solo_comas_tambien_falla(monkeypatch):
    # ",, ,"  -> tras el strip() de cada trozo, ninguna llave util: es el
    # mismo caso que la variable ausente, no uno distinto.
    monkeypatch.setenv("LLM_LIBRE_API_KEYS", " , ,  ,")
    with pytest.raises(RuntimeError, match="LLM_LIBRE_API_KEYS"):
        principal.crear_estado()


def test_crear_estado_con_llaves_configuradas_funciona(monkeypatch):
    monkeypatch.setenv("LLM_LIBRE_API_KEYS", "una-llave, otra-llave")
    estado = principal.crear_estado()
    assert estado.llaves == {"una-llave", "otra-llave"}
    assert estado.proveedores  # cargo el YAML real del repo
    assert estado.almacen is not None
    assert estado.proxy is not None
