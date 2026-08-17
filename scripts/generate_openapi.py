#!/usr/bin/env python3
"""Regenera `openapi.json` (raiz del repo) desde la app FastAPI real -- Task 14.

Construye la MISMA `crear_app` que sirve `/docs` en produccion, con un
`Estado` minimo en memoria (sin red, sin `.env`, sin `/datos`) igual que los
tests -- lo unico que le importa a `app.openapi()` es el arbol de rutas y
los `openapi_extra`/`responses` que cada endpoint declara, no los datos que
haya cargados. Asi `openapi.json` no puede desincronizarse de lo que el
servicio de verdad expone: es literalmente `app.openapi()`, no una copia
mantenida a mano.

Uso (ver README.md):

    venv/bin/python scripts/generate_openapi.py
"""

import json
from pathlib import Path

import httpx

from llm_libre.almacen import Almacen
from llm_libre.api import Estado, crear_app
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import Proxy

RAIZ = Path(__file__).resolve().parents[1]


def _estado_minimo() -> Estado:
    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    # Un solo proveedor de relleno: alcanza para que crear_app arme la app --
    # el catalogo/proveedores reales no afectan en nada al schema de OpenAPI.
    proveedores = {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test",
                                     "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={})))
    return Estado(almacen=almacen, proxy=Proxy(proveedores, almacen, http),
                  llaves={"placeholder"}, tope_pago_diario=200)


def main() -> None:
    app = crear_app(_estado_minimo())
    schema = app.openapi()
    destino = RAIZ / "openapi.json"
    destino.write_text(json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
                       encoding="utf-8")
    print(f"escrito {destino} ({len(schema.get('paths', {}))} paths)")


if __name__ == "__main__":
    main()
