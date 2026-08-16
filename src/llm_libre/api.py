from llm_libre.modelos import Pedido

PERFILES = {"rapido", "balanceado", "potente"}


def interpretar_pedido(cuerpo: dict) -> Pedido:
    modelo_pedido = (cuerpo.get("model") or "auto").strip()
    modelo, perfil = None, "balanceado"
    requiere_tools = bool(cuerpo.get("tools"))
    requiere_vision = False

    if modelo_pedido == "auto" or modelo_pedido.startswith("auto:"):
        sufijo = modelo_pedido[5:] if ":" in modelo_pedido else ""
        if sufijo in PERFILES:
            perfil = sufijo
        elif sufijo == "tools":
            requiere_tools = True
        elif sufijo == "vision":
            requiere_vision = True
    else:
        modelo = modelo_pedido

    exigidas = set(cuerpo.get("x_requiere") or [])
    return Pedido(
        modelo=modelo,
        requiere_tools=requiere_tools or "tools" in exigidas,
        requiere_vision=requiere_vision or "vision" in exigidas,
        min_contexto=int(cuerpo.get("x_min_contexto") or 0),
        perfil=perfil,
        permitir_pago=bool(cuerpo.get("x_permitir_pago", True)),
    )
