from pathlib import Path

from llm_libre.proveedores import cargar, rutas_fijas

YAML = str(Path(__file__).resolve().parents[1] / "proveedores.yaml")


def test_carga_los_tres_proveedores():
    ps = cargar(YAML, {"OPENROUTER_API_KEY": "or", "MINIMAX_API_KEY": "mm"})
    assert [p.id for p in ps] == ["kilo", "openrouter", "minimax"]


def test_kilo_sin_clave_queda_con_clave_vacia_y_sigue_siendo_valido():
    kilo = cargar(YAML, {})[0]
    assert kilo.clave == ""
    assert kilo.tier == "gratis"


def test_resuelve_las_claves_desde_el_entorno():
    ps = cargar(YAML, {"MINIMAX_API_KEY": "secreta"})
    assert next(p for p in ps if p.id == "minimax").clave == "secreta"


def test_las_cabeceras_extra_se_conservan():
    orouter = next(p for p in cargar(YAML, {}) if p.id == "openrouter")
    assert orouter.cabeceras_extra["X-Title"] == "llm-libre"


def test_los_modelos_fijos_se_vuelven_rutas_de_pago():
    minimax = next(p for p in cargar(YAML, {}) if p.id == "minimax")
    rutas = rutas_fijas(minimax)
    assert len(rutas) == 1
    assert rutas[0].clave == "minimax/MiniMax-M3"
    assert rutas[0].tier == "pago"
    assert rutas[0].capacidades.tools is True
    assert rutas[0].capacidades.vision is False


def test_un_proveedor_gratis_no_tiene_modelos_fijos():
    kilo = cargar(YAML, {})[0]
    assert rutas_fijas(kilo) == []
