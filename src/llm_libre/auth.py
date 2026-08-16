from collections import defaultdict


class LimitadorPorLlave:
    """Ventana deslizante simple, en memoria. Depende de correr con UN solo worker."""

    def __init__(self, por_minuto: int):
        self.por_minuto = por_minuto
        self._marcas: dict[str, list[float]] = defaultdict(list)

    def permitir(self, llave: str, ahora: float) -> bool:
        marcas = [t for t in self._marcas[llave] if ahora - t < 60.0]
        if len(marcas) >= self.por_minuto:
            self._marcas[llave] = marcas
            return False
        marcas.append(ahora)
        self._marcas[llave] = marcas
        return True
