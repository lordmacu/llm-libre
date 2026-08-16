ABRE = ("<think>", "<thinking>", "<reasoning>")
CIERRA = ("</think>", "</thinking>", "</reasoning>")


class RecortadorStream:
    """Separa el razonamiento del texto util sobre un flujo de deltas.

    El caso dificil es que las etiquetas llegan partidas entre chunks: hay que
    retener la cola del buffer que TODAVIA podria ser el prefijo de una etiqueta,
    y emitir todo lo demas de inmediato.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._dentro = False
        self.razonamiento = ""

    def alimentar(self, delta: str) -> str:
        self._buf += delta
        salida = ""
        while True:
            etiquetas = CIERRA if self._dentro else ABRE
            i, tag = _primera(self._buf, etiquetas)
            if i == -1:
                corte = len(self._buf) - _cola_ambigua(self._buf, etiquetas)
                trozo, self._buf = self._buf[:corte], self._buf[corte:]
                if self._dentro:
                    self.razonamiento += trozo
                else:
                    salida += trozo
                return salida
            trozo = self._buf[:i]
            if self._dentro:
                self.razonamiento += trozo
            else:
                salida += trozo
            self._buf = self._buf[i + len(tag):]
            self._dentro = not self._dentro

    def cerrar(self) -> str:
        """Cierra el flujo. Un bloque sin cerrar se considera razonamiento entero."""
        resto, self._buf = self._buf, ""
        if self._dentro:
            self.razonamiento += resto
            return ""
        return resto


def recortar(texto: str) -> tuple[str, str]:
    rec = RecortadorStream()
    limpio = rec.alimentar(texto) + rec.cerrar()
    return limpio, rec.razonamiento


def _primera(s: str, etiquetas: tuple[str, ...]) -> tuple[int, str]:
    mejor, cual = -1, ""
    for t in etiquetas:
        i = s.find(t)
        if i != -1 and (mejor == -1 or i < mejor):
            mejor, cual = i, t
    return mejor, cual


def _cola_ambigua(s: str, etiquetas: tuple[str, ...]) -> int:
    """Largo del sufijo de s que aun podria completar alguna etiqueta."""
    maximo = max(len(t) for t in etiquetas) - 1
    for largo in range(min(maximo, len(s)), 0, -1):
        cola = s[-largo:]
        if any(t.startswith(cola) for t in etiquetas):
            return largo
    return 0
