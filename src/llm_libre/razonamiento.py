import re

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
    return quitar_cercas_canvas(limpio), rec.razonamiento


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


# --- Cercas de canvas: ':::palabra{...atributos...}' ... ':::' ------------
#
# chatgpt-proxy (Task 13) filtra el modo "canvas" de ChatGPT al `content`,
# marcas y todo (verificado contra el proxy real, 2026-08-16):
#
#   :::writing{variant="document" id="58321" title="Bogota"}
#   Bogota, ciudad de niebla y montana,
#   ...
#   :::
#
# DIFERENCIA CLAVE respecto de <think>: adentro de la cerca esta la
# RESPUESTA, no algo para descartar. Solo se quitan las dos lineas de marca
# (la de apertura, con sus atributos, y el ':::' de cierre); todo lo demas
# -- incluido el contenido "adentro" -- se conserva tal cual, caracter por
# caracter, igual que el texto que nunca estuvo dentro de una cerca.
#
# Por eso este es un automata mas simple que RecortadorStream: no hay nada
# que acumular en un segundo canal (no existe un ".razonamiento" aca), solo
# decidir que DOS lineas completas no se emiten.
_RE_APERTURA_CANVAS = re.compile(r"^:::[A-Za-z][\w-]*(\{[^\n]*\})?$")
_RE_CIERRE_CANVAS = re.compile(r"^:::$")


def _puede_ser_apertura_canvas(s: str) -> bool:
    """True si `s` (una linea todavia incompleta, sin '\\n') es un prefijo
    valido de una linea de apertura ':::palabra{...}'. Sirve para decidir,
    ANTES de que llegue el resto de la linea, si hace falta seguir
    reteniendola o si ya se puede soltar como texto comun."""
    marca = ":::"
    n = min(len(s), len(marca))
    if s[:n] != marca[:n]:
        return False
    resto = s[len(marca):]
    if resto == "":
        return True
    if not resto[0].isalpha():
        return False
    i = 1
    while i < len(resto) and (resto[i].isalnum() or resto[i] in "_-"):
        i += 1
    if i == len(resto):
        return True   # todo lo visto hasta ahora es la "palabra"; falta ver que sigue
    return resto[i] == "{"   # ya empezaron los atributos: todo lo demas es ambiguo


def _puede_ser_cierre_canvas(s: str) -> bool:
    """True si `s` todavia podria completarse en la linea de cierre ':::'."""
    return ":::".startswith(s)


class RecortadorCercaCanvas:
    """Desenvuelve cercas de canvas sobre un flujo de deltas, conservando el
    contenido interior -- lo opuesto de RecortadorStream, que lo descarta.

    Como en RecortadorStream, la marca puede llegar partida entre chunks; a
    diferencia de RecortadorStream, aca la unidad de deteccion es la LINEA
    completa (hasta el '\\n'), porque la marca de apertura tiene forma
    variable (sus atributos). Para no demorar el streaming de texto normal,
    apenas una linea deja de poder ser una marca se suelta de inmediato
    (`_en_inicio_de_linea` pasa a False) y el resto de ESA linea ya no se
    vuelve a evaluar -- evita el falso positivo de un ':::' incrustado a
    mitad de una oracion, que nunca esta al inicio de una linea real.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._dentro = False
        self._en_inicio_de_linea = True

    def alimentar(self, delta: str) -> str:
        self._buf += delta
        salida = ""
        while True:
            if not self._en_inicio_de_linea:
                # Esta linea ya se descarto como candidata: pasa tal cual
                # hasta el proximo '\n' (inclusive), sin volver a mirarla.
                i = self._buf.find("\n")
                if i == -1:
                    salida += self._buf
                    self._buf = ""
                    return salida
                salida += self._buf[:i + 1]
                self._buf = self._buf[i + 1:]
                self._en_inicio_de_linea = True
                continue

            i = self._buf.find("\n")
            if i == -1:
                candidato = self._buf
                puede = (_puede_ser_cierre_canvas(candidato) if self._dentro
                         else _puede_ser_apertura_canvas(candidato))
                if puede:
                    return salida   # se retiene, a la espera del resto de la linea
                salida += self._buf
                self._buf = ""
                self._en_inicio_de_linea = False
                return salida

            linea, self._buf = self._buf[:i], self._buf[i + 1:]
            if self._dentro:
                if _RE_CIERRE_CANVAS.fullmatch(linea):
                    self._dentro = False
                else:
                    salida += linea + "\n"
                continue
            if _RE_APERTURA_CANVAS.fullmatch(linea):
                self._dentro = True
            else:
                salida += linea + "\n"
            continue

    def cerrar(self) -> str:
        """Cierra el flujo. Una cerca sin cerrar NO se traga su contenido
        (a diferencia de <think>): solo la marca de apertura, ya confirmada,
        se pierde -- todo lo demas ya salio por `alimentar`."""
        resto, self._buf = self._buf, ""
        if self._en_inicio_de_linea:
            if self._dentro and _RE_CIERRE_CANVAS.fullmatch(resto):
                self._dentro = False
                return ""
            if not self._dentro and _RE_APERTURA_CANVAS.fullmatch(resto):
                self._dentro = True
                return ""
        return resto


def quitar_cercas_canvas(texto: str) -> str:
    rec = RecortadorCercaCanvas()
    return rec.alimentar(texto) + rec.cerrar()


class RecortadorStreamCompuesto:
    """Encadena RecortadorStream (recorte de razonamiento, se descarta) con
    RecortadorCercaCanvas (cercas de canvas, se conserva el interior) para
    que el productor de streaming (proxy.py) hable con un solo objeto. El
    orden importa: primero se recorta el razonamiento, despues se
    desenvuelve la cerca sobre lo que YA es contenido visible."""

    def __init__(self) -> None:
        self._pensamiento = RecortadorStream()
        self._canvas = RecortadorCercaCanvas()

    @property
    def razonamiento(self) -> str:
        return self._pensamiento.razonamiento

    def alimentar(self, delta: str) -> str:
        return self._canvas.alimentar(self._pensamiento.alimentar(delta))

    def cerrar(self) -> str:
        return self._canvas.alimentar(self._pensamiento.cerrar()) + self._canvas.cerrar()
