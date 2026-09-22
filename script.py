"""
Análisis exploratorio de datos — Reseñas de vinos (WineEnthusiast)
Trabajo Práctico 2 — Seminario de Programación para Ciencia de Datos

Autor: Lucas Llano

Requisitos: pandas, numpy, matplotlib, seaborn, scipy
Uso:        python script.py
"""

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

# La consigna pide cargar "winemag-data-130k-v2.csv", pero el archivo que se
# entregó junto al enunciado es un .xlsx. No es un Excel con datos tabulados:
# es ese mismo CSV que pasó por una importación mal configurada en Excel.
# La sección 1 documenta y revierte ese daño.
RUTA_DATOS = "Entregable 2 - Documentación extra.xlsx"

# Cantidad de campos que debe tener cada línea del CSV original:
# 13 columnas de datos + 1 columna de índice sin nombre que quedó al exportar.
CAMPOS_POR_LINEA = 14


# ===========================================================================
# 1. CARGA DE DATOS
# ===========================================================================

def _reconstruir_lineas(ruta):
    """Deshace el daño de la importación a Excel y devuelve el CSV como texto.

    El archivo entregado tiene tres problemas, todos originados en el mismo
    hecho: el CSV se abrió en Excel con una configuración regional que usa el
    punto y coma como separador de campos.

    Problema 1 — Delimitador equivocado.
        Excel recorrió cada línea buscando ';' y cortó ahí, repartiendo el
        contenido en varias celdas y descartando el separador. Como el CSV real
        está delimitado por comas, el 96% de las líneas no contenía ningún ';'
        y quedó entera en la columna A; las 2.956 restantes (las que tenían un
        punto y coma dentro de alguna descripción) quedaron partidas.
        Se revierte volviendo a unir las celdas no vacías con ';'.

    Problema 2 — Reseñas repartidas en varias filas.
        Cuatro descripciones contenían saltos de línea reales, y al importarlas
        quedaron distribuidas en más de una fila de la hoja. Se detectan porque
        toda línea de datos sana empieza con el índice numérico seguido de coma
        ("48,US,..."); una línea que no cumple ese patrón es la continuación de
        la anterior.

    No se corrige acá el problema 3 (codificación), que se trata por separado
    en `_reparar_codificacion` porque opera sobre valores ya parseados.
    """
    crudo = pd.read_excel(ruta)

    # read_excel tomó la primera línea del CSV como encabezado y la convirtió en
    # el nombre de la columna 0. Hay que devolverla al cuerpo del texto, o se
    # pierden los nombres de las columnas y una reseña.
    lineas = [crudo.columns[0]]

    # Reponer el ';' que Excel consumió. El filtro por isinstance descarta los
    # NaN de las celdas vacías, que no son texto y romperían el join.
    for fila in crudo.itertuples(index=False):
        lineas.append(";".join(c for c in fila if isinstance(c, str)))

    # Volver a pegar las reseñas que quedaron partidas en varias filas.
    unidas = [lineas[0]]
    for linea in lineas[1:]:
        if re.match(r"^\d+,", linea):
            unidas.append(linea)
        else:
            unidas[-1] += "\n" + linea

    return "\n".join(unidas)


def _reparar_codificacion(texto):
    """Revierte el mojibake producido por leer UTF-8 como cp1252.

    Síntoma: 'Gewürztraminer' aparece como 'GewÃ¼rztraminer' y 'Albariño' como
    'AlbariÃ±o'. Ocurre cuando bytes UTF-8 se interpretan byte a byte con la
    codificación de Windows Europa Occidental.

    La reversión es exacta: se vuelve a codificar el texto en cp1252 para
    recuperar los bytes originales y se decodifica como UTF-8, que es lo que
    siempre fueron. Si alguno de los dos pasos falla, el valor se devuelve sin
    tocar en lugar de corromperlo más. Esto deja 134 celdas sin reparar sobre
    106.611 afectadas: son comillas tipográficas cuya secuencia de bytes incluye
    posiciones que cp1252 no define.

    La función es segura sobre texto ya correcto: un string ASCII atraviesa el
    ida y vuelta sin cambios, y uno con acentos legítimos falla al decodificar y
    se devuelve intacto.
    """
    if not isinstance(texto, str):
        return texto
    try:
        return texto.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def cargar_datos(ruta=RUTA_DATOS):
    """Carga el dataset de reseñas y devuelve un DataFrame verificado."""
    texto = _reconstruir_lineas(ruta)

    # index_col=0 adopta la columna de índice heredada del CSV como índice del
    # DataFrame, en vez de arrastrarla como una columna de datos redundante.
    df = pd.read_csv(io.StringIO(texto), index_col=0)

    for columna in df.select_dtypes(exclude="number").columns:
        df[columna] = df[columna].map(_reparar_codificacion)

    # Verificación explícita de la carga. Que read_csv no lance una excepción
    # sólo prueba que el texto era parseable, no que se haya cargado bien; estas
    # condiciones sí fallan si el archivo de origen cambia o la reconstrucción
    # se rompe.
    assert df.shape == (129971, 13), f"shape inesperado: {df.shape}"
    assert df["points"].dtype.kind == "i", "points debería ser entero"
    assert df["price"].dtype.kind == "f", "price debería ser float"
    assert df.index.is_unique, "el índice tiene valores repetidos"

    return df


# ===========================================================================
# 2. PERFILADO Y DEPURACIÓN INICIAL
# ===========================================================================

def perfil_columnas(df):
    """Resumen por columna: tipo, faltantes y cardinalidad.

    Reemplaza a `info()` porque agrega dos cosas que hacen falta para decidir
    el tratamiento de cada variable: el porcentaje de nulos (comparable entre
    columnas) y la cardinalidad relativa, que distingue una categórica de un
    identificador. Una columna con cardinalidad cercana al 100% (`description`,
    `title`) es texto libre o clave única y no admite análisis de frecuencias.
    """
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "no_nulos": df.notna().sum(),
        "nulos": df.isna().sum(),
        "nulos_%": (df.isna().mean() * 100).round(2),
        "unicos": df.nunique(),
        "cardinalidad_%": (df.nunique() / len(df) * 100).round(2),
    }).sort_values("nulos_%", ascending=False)


def quitar_duplicados(df):
    """Elimina reseñas repetidas, conservando la primera aparición de cada una.

    El dataset trae 9.983 pares de filas idénticas en las 13 columnas: misma
    descripción, mismo vino, mismo catador, mismo precio y mismo puntaje. Son
    un artefacto de la extracción del sitio, no vinos distintos que casualmente
    recibieron la misma reseña palabra por palabra.

    Se eliminan antes de cualquier análisis, y no por prolijidad: una fila
    contada dos veces duplica su peso en todo promedio, conteo, frecuencia y
    correlación que se calcule después. Con el 7,7% de las filas repetidas, los
    rankings por país y las medias por variedad quedarían sesgados hacia lo que
    la extracción repitió.

    Se comparan las 13 columnas y no sólo `description`, para no descartar dos
    reseñas legítimamente distintas que compartan texto. En este dataset ambos
    criterios coinciden (9.983 en los dos casos), lo que confirma que se trata
    de filas enteramente repetidas.
    """
    antes = len(df)
    limpio = df.drop_duplicates()
    print(f"Duplicados eliminados: {antes - len(limpio):,} "
          f"({(antes - len(limpio)) / antes * 100:.1f}%) -> {len(limpio):,} reseñas")
    return limpio


# ===========================================================================
# 3. VISUALIZACIÓN
# ===========================================================================

DIR_GRAFICOS = Path("graficos")


def _guardar(fig, nombre):
    """Guarda la figura en graficos/ y la devuelve para que el notebook la muestre."""
    DIR_GRAFICOS.mkdir(exist_ok=True)
    fig.savefig(DIR_GRAFICOS / f"{nombre}.png", dpi=150, bbox_inches="tight")
    return fig


def grafico_puntajes(df):
    """Distribución de `points`.

    Se usa un bin por puntaje entero (la variable toma sólo 21 valores
    distintos) para que cada barra represente un puntaje real y no un intervalo
    arbitrario. La línea en 80 marca el punto de censura declarado en el
    enunciado: WineEnthusiast no publica reseñas por debajo de ese valor, así
    que la distribución observada es la de los vinos publicados, no la de los
    vinos evaluados.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = range(df["points"].min(), df["points"].max() + 2)
    ax.hist(df["points"], bins=bins, color="#7b2d43", edgecolor="white", align="left")

    # Espacio libre arriba para que las anotaciones no se superpongan a las barras.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    alto = ax.get_ylim()[1]

    ax.axvline(80, color="#c8102e", ls="--", lw=1.5)
    ax.text(80.4, alto * 0.94, "censura: no se publican reseñas < 80",
            color="#c8102e", fontsize=9, va="top")
    ax.axvline(df["points"].mean(), color="black", ls=":", lw=1.5)
    ax.text(df["points"].mean() + 0.3, alto * 0.84,
            f"media {df['points'].mean():.1f}", fontsize=9, va="top")
    ax.set(xlabel="Puntaje", ylabel="Cantidad de reseñas",
           title="Distribución de puntajes (simétrica, pero censurada en 80)")
    return _guardar(fig, "01_distribucion_puntajes")


def grafico_precios(df):
    """Distribución de `price` en escala lineal y logarítmica.

    Las dos escalas van juntas a propósito: el panel izquierdo muestra por qué
    la escala lineal es inservible acá (la asimetría de 18 comprime el 99% de
    los datos contra el eje y deja el gráfico vacío), y el derecho muestra que
    en escala logarítmica la misma variable es aproximadamente simétrica. Esa
    comparación justifica usar log en el resto del análisis: no es un truco
    estético, es que `price` se comporta como una variable log-normal.
    """
    precio = df["price"].dropna()
    fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.5))

    izq.hist(precio, bins=100, color="#7b2d43", edgecolor="white")
    izq.set(xlabel="Precio (USD)", ylabel="Cantidad de reseñas",
            title=f"Escala lineal — asimetría {precio.skew():.1f}")

    der.hist(precio, bins=np.logspace(np.log10(precio.min()), np.log10(precio.max()), 60),
             color="#3d5a6c", edgecolor="white")
    der.set_xscale("log")
    der.axvline(precio.median(), color="#c8102e", ls="--", lw=1.5)
    der.text(precio.median() * 1.1, der.get_ylim()[1] * 0.9,
             f"mediana ${precio.median():.0f}", color="#c8102e", fontsize=9)
    der.set(xlabel="Precio (USD, escala log)", ylabel="",
            title=f"Escala logarítmica — asimetría {np.log(precio).skew():.2f}")

    fig.suptitle("La misma variable bajo dos escalas", y=1.02)
    return _guardar(fig, "02_distribucion_precios")


def grafico_categoricas(df, n=15):
    """Top-N de las categóricas con mayor peso analítico.

    Se grafican sólo las N categorías más frecuentes, ordenadas: con 707
    variedades y 425 provincias, un gráfico completo sería ilegible y no
    comunicaría nada. El subtítulo de cada panel informa qué porcentaje del
    total cubre el top-N, para que el recorte quede explícito y el lector sepa
    cuánto está viendo.
    """
    columnas = ["country", "variety", "province", "winery"]
    fig, ejes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, col in zip(ejes.flat, columnas):
        vc = df[col].value_counts().head(n).sort_values()
        cobertura = vc.sum() / df[col].notna().sum() * 100
        ax.barh(vc.index.astype(str), vc.values, color="#7b2d43")
        ax.set(xlabel="Reseñas",
               title=f"{col} — top {n} de {df[col].nunique()} ({cobertura:.0f}% del total)")
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    return _guardar(fig, "03_categoricas_top")


def grafico_precio_vs_puntaje(df):
    """Relación entre precio y puntaje.

    Un diagrama de dispersión con 120.000 puntos sería una mancha sólida, así
    que se usa un boxplot por puntaje: `points` toma sólo 21 valores, lo que lo
    vuelve un agrupador natural. El eje de precio va en escala logarítmica por
    lo visto en `grafico_precios`.

    La línea une las medianas para hacer visible la tendencia central sin que
    la arrastren los precios extremos.
    """
    datos = df.dropna(subset=["price"])
    puntajes = sorted(datos["points"].unique())
    grupos = [datos.loc[datos["points"] == p, "price"].values for p in puntajes]
    medianas = [np.median(g) for g in grupos]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(grupos, positions=puntajes, widths=0.6, showfliers=False,
               medianprops=dict(color="#c8102e"))
    ax.plot(puntajes, medianas, color="#c8102e", lw=2, marker="o", ms=4,
            label="mediana por puntaje")
    ax.set_yscale("log")
    ax.set(xlabel="Puntaje", ylabel="Precio (USD, escala log)",
           title="El precio crece de forma aproximadamente exponencial con el puntaje")
    ax.legend()
    return _guardar(fig, "04_precio_vs_puntaje")


def grafico_puntaje_por_pais(df, n=12, minimo=500):
    """Distribución de puntajes en los principales países productores.

    Se exige un mínimo de reseñas por país para que la comparación sea honesta:
    la mediana de un país con 20 reseñas no es comparable con la de uno que
    tiene 50.000, y ordenar sin ese filtro pondría países marginales arriba del
    ranking por puro ruido muestral. Los países se ordenan por mediana, no
    alfabéticamente, para que el gráfico se lea como un ranking.

    Se usa boxplot y no violín: el violín estima la densidad por kernel, un
    método que asume variable continua. Como `points` toma sólo 21 valores
    enteros, el suavizado produce ondulaciones que son artefactos del método y
    no estructura de los datos. El boxplot trabaja sobre cuantiles y no inventa
    forma donde no la hay.
    """
    conteos = df["country"].value_counts()
    candidatos = conteos[conteos >= minimo].index
    orden = (df[df["country"].isin(candidatos)]
             .groupby("country", observed=True)["points"].median()
             .sort_values(ascending=False).head(n).index)

    grupos = [df.loc[df["country"] == c, "points"].values for c in orden]

    fig, ax = plt.subplots(figsize=(11, 5))
    cajas = ax.boxplot(grupos, widths=0.6, patch_artist=True, showfliers=False,
                       medianprops=dict(color="#c8102e", lw=2))
    for caja in cajas["boxes"]:
        caja.set_facecolor("#7b2d43")
        caja.set_alpha(0.75)
    ax.set_xticks(range(1, len(orden) + 1))
    ax.set_xticklabels([f"{c}\n(n={conteos[c]:,})" for c in orden], fontsize=8)
    ax.set(ylabel="Puntaje",
           title=f"Puntajes por país — top {n} por mediana (mínimo {minimo} reseñas)")
    ax.grid(axis="y", alpha=0.3)
    return _guardar(fig, "05_puntaje_por_pais")


# ===========================================================================
# 4. TRATAMIENTO DE DATOS FALTANTES
# ===========================================================================

# Etiquetas para los faltantes estructurales. La clave de esta sección es que
# la mayoría de los nulos de este dataset NO son datos perdidos: son campos que
# no aplican al vino en cuestión. Reemplazarlos por una etiqueta explícita
# conserva esa información; imputarlos inventaría un dato que nunca existió.
ETIQUETAS_FALTANTES = {
    "region_2": "No aplica",        # sub-apelación exclusiva del sistema de EE.UU.
    "region_1": "Sin región",
    "designation": "Sin designación",
    "taster_name": "No informado",
    "taster_twitter_handle": "No informado",
    "country": "Desconocido",
    "province": "Desconocido",
    "variety": "Desconocida",
}


def diagnostico_faltantes(df):
    """Mide si la ausencia de `price` depende de otras variables observadas.

    Distinguir el mecanismo es lo que decide el tratamiento:

    - MCAR (falta completamente al azar): la ausencia no se relaciona con nada.
      Cualquier imputación razonable sirve y eliminar filas no sesga.
    - MAR (falta al azar condicionado): la ausencia depende de variables que sí
      se observan. Imputar con un valor global sesga; hay que condicionar.
    - MNAR: la ausencia depende del propio valor faltante. No es corregible
      sólo con los datos disponibles.

    En este dataset `price` resulta claramente MAR: Francia tiene 20% de
    precios ausentes contra 0,4% de Estados Unidos, y la tasa crece con el
    puntaje. Por eso se imputa condicionando por país, variedad y puntaje en
    lugar de usar la mediana general.
    """
    falta = df["price"].isna()
    por_pais = (df.assign(_f=falta).groupby("country", observed=True)["_f"]
                  .agg(reseñas="size", sin_precio_pct=lambda s: round(s.mean() * 100, 1)))
    por_pais = por_pais[por_pais["reseñas"] >= 1000].sort_values("sin_precio_pct",
                                                                ascending=False)
    por_puntaje = (df.assign(_f=falta).groupby("points")["_f"].mean() * 100).round(1)
    return por_pais, por_puntaje


def validar_imputacion_precio(df, semilla=42):
    """Compara estrategias de imputación ocultando precios que sí se conocen.

    Es la única forma honesta de elegir un método: se esconde una muestra de
    precios observados, se imputa con cada candidato y se mide el error contra
    el valor real, que en esas filas sí se conoce.

    El ocultamiento replica el patrón MAR real —se oculta más en los países que
    de verdad tienen más faltantes— en vez de sortear filas al azar. Evaluar
    contra un patrón MCAR artificial sobrestimaría a los métodos globales, que
    son justamente los que fallan cuando la ausencia está concentrada.
    """
    rng = np.random.default_rng(semilla)
    tasa = df.assign(_f=df["price"].isna()).groupby("country", observed=True)["_f"].mean()

    ocultar = []
    for pais, grupo in df[df["price"].notna()].groupby("country", observed=True):
        cuantos = int(len(grupo) * tasa.get(pais, 0.07))
        if cuantos:
            ocultar += list(rng.choice(grupo.index, size=cuantos, replace=False))
    ocultar = pd.Index(ocultar)

    real = df.loc[ocultar, "price"]
    prueba = df.copy()
    prueba.loc[ocultar, "price"] = np.nan

    def mediana_por(*columnas):
        return prueba.groupby(list(columnas), observed=True)["price"].transform("median")

    candidatos = {
        "media global": prueba["price"].fillna(prueba["price"].mean()),
        "mediana global": prueba["price"].fillna(prueba["price"].median()),
        "mediana por país": prueba["price"].fillna(mediana_por("country")),
        "mediana por variedad": prueba["price"].fillna(mediana_por("variety")),
        "mediana por país+variedad": prueba["price"].fillna(mediana_por("country", "variety")),
        "mediana por país+variedad+puntaje": prueba["price"].fillna(
            mediana_por("country", "variety", "points")),
    }

    filas = {}
    for nombre, estimado in candidatos.items():
        estimado = estimado.fillna(prueba["price"].median())
        error = estimado.loc[ocultar] - real
        filas[nombre] = {
            "MAE": error.abs().mean(),
            "error_mediano": error.abs().median(),
            "MAPE_%": (error.abs() / real * 100).median(),
            "sesgo": error.mean(),
        }
    return pd.DataFrame(filas).T.round(2)


def imputar_precio(df):
    """Imputa `price` con la mediana condicional por país, variedad y puntaje.

    Estrategia elegida por la validación de `validar_imputacion_precio`: baja el
    error absoluto medio de $21,96 (mediana global) a $14,88, y el error
    porcentual mediano del 47% al 25%. Incorporar `points` al agrupador es lo
    que más aporta, algo consistente con la relación exponencial entre precio y
    puntaje que muestra el gráfico 04.

    Se usan medianas y no medias porque `price` tiene asimetría 18: la media de
    cualquier grupo está inflada por unas pocas botellas muy caras.

    La cascada de respaldos cubre los grupos sin datos suficientes, de lo más
    específico a lo más general, y garantiza que no queden nulos.

    Se agrega la columna `price_imputado` para poder excluir estos valores de
    cualquier análisis donde comprometan la conclusión. Importa especialmente
    porque la imputación usa `points`: estudiar la relación precio-puntaje
    sobre valores imputados así reforzaría artificialmente esa misma relación.
    Es un caso de circularidad, y la bandera permite evitarlo.
    """
    df = df.copy()
    df["price_imputado"] = df["price"].isna()

    def mediana_por(*columnas):
        return df.groupby(list(columnas), observed=True)["price"].transform("median")

    df["price"] = (df["price"]
                   .fillna(mediana_por("country", "variety", "points"))
                   .fillna(mediana_por("country", "variety"))
                   .fillna(mediana_por("variety"))
                   .fillna(mediana_por("country"))
                   .fillna(df["price"].median()))

    assert df["price"].notna().all(), "quedaron precios sin imputar"
    return df


def tratar_faltantes(df):
    """Aplica el tratamiento de faltantes decidido para cada columna.

    El orden importa: primero se etiquetan las categóricas, para que las filas
    sin país o sin variedad formen su propio grupo en lugar de desaparecer de
    los agrupamientos que después usa la imputación de precio.

    No se elimina ninguna fila. Las 63 sin país y la única sin variedad tienen
    puntaje, precio y descripción válidos: descartarlas perdería esa
    información a cambio de nada, mientras que etiquetarlas deja la ignorancia
    declarada y visible en cualquier tabla de frecuencias.
    """
    df = df.copy()
    for columna, etiqueta in ETIQUETAS_FALTANTES.items():
        df[columna] = df[columna].fillna(etiqueta)
    return imputar_precio(df)


def grafico_faltantes(df):
    """Mapa de faltantes por columna y de su co-ocurrencia.

    El panel izquierdo ordena las columnas por porcentaje de nulos. El derecho
    muestra, para cada par de columnas, qué proporción de las filas a las que
    les falta una también tiene ausente la otra: es lo que distingue un
    faltante estructural de uno aleatorio. El 1,00 entre `taster_name` y
    `taster_twitter_handle` prueba que la segunda es función de la primera.
    """
    con_nulos = df.columns[df.isna().any()]
    nulos = df[con_nulos].isna()
    orden = nulos.mean().sort_values(ascending=False).index
    nulos = nulos[orden]

    fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    pct = nulos.mean() * 100
    izq.barh(range(len(pct)), pct.values, color="#7b2d43")
    izq.set_yticks(range(len(pct)))
    izq.set_yticklabels(pct.index, fontsize=9)
    izq.invert_yaxis()
    for i, v in enumerate(pct.values):
        izq.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    izq.set(xlabel="% de valores faltantes", title="Faltantes por columna",
            xlim=(0, pct.max() * 1.25))

    # P(falta B | falta A): condicional, no simétrica, y por eso más informativa
    # que una correlación para detectar dependencias estructurales.
    cond = pd.DataFrame(
        {b: [nulos[b][nulos[a]].mean() for a in orden] for b in orden}, index=orden)
    im = der.imshow(cond.values, cmap="RdPu", vmin=0, vmax=1)
    der.set_xticks(range(len(orden)))
    der.set_xticklabels(orden, rotation=45, ha="right", fontsize=8)
    der.set_yticks(range(len(orden)))
    der.set_yticklabels(orden, fontsize=8)
    for i in range(len(orden)):
        for j in range(len(orden)):
            v = cond.values[i, j]
            der.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if v > 0.6 else "black")
    # Se lee por fila: "de las filas a las que les falta <fila>, esta proporción
    # tiene también ausente <columna>". No es simétrica, y ahí está su valor.
    der.set_title("P(falta la columna | falta la fila)")
    fig.colorbar(im, ax=der, shrink=0.8)

    fig.tight_layout()
    return _guardar(fig, "06_faltantes")


# ===========================================================================
# EJECUCIÓN
# ===========================================================================

if __name__ == "__main__":
    # Sin backend interactivo: el script guarda las figuras en graficos/ en vez
    # de abrir ventanas, para que pueda correrse de punta a punta sin bloquearse.
    import matplotlib
    matplotlib.use("Agg")

    vinos = cargar_datos()
    print(f"Dataset cargado: {vinos.shape[0]:,} reseñas x {vinos.shape[1]} columnas")

    print("\n--- Perfil de columnas ---")
    print(perfil_columnas(vinos).to_string())

    print("\n--- Depuración ---")
    vinos = quitar_duplicados(vinos)

    print("\n--- Gráficos ---")
    for funcion in (grafico_puntajes, grafico_precios, grafico_categoricas,
                    grafico_precio_vs_puntaje, grafico_puntaje_por_pais,
                    grafico_faltantes):
        funcion(vinos)
        plt.close("all")
    print(f"Guardados en {DIR_GRAFICOS.resolve()}")

    print("\n--- Faltantes: diagnóstico del mecanismo ---")
    por_pais, por_puntaje = diagnostico_faltantes(vinos)
    print("Precio ausente por país (n >= 1000):")
    print(por_pais.head(6).to_string())
    print(f"\nPrecio ausente según puntaje: {por_puntaje.loc[80]}% en 80 "
          f"-> {por_puntaje.loc[99]}% en 99")
    print("La ausencia depende de variables observadas: el mecanismo es MAR.")

    print("\n--- Faltantes: validación de estrategias de imputación ---")
    print(validar_imputacion_precio(vinos).to_string())

    print("\n--- Faltantes: tratamiento aplicado ---")
    vinos = tratar_faltantes(vinos)
    print(f"Precios imputados: {vinos['price_imputado'].sum():,} "
          f"({vinos['price_imputado'].mean() * 100:.1f}%)")
    print(f"Nulos restantes en el dataset: {vinos.isna().sum().sum()}")
