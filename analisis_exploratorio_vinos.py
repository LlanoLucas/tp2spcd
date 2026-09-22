"""
Análisis exploratorio de datos de reseñas de vinos (WineEnthusiast)
Trabajo Práctico 2, Seminario de Programación para Ciencia de Datos

Autor: Lucas Llano

Requisitos: pandas, numpy, matplotlib, seaborn, scipy
Uso:        python analisis_exploratorio_vinos.py
"""

import csv
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

# El script acepta las dos formas en que puede llegar el dataset y elige el
# camino según la extensión:
#
#   .csv   el archivo original, que se lee directo.
#   .xlsx  el archivo que vino con el enunciado, que no es una hoja de cálculo
#          sino ese mismo CSV dañado por una importación mal configurada en
#          Excel. La sección 1 documenta y revierte ese daño.
#
# Se buscan en orden: el primero que exista es el que se usa.
RUTAS_POSIBLES = [
    "winemag-data-130k-v2.csv",
    "Entregable 2 - Documentación extra.xlsx",
]

# Campos que debe tener cada línea del CSV: 13 columnas de datos más la columna
# de índice sin nombre que quedó al exportar. Es la condición que verifica si la
# reconstrucción del archivo dañado salió bien.
CAMPOS_POR_LINEA = 14


# ===========================================================================
# 1. CARGA DE DATOS
# ===========================================================================

def _reconstruir_lineas(ruta):
    """Deshace el daño de la importación a Excel y devuelve el CSV como texto.

    El archivo entregado tiene tres problemas, todos originados en el mismo
    hecho: el CSV se abrió en Excel con una configuración regional que usa el
    punto y coma como separador de campos.

    Problema 1. Delimitador equivocado.
        Excel recorrió cada línea buscando ';' y cortó ahí, repartiendo el
        contenido en varias celdas y descartando el separador. Como el CSV real
        está delimitado por comas, el 96% de las líneas no contenía ningún ';'
        y quedó entera en la columna A; las 2.956 restantes (las que tenían un
        punto y coma dentro de alguna descripción) quedaron partidas.
        Se revierte volviendo a unir las celdas no vacías con ';'.

    Problema 2. Reseñas repartidas en varias filas.
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

    texto = "\n".join(unidas)

    # El criterio de éxito se fija antes de cargar, no después: si la
    # reconstrucción es correcta, toda línea da CAMPOS_POR_LINEA campos al
    # analizarse como CSV. Se usa csv.reader y no split(",") porque las
    # descripciones tienen comas dentro de comillas.
    anchos = {len(campos) for campos in csv.reader(io.StringIO(texto))}
    assert anchos == {CAMPOS_POR_LINEA}, (
        f"la reconstrucción falló: se esperaban {CAMPOS_POR_LINEA} campos por "
        f"línea y se encontraron anchos {sorted(anchos)}")

    return texto


def _reparar_codificacion(texto):
    """Revierte el mojibake producido por leer UTF-8 como cp1252.

    Síntoma: 'Gewürztraminer' aparece como 'GewÃ¼rztraminer' y 'Albariño' como
    'AlbariÃ±o'. Ocurre cuando bytes UTF-8 se interpretan byte a byte con la
    codificación de Windows Europa Occidental.

    La vuelta atrás es exacta: se codifica el texto en cp1252 para recuperar los
    bytes originales y se decodifica como UTF-8, que es lo que siempre fueron.
    Si cualquiera de los dos pasos falla, el valor vuelve sin tocar en lugar de
    corromperse más. Quedan así 132 celdas sin reparar sobre 106.611 afectadas:
    son comillas tipográficas cuya secuencia de bytes usa posiciones que cp1252
    no define.
    """
    if not isinstance(texto, str):
        return texto

    # Tres celdas pasaron dos veces por la conversión equivocada y quedaron con
    # mojibake sobre mojibake: "crème" aparece como "crÃƒÂ¨me", y una sola
    # vuelta la deja en "crÃ¨me", todavía rota. Se repite hasta que el texto
    # deje de cambiar.
    #
    # El ciclo siempre termina: cada vuelta que tiene efecto acorta el texto,
    # porque el mojibake usa más caracteres que el original. Y no puede pasarse
    # de largo sobre texto ya correcto: "Gewürztraminer" da los bytes de cp1252
    # 'Gew\xfcrztraminer', que no son UTF-8 válido, así que falla y se devuelve
    # intacto. El tope es una red de seguridad, no parte de la lógica.
    for _ in range(5):
        try:
            candidato = texto.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return texto
        if candidato == texto:
            return texto
        texto = candidato

    return texto


def _elegir_ruta():
    """Devuelve el primer archivo de datos que exista entre los aceptados."""
    for candidata in RUTAS_POSIBLES:
        if Path(candidata).exists():
            return candidata
    raise FileNotFoundError(
        "No se encontró el archivo de datos. Se buscaron, en este orden: "
        + ", ".join(RUTAS_POSIBLES))


def cargar_datos(ruta=None):
    """Carga el dataset de reseñas y devuelve un DataFrame verificado.

    Acepta tanto el CSV original como el .xlsx dañado que vino con el enunciado,
    y elige qué hacer según la extensión. El CSV se lee directo; el .xlsx pasa
    antes por la reconstrucción de `_reconstruir_lineas`.

    Los dos caminos terminan en el mismo lugar: la reparación de codificación se
    aplica igual, porque es inocua sobre texto que ya está bien, y las mismas
    aserciones verifican el resultado venga de donde venga.
    """
    ruta = ruta or _elegir_ruta()

    # index_col=0 adopta la columna de índice heredada del CSV como índice del
    # DataFrame, en vez de arrastrarla como una columna de datos redundante.
    if str(ruta).lower().endswith(".csv"):
        df = pd.read_csv(ruta, index_col=0)
    else:
        df = pd.read_csv(io.StringIO(_reconstruir_lineas(ruta)), index_col=0)

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


def resumen_categoricas(df, columnas=None):
    """Resumen estadístico de las variables categóricas.

    Las medidas de tendencia central y dispersión que sirven para una variable
    numérica no tienen sentido acá: no hay promedio de países ni desvío de
    variedades. El equivalente para una categórica son la moda, su peso
    relativo y alguna medida de cuán repartida está la masa entre categorías.

    La entropía normalizada cumple ese papel. Vale 0 cuando todas las reseñas
    caen en una sola categoría y 1 cuando se reparten en partes iguales, así que
    compara concentración entre variables con distinta cantidad de categorías.
    Es lo que distingue a `country`, donde unos pocos países se llevan casi
    todo, de `winery`, donde 16.757 bodegas se reparten el conjunto de forma
    pareja.

    Se excluye el texto libre (`description`, `title`): con más del 90% de
    valores únicos, su moda aparece dos o tres veces y no dice nada.
    """
    if columnas is None:
        columnas = ["country", "province", "region_1", "region_2", "variety",
                    "winery", "designation", "taster_name"]

    filas = {}
    for col in columnas:
        valores = df[col].dropna()
        frecuencias = valores.value_counts()
        proporciones = frecuencias / len(valores)

        # Entropía de Shannon dividida por su máximo posible, log(k).
        entropia = -(proporciones * np.log(proporciones)).sum()
        maxima = np.log(len(frecuencias)) if len(frecuencias) > 1 else 1

        filas[col] = {
            "categorías": len(frecuencias),
            "moda": frecuencias.index[0],
            "frec_moda": int(frecuencias.iloc[0]),
            "moda_%": round(proporciones.iloc[0] * 100, 1),
            "top10_%": round(frecuencias.head(10).sum() / len(valores) * 100, 1),
            "entropía_norm": round(entropia / maxima, 3),
        }

    return pd.DataFrame(filas).T


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
            title=f"Escala lineal: asimetría {precio.skew():.1f}")

    der.hist(precio, bins=np.logspace(np.log10(precio.min()), np.log10(precio.max()), 60),
             color="#3d5a6c", edgecolor="white")
    der.set_xscale("log")
    der.axvline(precio.median(), color="#c8102e", ls="--", lw=1.5)
    der.text(precio.median() * 1.1, der.get_ylim()[1] * 0.9,
             f"mediana ${precio.median():.0f}", color="#c8102e", fontsize=9)
    der.set(xlabel="Precio (USD, escala log)", ylabel="",
            title=f"Escala logarítmica: asimetría {np.log(precio).skew():.2f}")

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
               title=f"{col}: top {n} de {df[col].nunique()} ({cobertura:.0f}% del total)")
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
           title=f"Puntajes por país, top {n} por mediana (mínimo {minimo} reseñas)")
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

    En este dataset `price` es claramente MAR: Francia tiene 20% de
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

    El ocultamiento replica el patrón MAR real, ocultando más en los países que
    de verdad tienen más faltantes, en vez de sortear filas al azar. Evaluar
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
# 5. ANÁLISIS DE DATOS ATÍPICOS
# ===========================================================================

# Un precio se considera sospechoso cuando supera en este factor a la mediana
# de su grupo de pares. El umbral sale de la distribución observada de ratios:
# el percentil 99,9 está en 7,3x, así que 10x deja fuera del alcance a la
# variación legítima de mercado y sólo alcanza a casos que exigen explicación.
UMBRAL_RATIO_PARES = 10

# Mínimo de vinos que debe tener un grupo de pares para que su mediana sirva
# como referencia. Con menos, la mediana es demasiado inestable.
MINIMO_GRUPO_PARES = 30


def detectar_atipicos(df):
    """Compara métodos de detección de atípicos sobre las variables numéricas.

    Se aplican cuatro criterios a propósito, porque discrepan y la discrepancia
    es el hallazgo:

    - IQR clásico: asume distribución aproximadamente simétrica. Sobre `price`
      marca el 6,2% de los datos, lo que no describe casos excepcionales sino
      la cola natural de una distribución asimétrica.
    - z-score: asume normalidad, y además usa media y desvío, que son
      justamente los estadísticos que los atípicos distorsionan. El resultado
      es circular y sub-detecta.
    - z modificado sobre la MAD: reemplaza media y desvío por mediana y
      desviación absoluta mediana, que los valores extremos no arrastran.
    - IQR sobre log(price): aplica el criterio sobre la escala en la que la
      variable sí es simétrica. Es el más defendible para esta variable.

    Sólo se usan precios observados: incluir los imputados mezclaría valores
    sintéticos, que por construcción están cerca de la mediana de su grupo y
    nunca serían atípicos.
    """
    observados = df[~df["price_imputado"]]
    filas = {}

    for variable in ["price", "points"]:
        x = observados[variable]
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        z = (x - x.mean()) / x.std()
        mad = (x - x.median()).abs().median()
        z_mod = 0.6745 * (x - x.median()) / mad
        log_x = np.log(x)
        lq1, lq3 = log_x.quantile([0.25, 0.75])
        liqr = lq3 - lq1

        filas[variable] = {
            "IQR clásico": ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() * 100,
            "z-score |z|>3": (z.abs() > 3).mean() * 100,
            "z modificado (MAD) |z|>3.5": (z_mod.abs() > 3.5).mean() * 100,
            "IQR sobre log(x)": ((log_x < lq1 - 1.5 * liqr)
                                 | (log_x > lq3 + 1.5 * liqr)).mean() * 100,
        }

    return pd.DataFrame(filas).round(2).rename_axis("% marcado como atípico")


def atipicos_contextuales(df, umbral=UMBRAL_RATIO_PARES):
    """Identifica precios implausibles comparando cada vino con sus pares.

    Los métodos estadísticos del apartado anterior sólo responden "¿es un valor
    extremo?". La pregunta que pide la consigna es otra: "¿es un error?". Y esa
    no se contesta mirando la distribución global, porque un Château Pétrus a
    $2.500 es extremo y correcto, mientras que un Médoc corriente a $3.300 es
    igual de extremo y es un error de carga.

    La diferencia sólo aparece en contexto. Se compara cada precio con la
    mediana de su grupo de pares, o sea los vinos de la misma provincia y el
    mismo puntaje, y se reporta el cociente. Un vino caro entre vinos caros da un ratio cercano a
    1; un vino caro entre vinos baratos se delata.

    Devuelve los casos por encima del umbral, ordenados por ratio, para
    inspección manual. Es deliberado que no los corrija de forma automática:
    decidir si un precio es erróneo requiere criterio, y la función entrega la
    evidencia para ejercerlo.
    """
    observados = df[~df["price_imputado"]]
    grupo = observados.groupby(["province", "points"], observed=True)["price"]
    mediana_pares = grupo.transform("median")
    tamaño = grupo.transform("size")

    confiable = tamaño >= MINIMO_GRUPO_PARES
    ratio = (observados["price"] / mediana_pares)[confiable]

    sospechosos = observados.loc[ratio[ratio > umbral].index].copy()
    sospechosos["mediana_pares"] = mediana_pares[sospechosos.index]
    sospechosos["ratio"] = ratio[sospechosos.index].round(1)

    columnas = ["title", "province", "points", "price", "mediana_pares", "ratio"]
    return sospechosos[columnas].sort_values("ratio", ascending=False)


def impacto_atipicos(df):
    """Mide cómo cambia la relación precio-puntaje según se traten los atípicos.

    Es la parte de la consigna que pide analizar cómo los valores atípicos
    afectan las decisiones basadas en los datos, y el resultado es contundente.

    Pearson mide asociación *lineal*, así que sobre un precio con asimetría 18
    subestima la relación real. Spearman trabaja sobre rangos y es insensible a
    la escala, por eso apenas se mueve entre escenarios: es la referencia
    robusta contra la cual comparar.

    La conclusión práctica es que eliminar atípicos y transformar la escala no
    son dos caminos igual de buenos hacia el mismo lugar. Descartar los precios
    altos sube el Pearson pero elimina decenas de miles de vinos reales; el
    logaritmo alcanza un valor todavía mayor sin perder una sola observación.
    Para esta variable, la respuesta correcta a la asimetría es cambiar de
    escala, no recortar los datos.

    Se excluye del cálculo el precio imputado, que se generó usando `points` y
    por lo tanto inflaría artificialmente cualquier correlación con esa misma
    variable.
    """
    observados = df[~df["price_imputado"]]
    precio, puntaje = observados["price"], observados["points"]

    q1, q3 = precio.quantile([0.25, 0.75])
    limite = q3 + 1.5 * (q3 - q1)
    recortado = observados[precio <= limite]

    escenarios = {
        "precio crudo, con atípicos": (precio, puntaje),
        "precio crudo, sin atípicos (IQR)": (recortado["price"], recortado["points"]),
        "log(precio), con atípicos": (np.log(precio), puntaje),
    }

    filas = {}
    for nombre, (x, y) in escenarios.items():
        filas[nombre] = {
            "n": len(x),
            "Pearson": stats.pearsonr(x, y)[0],
            "Spearman": stats.spearmanr(x, y)[0],
        }
    resultado = pd.DataFrame(filas).T
    resultado["n"] = resultado["n"].astype(int)
    return resultado.round(3)


def grafico_atipicos(df):
    """Panorama de los atípicos de precio y del criterio usado para juzgarlos.

    Izquierda: la cola de precios en escala logarítmica, con los umbrales de
    los dos criterios que más discrepan, para hacer visible que sobre una
    variable asimétrica el IQR clásico marca la cola entera y no casos raros.

    Derecha: la distribución del cociente contra el grupo de pares. Es el
    gráfico que justifica el umbral elegido, porque muestra que la masa se
    concentra alrededor de 1 y que por encima de 10 quedan unos pocos casos
    separados del resto.
    """
    observados = df[~df["price_imputado"]]
    precio = observados["price"]

    fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.5))

    q1, q3 = precio.quantile([0.25, 0.75])
    limite_iqr = q3 + 1.5 * (q3 - q1)
    log_p = np.log(precio)
    lq1, lq3 = log_p.quantile([0.25, 0.75])
    limite_log = np.exp(lq3 + 1.5 * (lq3 - lq1))

    izq.hist(precio, bins=np.logspace(np.log10(precio.min()),
                                      np.log10(precio.max()), 70),
             color="#3d5a6c", edgecolor="white")
    izq.set_xscale("log")
    izq.axvline(limite_iqr, color="#c8102e", ls="--",
                label=f"IQR clásico: ${limite_iqr:.0f} ({(precio > limite_iqr).mean() * 100:.1f}%)")
    izq.axvline(limite_log, color="#e07a00", ls="--",
                label=f"IQR sobre log: ${limite_log:.0f} ({(precio > limite_log).mean() * 100:.1f}%)")
    izq.set(xlabel="Precio (USD, escala log)", ylabel="Cantidad de reseñas",
            title="Dos criterios, seis veces de diferencia")
    izq.legend(fontsize=8)

    grupo = observados.groupby(["province", "points"], observed=True)["price"]
    ratio = (observados["price"] / grupo.transform("median"))[
        grupo.transform("size") >= MINIMO_GRUPO_PARES]

    der.hist(ratio, bins=np.logspace(np.log10(ratio.min()), np.log10(ratio.max()), 70),
             color="#7b2d43", edgecolor="white")
    der.set_xscale("log")
    der.set_yscale("log")
    der.axvline(UMBRAL_RATIO_PARES, color="#c8102e", ls="--",
                label=f"umbral {UMBRAL_RATIO_PARES}x ({(ratio > UMBRAL_RATIO_PARES).sum()} casos)")
    der.set(xlabel="Precio / mediana del grupo de pares (provincia + puntaje)",
            ylabel="Cantidad de reseñas",
            title="Contra los pares, los errores se separan del resto")
    der.legend(fontsize=8)

    fig.tight_layout()
    return _guardar(fig, "07_atipicos")


# ===========================================================================
# 6. ANÁLISIS COMPARATIVO
# ===========================================================================

# Mínimo de reseñas para que un grupo entre en una comparación. Sin este filtro
# los rankings los encabezan grupos de tres observaciones por puro azar
# muestral, y la comparación deja de significar algo.
MINIMO_GRUPO = 200

# La extracción del sitio es de 2017: cosechas posteriores serían imposibles.
ANIO_MAXIMO = 2017
ANIO_MINIMO = 1900


def extraer_anio(df):
    """Deriva el año de cosecha desde `title` y lo agrega como columna.

    El título sigue el formato "<bodega> <año> <nombre del vino> (<región>)",
    así que el año está disponible aunque no exista como columna propia.

    El detalle que arruina la extracción ingenua: muchas bodegas llevan un año
    en su propio nombre. Un patrón aplicado al título completo devuelve 1852
    para "Hazlitt 1852 Vineyards 2005 Cabernet Sauvignon", cuando la cosecha es
    2005. Afecta a unos 150 vinos, suficientes para ensuciar cualquier análisis
    temporal.

    La solución aprovecha que el nombre de la bodega ya está en su propia
    columna y que el título siempre empieza con él, cosa verificada sobre el
    100% de las filas: se recorta ese prefijo y recién entonces se busca el año.

    Las filas sin año son vinos espumantes rotulados "NV" (non-vintage), que
    son mezclas de varias cosechas y legítimamente no tienen uno. Quedan como
    nulos y no se imputan: inventarles un año sería fabricar un dato que no
    existe.
    """
    sin_bodega = [t[len(w):] if t.startswith(w) else t
                  for t, w in zip(df["title"], df["winery"])]

    anio = (pd.Series(sin_bodega, index=df.index)
              .str.extract(r"\b(19\d{2}|20[0-1]\d)\b")[0]
              .astype("Float64"))

    # Descarta años imposibles que hayan sobrevivido al recorte del prefijo.
    anio = anio.where(anio.between(ANIO_MINIMO, ANIO_MAXIMO))

    return df.assign(anio=anio.astype("Int64"))


def agregar_longitud(df):
    """Agrega la cantidad de palabras de cada nota de cata.

    `description` es texto libre y no admite las frecuencias que se usan con las
    demás variables, pero sí una medida simple: cuánto escribió el catador. Es
    la única variable que se puede derivar de esa columna sin entrar en
    procesamiento de lenguaje natural.

    Interesa porque la extensión de una reseña no la decide el vino sino quien
    la escribe, así que sirve para preguntar si los catadores se explayan más
    con los vinos que puntúan alto.
    """
    return df.assign(palabras=df["description"].str.split().str.len())


def analisis_longitud(df):
    """Relación entre el largo de la reseña, el puntaje y el precio.

    Devuelve las correlaciones y el promedio de palabras por tramo de puntaje.
    Se usa Spearman además de Pearson porque el precio sigue siendo asimétrico,
    y se trabaja sólo con precios observados para no medir contra valores
    imputados.
    """
    datos = df[~df["price_imputado"]].dropna(subset=["palabras"])

    correlaciones = pd.DataFrame({
        "Pearson": [stats.pearsonr(datos["palabras"], datos["points"])[0],
                    stats.pearsonr(datos["palabras"], np.log(datos["price"]))[0]],
        "Spearman": [stats.spearmanr(datos["palabras"], datos["points"])[0],
                     stats.spearmanr(datos["palabras"], datos["price"])[0]],
    }, index=["palabras vs puntaje", "palabras vs precio"]).round(3)

    por_puntaje = (datos.groupby("points")["palabras"]
                   .agg(reseñas="size", palabras_medias="mean")
                   .round(1))

    return correlaciones, por_puntaje


def analisis_temporal(df, minimo=500):
    """Puntaje y precio por año de cosecha.

    Se descartan las cosechas con pocas reseñas porque las más antiguas sufren
    un sesgo de supervivencia severo: de 1970 sólo se siguen reseñando los
    pocos vinos excepcionales que aún se venden, mientras que de 2014 se reseña
    la producción corriente. Comparar esas medias sin advertirlo llevaría a
    concluir que los vinos viejos son mejores, cuando lo que cambió es el
    criterio con el que llegaron a la muestra.
    """
    con_anio = df.dropna(subset=["anio"])
    resumen = con_anio.groupby("anio", observed=True).agg(
        reseñas=("points", "size"),
        puntaje_medio=("points", "mean"),
        precio_mediano=("price", "median"),
    )
    return resumen[resumen["reseñas"] >= minimo].round(2)


def efecto_catador(df, minimo=1500):
    """Separa la severidad del catador de los vinos que le tocaron reseñar.

    Comparar las medias crudas de los catadores sugiere diferencias enormes de
    criterio: 86,9 para el más bajo contra 90,6 para el más alto, una brecha
    mayor que el desvío estándar de toda la variable.

    Pero los catadores no se reparten los vinos al azar: cada uno cubre las
    regiones de su especialidad. El de media más baja reseña 43% España y 29%
    Chile; la de media más alta, 60% Austria y 38% Francia. La diferencia de
    puntajes puede ser entonces de los vinos y no de quien los juzga.

    Para separarlo se mide, para cada reseña, cuánto se aparta del promedio de
    su propio grupo de país y variedad. Ese desvío responde la pregunta
    correcta: frente al mismo tipo de vino, ¿este catador puntúa por encima o
    por debajo de lo habitual?

    La brecha de 3,77 puntos cae a 0,78, y el catador que parecía el más severo
    puntúa en el promedio exacto (-0,02). Casi toda la diferencia venía de qué
    vinos le tocaban, no de su criterio. Es confusión por una variable omitida,
    y comparar las medias crudas habría llevado a una conclusión equivocada
    sobre el trabajo de personas con nombre y apellido.
    """
    conocidos = df[df["taster_name"] != ETIQUETAS_FALTANTES["taster_name"]]
    grupo = conocidos.groupby(["country", "variety"], observed=True)["points"]
    desvio = (conocidos["points"] - grupo.transform("mean"))[
        grupo.transform("size") >= MINIMO_GRUPO]

    crudo = conocidos.groupby("taster_name", observed=True)["points"].agg(
        reseñas="size", media_cruda="mean")
    controlado = (conocidos.loc[desvio.index].assign(_d=desvio)
                  .groupby("taster_name", observed=True)["_d"]
                  .agg(comparables="size", desvio_controlado="mean"))

    resultado = crudo.join(controlado)
    resultado = resultado[resultado["reseñas"] >= minimo]
    return resultado.sort_values("media_cruda", ascending=False).round(2)


def mejor_relacion_calidad_precio(df, n=12):
    """Variedades con mejor puntaje relativo a su precio.

    Se compara la mediana de puntaje contra la mediana de precio por variedad,
    sobre precios observados y con un mínimo de reseñas por grupo. La mediana
    de precio, y no la media, porque dentro de cada variedad la asimetría sigue
    siendo fuerte.

    El indicador es descriptivo y no una recomendación de compra: mide qué
    variedades concentran puntajes altos en rangos de precio bajos, que es una
    característica del segmento de mercado, no una medida de calidad absoluta.
    """
    observados = df[~df["price_imputado"]]
    resumen = observados.groupby("variety", observed=True).agg(
        reseñas=("points", "size"),
        puntaje_mediano=("points", "median"),
        precio_mediano=("price", "median"),
    )
    resumen = resumen[resumen["reseñas"] >= MINIMO_GRUPO]
    resumen["puntos_por_dolar"] = (resumen["puntaje_mediano"]
                                   / resumen["precio_mediano"]).round(2)
    return resumen.sort_values("puntos_por_dolar", ascending=False).head(n)


def grafico_comparativo(df):
    """Dos comparaciones que sostienen las conclusiones del análisis.

    Izquierda: evolución del puntaje medio por cosecha, con el tamaño de
    muestra en el eje secundario. Las dos series van juntas para que se vea que
    los tramos más volátiles son justamente los de menos reseñas.

    Derecha: el efecto catador antes y después de controlar por país y
    variedad. Es la figura que muestra el hallazgo central del apartado: las
    barras crudas se despliegan sobre casi cuatro puntos y las controladas se
    comprimen alrededor de cero.
    """
    fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 5))

    temporal = analisis_temporal(df)
    izq.plot(temporal.index.astype(int), temporal["puntaje_medio"],
             color="#7b2d43", marker="o", ms=4, label="puntaje medio")
    izq.set(xlabel="Año de cosecha", ylabel="Puntaje medio",
            title="Puntaje por cosecha y tamaño de muestra")
    izq.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    volumen = izq.twinx()
    volumen.fill_between(temporal.index.astype(int), temporal["reseñas"],
                         color="#3d5a6c", alpha=0.18)
    volumen.set_ylabel("Reseñas (área)")
    izq.legend(loc="upper left", fontsize=8)

    catadores = efecto_catador(df).sort_values("media_cruda")
    etiquetas = [n.split()[0] if n else n for n in catadores.index]
    posicion = np.arange(len(catadores))

    der.barh(posicion - 0.2, catadores["media_cruda"] - df["points"].mean(),
             height=0.4, color="#7b2d43", label="media cruda (vs media general)")
    der.barh(posicion + 0.2, catadores["desvio_controlado"], height=0.4,
             color="#e07a00", label="controlado por país y variedad")
    der.axvline(0, color="black", lw=0.8)
    der.set_yticks(posicion)
    der.set_yticklabels(etiquetas, fontsize=8)
    der.set(xlabel="Desvío en puntos respecto del promedio",
            title="La severidad aparente era asignación de vinos")
    der.legend(fontsize=8)

    fig.tight_layout()
    return _guardar(fig, "08_comparativo")


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

    print("\n--- Resumen estadístico de las variables categóricas ---")
    print(resumen_categoricas(vinos).to_string())

    # Estos gráficos se generan ANTES del tratamiento de faltantes: el mapa de
    # nulos necesita ver los nulos, y los demás describen los datos tal como
    # llegaron, sin valores imputados mezclados.
    print("\n--- Gráficos sobre los datos sin imputar ---")
    for funcion in (grafico_puntajes, grafico_precios, grafico_categoricas,
                    grafico_precio_vs_puntaje, grafico_puntaje_por_pais,
                    grafico_faltantes):
        funcion(vinos)
        plt.close("all")

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

    print("\n--- Atípicos: comparación de criterios de detección ---")
    print(detectar_atipicos(vinos).to_string())

    print("\n--- Atípicos: precios implausibles contra su grupo de pares ---")
    print(atipicos_contextuales(vinos).head(6).to_string(index=False))

    print("\n--- Atípicos: impacto sobre la relación precio-puntaje ---")
    print(impacto_atipicos(vinos).to_string())

    # Este gráfico va después del tratamiento porque necesita `price_imputado`
    # para excluir los valores sintéticos del análisis de atípicos.
    grafico_atipicos(vinos)
    plt.close("all")

    print("\n--- Comparativo: año de cosecha derivado del título ---")
    vinos = extraer_anio(vinos)
    print(f"Año extraído en {vinos['anio'].notna().sum():,} reseñas "
          f"({vinos['anio'].isna().mean() * 100:.1f}% son espumantes sin cosecha)")
    print(analisis_temporal(vinos).tail(10).to_string())

    print("\n--- Comparativo: efecto catador, crudo contra controlado ---")
    print(efecto_catador(vinos).to_string())

    print("\n--- Comparativo: variedades con mejor puntaje por dólar ---")
    print(mejor_relacion_calidad_precio(vinos).to_string())

    print("\n--- Comparativo: largo de la reseña ---")
    vinos = agregar_longitud(vinos)
    correlaciones, palabras_por_puntaje = analisis_longitud(vinos)
    print(correlaciones.to_string())
    print(f"\nDe {palabras_por_puntaje.loc[80, 'palabras_medias']:.0f} palabras en 80 puntos "
          f"a {palabras_por_puntaje.loc[100, 'palabras_medias']:.0f} en 100.")

    grafico_comparativo(vinos)
    plt.close("all")

    print(f"\nGráficos guardados en {DIR_GRAFICOS.resolve()}")
