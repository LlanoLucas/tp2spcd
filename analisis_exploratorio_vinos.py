"""
TP2 - Seminario de Programación para Ciencia de Datos
Análisis exploratorio de las reseñas de vinos de WineEnthusiast.
Lucas Llano

Necesita pandas, numpy, matplotlib y scipy (y openpyxl si se usa el .xlsx).
El archivo de datos tiene que estar en la misma carpeta que este script.
Para correrlo: python analisis_exploratorio_vinos.py
"""

import csv
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Busco todo en la carpeta del script, así anda aunque se ejecute desde otro lado
CARPETA = Path(__file__).resolve().parent

# Uso el primer archivo que encuentre. El .xlsx es el que vino con el enunciado:
# es el mismo CSV, pero se rompió al abrirlo en Excel (ver sección 1).
RUTAS_POSIBLES = [
    CARPETA / "winemag-data-130k-v2.csv",
    CARPETA / "Entregable 2 - Documentación extra.xlsx",
]

DIR_GRAFICOS = CARPETA / "graficos"

COLUMNAS = ["country", "description", "designation", "points", "price", "province",
            "region_1", "region_2", "taster_name", "taster_twitter_handle", "title",
            "variety", "winery"]


# ===========================================================================
# 1. CARGA DE DATOS
# ===========================================================================

def reconstruir_csv(ruta):
    """Arma de nuevo el CSV original a partir del .xlsx que vino roto.

    Excel cortó las líneas en cada ';' (que en realidad era parte del texto de
    algunas reseñas) y partió en varias filas las reseñas con saltos de línea.
    """
    crudo = pd.read_excel(ruta)

    # read_excel tomó la primera línea como encabezado, así que la vuelvo a poner.
    lineas = [crudo.columns[0]]
    for fila in crudo.itertuples(index=False):
        # junto con ';' las celdas que Excel separó (las vacías vienen como NaN)
        lineas.append(";".join(c for c in fila if isinstance(c, str)))

    # Las líneas bien formadas empiezan con el índice y una coma ("48,US,...").
    # Si una no empieza así, es un pedazo de la reseña anterior.
    unidas = [lineas[0]]
    for linea in lineas[1:]:
        if re.match(r"^\d+,", linea):
            unidas.append(linea)
        else:
            unidas[-1] += "\n" + linea
    texto = "\n".join(unidas)

    # Chequeo: todas las líneas tienen que tener 14 campos (índice + 13 columnas).
    anchos = {len(campos) for campos in csv.reader(io.StringIO(texto))}
    assert anchos == {14}, f"la reconstrucción falló, anchos: {sorted(anchos)}"
    return texto


def arreglar_codificacion(texto):
    """Arregla los acentos rotos: 'GewÃ¼rztraminer' pasa a 'Gewürztraminer'.

    Esto pasa cuando un texto en UTF-8 se lee como si fuera cp1252.
    """
    if not isinstance(texto, str):
        return texto
    # Lo repito porque algunas celdas se rompieron dos veces. Si el texto ya
    # estaba bien, el decode falla y lo devuelvo como estaba.
    for _ in range(5):
        try:
            nuevo = texto.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return texto
        if nuevo == texto:
            return texto
        texto = nuevo
    return texto


def buscar_archivo():
    """Devuelve el primer archivo de datos que encuentre en la carpeta."""
    for ruta in RUTAS_POSIBLES:
        if ruta.exists():
            return ruta
    raise FileNotFoundError("No encontré ninguno de estos archivos en " + str(CARPETA)
                            + ": " + ", ".join(r.name for r in RUTAS_POSIBLES))


def cargar_datos(ruta=None):
    """Carga el dataset (el CSV o el .xlsx roto) y revisa que haya quedado bien."""
    ruta = Path(ruta) if ruta else buscar_archivo()

    if ruta.suffix.lower() == ".csv":
        df = pd.read_csv(ruta)
    else:
        df = pd.read_csv(io.StringIO(reconstruir_csv(ruta)))
        # los acentos solo vienen rotos en el .xlsx
        for col in df.select_dtypes(exclude="number").columns:
            df[col] = df[col].map(arreglar_codificacion)

    # La primera columna sin nombre es un índice que quedó al exportar el CSV
    if df.columns[0].startswith("Unnamed"):
        df = df.set_index(df.columns[0]).rename_axis(None)

    # Que no tire error no quiere decir que esté bien, así que reviso columnas y tipos
    faltan = set(COLUMNAS) - set(df.columns)
    assert not faltan, f"faltan columnas: {faltan}"
    assert df["points"].dtype.kind == "i", "points debería ser entero"
    assert df["price"].dtype.kind == "f", "price debería ser float"
    return df


# ===========================================================================
# 2. ESTRUCTURA Y RESUMEN ESTADÍSTICO
# ===========================================================================

def perfil_columnas(df):
    """Para cada columna: tipo, cantidad de nulos y cuántos valores distintos tiene."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "nulos_%": (df.isna().mean() * 100).round(2),
        "unicos": df.nunique(),
        "unicos_%": (df.nunique() / len(df) * 100).round(2),
    }).sort_values("nulos_%", ascending=False)


def quitar_duplicados(df):
    """Saca las filas repetidas (iguales en las 13 columnas).

    Seguramente vienen de cómo se bajaron los datos del sitio. Si las dejo,
    esas reseñas cuentan doble en todos los promedios.
    """
    antes = len(df)
    limpio = df.drop_duplicates()
    print(f"Duplicados eliminados: {antes - len(limpio):,} "
          f"({(antes - len(limpio)) / antes * 100:.1f}%) -> {len(limpio):,} reseñas")
    return limpio


def resumen_numericas(df):
    """Resumen de points y price.

    Además de lo que da describe() agrego el IQR y la asimetría, para ver si
    conviene resumir cada variable con la media o con la mediana.
    """
    num = df[["points", "price"]]
    return pd.DataFrame({
        "media": num.mean(),
        "mediana": num.median(),
        "desvio": num.std(),
        "min": num.min(),
        "max": num.max(),
        "IQR": num.quantile(0.75) - num.quantile(0.25),
        "asimetria": num.skew(),
    }).T.round(2)


def resumen_categoricas(df):
    """Moda, cuánto pesa la moda y cuánto suman las 10 categorías más comunes.

    Con eso se ve qué tan concentrada está cada variable. description y title
    no entran porque son texto libre.
    """
    columnas = ["country", "province", "region_1", "region_2", "variety",
                "winery", "designation", "taster_name"]
    filas = {}
    for col in columnas:
        frec = df[col].value_counts()
        prop = frec / frec.sum()
        filas[col] = {
            "categorías": len(frec),
            "moda": frec.index[0],
            "moda_%": round(prop.iloc[0] * 100, 1),
            "top10_%": round(prop.head(10).sum() * 100, 1),
        }
    return pd.DataFrame(filas).T


# ===========================================================================
# 3. GRÁFICOS
# ===========================================================================

def guardar_figura(fig, nombre):
    """Guarda el gráfico como PNG en la carpeta graficos/."""
    DIR_GRAFICOS.mkdir(exist_ok=True)
    fig.savefig(DIR_GRAFICOS / f"{nombre}.png", dpi=150, bbox_inches="tight")
    return fig


def grafico_puntajes(df):
    """Histograma de points, con una barra por cada puntaje."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = range(df["points"].min(), df["points"].max() + 2)
    ax.hist(df["points"], bins=bins, color="#7b2d43", edgecolor="white", align="left")

    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)  # dejo lugar arriba para los textos
    alto = ax.get_ylim()[1]
    media = df["points"].mean()
    ax.axvline(80, color="#c8102e", ls="--", lw=1.5)
    ax.text(80.4, alto * 0.94, "no se publican reseñas < 80",
            color="#c8102e", fontsize=9, va="top")
    ax.axvline(media, color="black", ls=":", lw=1.5)
    ax.text(media + 0.3, alto * 0.84, f"media {media:.1f}", fontsize=9, va="top")
    ax.set(xlabel="Puntaje", ylabel="Cantidad de reseñas",
           title="Distribución de puntajes")
    return guardar_figura(fig, "01_distribucion_puntajes")


def grafico_precios(df):
    """Histograma del precio en escala normal y en escala logarítmica."""
    precio = df["price"].dropna()
    fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.5))

    izq.hist(precio, bins=100, color="#7b2d43", edgecolor="white")
    izq.set(xlabel="Precio (USD)", ylabel="Cantidad de reseñas",
            title=f"Escala lineal (asimetría {precio.skew():.1f})")

    bins_log = np.logspace(np.log10(precio.min()), np.log10(precio.max()), 60)
    der.hist(precio, bins=bins_log, color="#3d5a6c", edgecolor="white")
    der.set_xscale("log")
    der.axvline(precio.median(), color="#c8102e", ls="--", lw=1.5)
    der.text(precio.median() * 1.1, der.get_ylim()[1] * 0.9,
             f"mediana ${precio.median():.0f}", color="#c8102e", fontsize=9)
    der.set(xlabel="Precio (USD, escala log)",
            title=f"Escala logarítmica (asimetría {np.log(precio).skew():.2f})")

    fig.suptitle("Distribución de precios", y=1.02)
    return guardar_figura(fig, "02_distribucion_precios")


def grafico_categoricas(df, n=15):
    """Las n categorías más comunes de country, variety, province y winery."""
    fig, ejes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(ejes.flat, ["country", "variety", "province", "winery"]):
        vc = df[col].value_counts().head(n).sort_values()
        cobertura = vc.sum() / df[col].notna().sum() * 100
        ax.barh(vc.index.astype(str), vc.values, color="#7b2d43")
        ax.set(xlabel="Reseñas",
               title=f"{col}: top {n} de {df[col].nunique()} ({cobertura:.0f}% del total)")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return guardar_figura(fig, "03_categoricas_top")


def grafico_precio_vs_puntaje(df):
    """Boxplot del precio para cada puntaje.

    Uso boxplot porque un scatter con 120 mil puntos queda como una mancha.
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
           title="Precio según puntaje")
    ax.legend()
    return guardar_figura(fig, "04_precio_vs_puntaje")


def grafico_puntaje_por_pais(df, n=12, minimo=500):
    """Boxplot de puntajes de los países con mejor mediana.

    Solo tomo países con al menos 500 reseñas, porque con pocas la mediana no
    es confiable.
    """
    conteos = df["country"].value_counts()
    candidatos = conteos[conteos >= minimo].index
    orden = (df[df["country"].isin(candidatos)]
             .groupby("country")["points"].median()
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
    ax.set(ylabel="Puntaje", title=f"Puntajes por país (mínimo {minimo} reseñas)")
    ax.grid(axis="y", alpha=0.3)
    return guardar_figura(fig, "05_puntaje_por_pais")


def grafico_faltantes(df):
    """Gráfico de barras con el porcentaje de nulos de cada columna."""
    pct = (df.isna().mean() * 100).sort_values()
    pct = pct[pct > 0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(pct.index, pct.values, color="#7b2d43")
    for i, v in enumerate(pct.values):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set(xlabel="% de valores faltantes", title="Faltantes por columna",
           xlim=(0, pct.max() * 1.2))
    ax.tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    return guardar_figura(fig, "06_faltantes")


# ===========================================================================
# 4. DATOS FALTANTES
# ===========================================================================

# En las categóricas el nulo casi siempre quiere decir "no aplica" (por ejemplo,
# region_2 solo existe para vinos de EE.UU.). Por eso no invento un valor, le
# pongo una etiqueta.
ETIQUETAS_FALTANTES = {
    "region_2": "No aplica",
    "region_1": "Sin región",
    "designation": "Sin designación",
    "taster_name": "No informado",
    "taster_twitter_handle": "No informado",
    "country": "Desconocido",
    "province": "Desconocido",
    "variety": "Desconocida",
}


def diagnostico_faltantes(df):
    """Porcentaje de precios faltantes por país y por puntaje.

    Sirve para ver si el precio falta al azar o no. Si depende de otras
    variables (MAR), imputar con la mediana general da mal y conviene hacerlo
    por grupos.
    """
    falta = df["price"].isna()
    por_pais = (df.assign(falta=falta).groupby("country")["falta"]
                  .agg(reseñas="size", sin_precio_pct=lambda s: round(s.mean() * 100, 1)))
    por_pais = por_pais[por_pais["reseñas"] >= 1000].sort_values(
        "sin_precio_pct", ascending=False)
    por_puntaje = (df.assign(falta=falta).groupby("points")["falta"].mean() * 100).round(1)
    return por_pais, por_puntaje


def validar_imputacion_precio(df, semilla=42):
    """Prueba varios métodos de imputación con precios que sí conozco.

    Escondo algunos precios (más en los países donde de verdad faltan más, para
    que se parezca al caso real), los imputo con cada método y comparo con el
    valor verdadero.
    """
    rng = np.random.default_rng(semilla)
    tasa = df["price"].isna().groupby(df["country"]).mean()

    ocultar = []
    for pais, grupo in df[df["price"].notna()].groupby("country"):
        cuantos = int(len(grupo) * tasa.get(pais, 0.07))
        if cuantos:
            ocultar += list(rng.choice(grupo.index, size=cuantos, replace=False))

    real = df.loc[ocultar, "price"]
    prueba = df.copy()
    prueba.loc[ocultar, "price"] = np.nan

    def mediana_por(*columnas):
        return prueba.groupby(list(columnas))["price"].transform("median")

    metodos = {
        "media global": prueba["price"].fillna(prueba["price"].mean()),
        "mediana global": prueba["price"].fillna(prueba["price"].median()),
        "mediana por país": prueba["price"].fillna(mediana_por("country")),
        "mediana por variedad": prueba["price"].fillna(mediana_por("variety")),
        "mediana por país+variedad": prueba["price"].fillna(mediana_por("country", "variety")),
        "mediana por país+variedad+puntaje": prueba["price"].fillna(
            mediana_por("country", "variety", "points")),
    }

    filas = {}
    for nombre, estimado in metodos.items():
        # si un grupo se quedó sin precios, uso la mediana general
        error = estimado.fillna(prueba["price"].median()).loc[ocultar] - real
        filas[nombre] = {
            "MAE": error.abs().mean(),
            "error_mediano": error.abs().median(),
            "error_%_mediano": (error.abs() / real * 100).median(),
        }
    return pd.DataFrame(filas).T.round(2)


def imputar_precio(df):
    """Completa price con la mediana del grupo país + variedad + puntaje.

    Fue el método que mejor dio en la validación (error medio de $14,92 contra
    $21,96 de la mediana general). Si el grupo no tiene datos, uso uno más
    general. En price_imputado queda marcado qué precios completé yo.
    """
    df = df.copy()
    df["price_imputado"] = df["price"].isna()

    def mediana_por(*columnas):
        return df.groupby(list(columnas))["price"].transform("median")

    df["price"] = (df["price"]
                   .fillna(mediana_por("country", "variety", "points"))
                   .fillna(mediana_por("country", "variety"))
                   .fillna(mediana_por("variety"))
                   .fillna(mediana_por("country"))
                   .fillna(df["price"].median()))
    return df


def tratar_faltantes(df):
    """Pone las etiquetas en las categóricas y después imputa el precio (no borro filas)."""
    df = df.copy()
    for col, etiqueta in ETIQUETAS_FALTANTES.items():
        df[col] = df[col].fillna(etiqueta)
    return imputar_precio(df)


# ===========================================================================
# 5. DATOS ATÍPICOS
# ===========================================================================

# Un precio me parece sospechoso si es más de 10 veces la mediana de sus pares:
# que un vino cueste diez veces lo que cuestan vinos parecidos es muy raro.
UMBRAL_RATIO = 10
# Con menos de 30 vinos en el grupo la mediana no es confiable
MINIMO_PARES = 30


def detectar_atipicos(df):
    """Porcentaje de valores que el IQR marca como atípicos.

    Uso el IQR porque trabaja con cuartiles, que no se mueven por los valores
    extremos, y no hace falta que los datos sean normales. Como el precio es muy
    asimétrico, también lo aplico sobre el logaritmo.
    Uso solo los precios observados, porque los imputados están cerca de la
    mediana y nunca van a salir como atípicos.
    """
    def pct_iqr(x):
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        return ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() * 100

    observados = df[~df["price_imputado"]]
    filas = {}
    for variable in ["price", "points"]:
        x = observados[variable]
        filas[variable] = {
            "IQR": pct_iqr(x),
            "IQR sobre log": pct_iqr(np.log(x)),
        }
    return pd.DataFrame(filas).round(2)


def ratio_contra_pares(observados):
    """Divide cada precio por la mediana de los vinos de su provincia y puntaje."""
    grupo = observados.groupby(["province", "points"])["price"]
    mediana = grupo.transform("median")
    confiable = grupo.transform("size") >= MINIMO_PARES
    return (observados["price"] / mediana)[confiable], mediana


def atipicos_contextuales(df, umbral=UMBRAL_RATIO):
    """Vinos con un precio muy alto comparado con vinos parecidos.

    Un Pétrus a $2.500 es caro pero real, y un Médoc común a $3.300 no tiene
    sentido. Para diferenciarlos comparo con sus pares. Los casos quedan para
    revisarlos a mano, no los corrijo solos.
    """
    observados = df[~df["price_imputado"]]
    ratio, mediana = ratio_contra_pares(observados)

    sospechosos = observados.loc[ratio[ratio > umbral].index].copy()
    sospechosos["mediana_pares"] = mediana[sospechosos.index]
    sospechosos["ratio"] = ratio[sospechosos.index].round(1)
    columnas = ["title", "province", "points", "price", "mediana_pares", "ratio"]
    return sospechosos[columnas].sort_values("ratio", ascending=False)


def impacto_atipicos(df):
    """Correlación entre precio y puntaje con atípicos, sin ellos y con log(precio).

    Así veo si conviene sacar los atípicos o cambiar de escala. Uso solo precios
    observados porque los imputados se calcularon a partir de points.
    """
    observados = df[~df["price_imputado"]]
    precio, puntaje = observados["price"], observados["points"]

    q1, q3 = precio.quantile([0.25, 0.75])
    recortado = observados[precio <= q3 + 1.5 * (q3 - q1)]

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
    """Izquierda: dónde corta cada criterio en el precio. Derecha: cociente con los pares."""
    observados = df[~df["price_imputado"]]
    precio = observados["price"]

    q1, q3 = precio.quantile([0.25, 0.75])
    limite_iqr = q3 + 1.5 * (q3 - q1)
    lq1, lq3 = np.log(precio).quantile([0.25, 0.75])
    limite_log = np.exp(lq3 + 1.5 * (lq3 - lq1))

    fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.5))

    izq.hist(precio, bins=np.logspace(np.log10(precio.min()), np.log10(precio.max()), 70),
             color="#3d5a6c", edgecolor="white")
    izq.set_xscale("log")
    izq.axvline(limite_iqr, color="#c8102e", ls="--",
                label=f"IQR clásico: ${limite_iqr:.0f} ({(precio > limite_iqr).mean() * 100:.1f}%)")
    izq.axvline(limite_log, color="#e07a00", ls="--",
                label=f"IQR sobre log: ${limite_log:.0f} ({(precio > limite_log).mean() * 100:.1f}%)")
    izq.set(xlabel="Precio (USD, escala log)", ylabel="Cantidad de reseñas",
            title="Umbrales de atípicos en el precio")
    izq.legend(fontsize=8)

    ratio, _ = ratio_contra_pares(observados)
    der.hist(ratio, bins=np.logspace(np.log10(ratio.min()), np.log10(ratio.max()), 70),
             color="#7b2d43", edgecolor="white")
    der.set_xscale("log")
    der.set_yscale("log")
    der.axvline(UMBRAL_RATIO, color="#c8102e", ls="--",
                label=f"umbral {UMBRAL_RATIO}x ({(ratio > UMBRAL_RATIO).sum()} casos)")
    der.set(xlabel="Precio / mediana de los pares (provincia + puntaje)",
            ylabel="Cantidad de reseñas", title="Precio comparado con sus pares")
    der.legend(fontsize=8)

    fig.tight_layout()
    return guardar_figura(fig, "07_atipicos")


# ===========================================================================
# 6. ANÁLISIS COMPARATIVO
# ===========================================================================

# Mínimo de reseñas para que un grupo entre en las comparaciones
MINIMO_GRUPO = 200


def extraer_anio(df):
    """Saca el año de cosecha del título.

    Antes le saco el nombre de la bodega, porque algunas tienen un año en el
    nombre ("Hazlitt 1852 Vineyards 2005 ..." daría 1852 y no 2005).
    Los que no tienen año son espumantes "NV" y los dejo como nulos.
    """
    sin_bodega = [t[len(w):] if t.startswith(w) else t
                  for t, w in zip(df["title"], df["winery"])]
    anio = pd.to_numeric(pd.Series(sin_bodega, index=df.index)
                         .str.extract(r"\b(19\d{2}|20[0-1]\d)\b")[0])
    anio = anio.where(anio.between(1900, 2017))  # el dataset es de 2017
    return df.assign(anio=anio.astype("Int64"))


def agregar_longitud(df):
    """Agrega cuántas palabras tiene cada reseña."""
    return df.assign(palabras=df["description"].str.split().str.len())


def analisis_longitud(df):
    """Correlación entre el largo de la reseña y el puntaje o el precio."""
    datos = df[~df["price_imputado"]]
    correlaciones = pd.DataFrame({
        "Pearson": [stats.pearsonr(datos["palabras"], datos["points"])[0],
                    stats.pearsonr(datos["palabras"], np.log(datos["price"]))[0]],
        "Spearman": [stats.spearmanr(datos["palabras"], datos["points"])[0],
                     stats.spearmanr(datos["palabras"], datos["price"])[0]],
    }, index=["palabras vs puntaje", "palabras vs precio"]).round(3)

    por_puntaje = (datos.groupby("points")["palabras"]
                   .agg(reseñas="size", palabras_medias="mean").round(1))
    return correlaciones, por_puntaje


def analisis_temporal(df, minimo=500):
    """Puntaje medio y precio mediano por cosecha (años con 500 reseñas o más)."""
    resumen = df.dropna(subset=["anio"]).groupby("anio").agg(
        reseñas=("points", "size"),
        puntaje_medio=("points", "mean"),
        precio_mediano=("price", "median"),
    )
    return resumen[resumen["reseñas"] >= minimo].round(2)


def efecto_catador(df, minimo=1500):
    """Promedio de cada catador, crudo y controlando por país y variedad.

    Cada catador reseña vinos de ciertas regiones, así que su promedio crudo
    mezcla su criterio con los vinos que le tocaron. Para separar eso, mido
    cuánto se aleja cada reseña del promedio de los vinos del mismo país y
    variedad.
    """
    conocidos = df[df["taster_name"] != ETIQUETAS_FALTANTES["taster_name"]]
    grupo = conocidos.groupby(["country", "variety"])["points"]
    desvio = (conocidos["points"] - grupo.transform("mean"))[
        grupo.transform("size") >= MINIMO_GRUPO]

    crudo = conocidos.groupby("taster_name")["points"].agg(
        reseñas="size", media_cruda="mean")
    controlado = (desvio.groupby(conocidos.loc[desvio.index, "taster_name"])
                  .agg(comparables="size", desvio_controlado="mean"))

    resultado = crudo.join(controlado)
    resultado = resultado[resultado["reseñas"] >= minimo]
    return resultado.sort_values("media_cruda", ascending=False).round(2)


def grafico_comparativo(df):
    """Izquierda: puntaje medio por cosecha. Derecha: efecto del catador."""
    fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 5))

    temporal = analisis_temporal(df)
    anios = temporal.index.astype(int)
    izq.plot(anios, temporal["puntaje_medio"], color="#7b2d43", marker="o", ms=4,
             label="puntaje medio")
    izq.set(xlabel="Año de cosecha", ylabel="Puntaje medio",
            title="Puntaje medio por cosecha")
    izq.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    volumen = izq.twinx()
    volumen.fill_between(anios, temporal["reseñas"], color="#3d5a6c", alpha=0.18)
    volumen.set_ylabel("Reseñas (área)")
    izq.legend(loc="upper left", fontsize=8)

    catadores = efecto_catador(df).sort_values("media_cruda")
    posicion = np.arange(len(catadores))
    der.barh(posicion - 0.2, catadores["media_cruda"] - df["points"].mean(),
             height=0.4, color="#7b2d43", label="media cruda (vs media general)")
    der.barh(posicion + 0.2, catadores["desvio_controlado"], height=0.4,
             color="#e07a00", label="controlado por país y variedad")
    der.axvline(0, color="black", lw=0.8)
    der.set_yticks(posicion)
    der.set_yticklabels([n.split()[0] for n in catadores.index], fontsize=8)
    der.set(xlabel="Diferencia en puntos respecto del promedio",
            title="Efecto del catador")
    der.legend(fontsize=8)

    fig.tight_layout()
    return guardar_figura(fig, "08_comparativo")


# ===========================================================================
# EJECUCIÓN
# ===========================================================================

if __name__ == "__main__":
    # --- 1. Carga ---
    vinos = cargar_datos()
    print(f"Dataset cargado: {vinos.shape[0]:,} reseñas x {vinos.shape[1]} columnas")
    print(vinos.dtypes.to_string())

    # --- 2. Estructura y resumen ---
    print("\n--- Perfil de columnas ---")
    print(perfil_columnas(vinos).to_string())

    print("\n--- Duplicados ---")
    vinos = quitar_duplicados(vinos)

    print("\n--- Resumen de variables numéricas ---")
    print(resumen_numericas(vinos).to_string())

    print("\n--- Resumen de variables categóricas ---")
    print(resumen_categoricas(vinos).to_string())

    # --- 3. Gráficos (antes de imputar, para que muestren los datos originales) ---
    for funcion in (grafico_puntajes, grafico_precios, grafico_categoricas,
                    grafico_precio_vs_puntaje, grafico_puntaje_por_pais,
                    grafico_faltantes):
        funcion(vinos)
        plt.close("all")

    # --- 4. Faltantes ---
    print("\n--- Faltantes por columna (sin duplicados) ---")
    faltantes = vinos.isna().sum()
    print(pd.DataFrame({"faltantes": faltantes,
                        "%": (faltantes / len(vinos) * 100).round(2)})
          .query("faltantes > 0").sort_values("faltantes", ascending=False).to_string())

    # Columnas que faltan juntas
    sin_catador = vinos["taster_name"].isna()
    sin_precio = vinos["price"].isna()
    pct_twitter = vinos.loc[sin_catador, "taster_twitter_handle"].isna().mean() * 100
    pct_region = vinos.loc[sin_precio, "region_2"].isna().mean() * 100
    print(f"\nDe las reseñas sin catador, el {pct_twitter:.0f}% tampoco tiene Twitter")
    print(f"De las reseñas sin precio, el {pct_region:.0f}% tampoco tiene region_2")

    por_pais, por_puntaje = diagnostico_faltantes(vinos)
    print("\nPrecio faltante por país (países con al menos 1000 reseñas):")
    print(por_pais.to_string())
    print(f"\nPrecio faltante según puntaje: {por_puntaje.loc[80]}% en 80 "
          f"-> {por_puntaje.loc[99]}% en 99")

    print("\n--- Validación de métodos de imputación ---")
    print(validar_imputacion_precio(vinos).to_string())

    vinos = tratar_faltantes(vinos)
    print(f"\nPrecios imputados: {vinos['price_imputado'].sum():,} "
          f"({vinos['price_imputado'].mean() * 100:.1f}%)")
    print(f"Nulos restantes: {vinos.isna().sum().sum()}")

    # --- 5. Atípicos ---
    print("\n--- Atípicos: porcentaje marcado por el IQR ---")
    print(detectar_atipicos(vinos).to_string())

    print("\n--- Atípicos: precios muy altos contra sus pares ---")
    print(atipicos_contextuales(vinos).head(6).to_string(index=False))

    print("\n--- Atípicos: impacto en la correlación precio-puntaje ---")
    print(impacto_atipicos(vinos).to_string())

    grafico_atipicos(vinos)
    plt.close("all")

    # --- 6. Comparativo ---
    vinos = extraer_anio(vinos)
    print(f"\nAño extraído en {vinos['anio'].notna().sum():,} reseñas "
          f"({vinos['anio'].isna().mean() * 100:.1f}% sin año)")
    print(analisis_temporal(vinos).tail(10).to_string())

    print("\n--- Efecto del catador ---")
    print(efecto_catador(vinos).to_string())

    print("\n--- Largo de la reseña ---")
    vinos = agregar_longitud(vinos)
    correlaciones, por_puntaje = analisis_longitud(vinos)
    print(correlaciones.to_string())
    print(f"\nPalabras promedio: {vinos['palabras'].mean():.0f} "
          f"(mín {vinos['palabras'].min()}, máx {vinos['palabras'].max()})")
    print(f"De {por_puntaje.loc[80, 'palabras_medias']:.0f} palabras en 80 puntos "
          f"a {por_puntaje.loc[100, 'palabras_medias']:.0f} en 100")

    grafico_comparativo(vinos)
    plt.close("all")

    print(f"\nGráficos guardados en {DIR_GRAFICOS.resolve()}")
