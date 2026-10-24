# Electromagnetismo I — Tema 3: Conductores en equilibrio electrostático

Apuntes formalizados (LaTeX) con seminarios prácticos, en la línea de los temas 0, 1 y 2 en cuanto a entornos formales
(definición, teorema, proposición, lema, corolario, observación, demostración), notación y estilo gráfico, y con la
tipografía, márgenes y espaciado del informe de referencia: clase `article` a 11 pt, Latin Modern y márgenes de 2,5 cm.
La paleta es de **azules sobrios con degradados suaves** (filetes de sección, definiciones, teoremas, ecuaciones
destacadas, portada); los únicos acentos no azules son el rojo de las cargas positivas y un verde azulado para las normales.

## Compilación

```bash
latexmk -pdf main.tex      # pdfLaTeX (-shell-escape, ya en latexmkrc) + biber
```

* **Overleaf:** subir el ZIP como proyecto nuevo (*New Project → Upload Project*), compilador *pdfLaTeX* y documento
  principal `main.tex`. La bibliografía usa *biber*, que Overleaf detecta solo.
* **Figuras externalizadas.** Cada figura TikZ/pgfplots se compila una vez a PDF en `figuras/ext/` (librería
  `external`, comprobación MD5). Con esos PDF ya incluidos la compilación tarda unos segundos; desde cero, unos
  4 minutos. Si se modifica una figura, se regenera sola en la siguiente compilación.

## Figuras «tipo Mathematica»

Las figuras numéricas del tema se calculan con `scripts/generar_datos.py` (requiere `numpy`, `scipy` y `matplotlib`) y
se dibujan con TikZ a partir de coordenadas precalculadas en `figuras/datos/` (ya incluidas):

```bash
python3 scripts/generar_datos.py
```

| Datos | Problema físico | Método |
|---|---|---|
| `esfera_uniforme_*` | Esfera conductora neutra en un campo uniforme | Analítico (función de corriente de Stokes) |
| `irregular_uniforme_*` | Conductor neutro irregular en un campo uniforme | Laplace 2D por diferencias finitas, carga neta nula |
| `punta_*` | Conductor cargado con punta (efecto punta) | Laplace 2D; σ en el contorno |
| `corteza_{a,b,c,d}_*` | Corteza esférica con carga en la cavidad y/o carga exterior | Método de las imágenes (exacto) |
| `sistema_*` | Cuatro conductores en una caja a tierra: estados elementales, $c_{ij}$, $p_{ij}$ | Laplace 2D, Gauss discreto |

Las líneas de campo se integran con RK4 y se siembran con **flujo constante entre líneas consecutivas**, de modo que su
densidad es proporcional al campo, como en `StreamPlot`. Los mapas de potencial y de módulo del campo son PNG en la
escala azul del documento; las equipotenciales y los contornos de los conductores se dibujan en vectorial encima.

## Estructura

| Fichero | Contenido |
|---|---|
| `main.tex`, `portada.tex`, `preambulo.tex`, `latexmkrc` | Documento, portada tipo artículo (resumen, palabras clave, PACS, emblema numérico) y preámbulo |
| `secciones/01_introduccion.tex` | Conductor perfecto, equilibrio electrostático, campo nulo en el interior, notación |
| `secciones/02_propiedades.tex` | Propiedades básicas, líneas de campo, teorema de Thomson, efecto punta, pararrayos |
| `secciones/03_presion.tex` | Presión electrostática: campo del resto de la carga, $\sigma^2/2\varepsilon_0$, hemisferios |
| `secciones/04_cavidades.tex` | Poisson y Laplace, principio del máximo, unicidad, jaula de Faraday, regla de las caras, apantallamiento, capacidad |
| `secciones/05_sistemas.tex` | Coeficientes de potencial, reciprocidad de Green, reglas prácticas, coeficientes de capacidad e influencia, influencia total, comprobación numérica |
| `secciones/06_condensadores.tex` | Condensador, capacidad con coeficientes, conductor encerrado, condensadores esféricos (delgado y grueso), canónicos |
| `secciones/07_seminarios.tex` | Métodos M1–M5 y problemas 3.1–3.7 de la relación, ejemplo modelo, resumen y lista de control |
| `secciones/08_apendice.tex` | Formulario del tema |
| `figuras/` | Figuras (TikZ / pgfplots), `estilos.tex`, `tab_sistema.tex`; `datos/`, datos numéricos; `ext/`, figuras compiladas |
| `scripts/generar_datos.py` | Generador documentado de los datos numéricos |
| `referencias.bib` | Bibliografía |
