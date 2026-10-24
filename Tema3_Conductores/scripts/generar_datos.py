#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electromagnetismo I - Tema 3: Conductores en equilibrio electrostático
=======================================================================

Generador de los datos numéricos de las figuras «tipo Mathematica» del tema.
El script NO maqueta nada: calcula curvas y mapas y los escribe como llamadas
a macros TikZ en ``figuras/datos/*.tex`` (más algunos mapas de color en PNG y
tablas ``.dat``); el estilo (colores, grosores, puntas de flecha) se fija en
LaTeX, en ``figuras/estilos.tex`` y en cada ``figuras/fig_*.tex``.

Objetos que se escriben (todos en coordenadas del dibujo):

* ``\\LC{(x,y) (x,y) ...}``      línea de campo;
* ``\\EQ{(x,y) (x,y) ...}``      equipotencial (curva de nivel de phi);
* ``\\FL{x}{y}{ang}``            punta de flecha sobre una línea de campo;
* ``\\CP{x}{y}`` / ``\\CN{x}{y}``  signo + / - de la carga inducida;
* ``\\SG{x1}{y1}{x2}{y2}{t}``    tramo de superficie coloreado por sigma (t en [0,1]).

Problemas que se resuelven
--------------------------
1. Conductor esférico neutro en un campo uniforme (solución analítica en el
   plano meridiano; las líneas de campo son curvas de nivel de la función de
   corriente de Stokes).
2. Conductor neutro de forma arbitraria en un campo uniforme y conductor
   cargado con una punta (efecto punta): ecuación de Laplace en 2D resuelta por
   diferencias finitas (5 puntos) con condiciones de Dirichlet.
3. Corteza esférica gruesa con una carga en la cavidad y/o una carga exterior:
   método de las imágenes (solución exacta), como en las figuras del tema.
4. Sistema de cuatro conductores dentro de una caja a tierra: los cuatro
   estados elementales phi_j = 1, phi_k = 0 (k != j) dan, por el teorema de
   Gauss discreto, la matriz de coeficientes de capacidad e influencia c_ij;
   su inversa es la matriz de coeficientes de potencial p_ij. También se
   calcula el estado «conductor j cargado, los demás neutros».

Unidades: eps0 = 1 y longitudes adimensionales. Los problemas 2 y 4 son
bidimensionales (sección de conductores cilíndricos indefinidos): las cargas
son por unidad de longitud.

Uso:  python3 scripts/generar_datos.py        (desde la carpeta del proyecto)
Requiere numpy, scipy y matplotlib (solo para las curvas de nivel y los PNG).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import scipy.sparse as sp
import scipy.sparse.linalg as spla

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "..", "figuras", "datos")
os.makedirs(SALIDA, exist_ok=True)

# Escalas cromáticas del documento (las mismas que en preambulo.tex)
AZUL_POT = LinearSegmentedColormap.from_list(
    "azulpot", ["#F5F8FC", "#DCE7F5", "#9DB9DD", "#3A6EA5", "#1F4E89", "#0B2545"])
AZUL_CAMPO = LinearSegmentedColormap.from_list(
    "azulcampo", ["#0B2545", "#1F4E89", "#3A86B5", "#7FC3D9", "#CFEAF1", "#FFFFFF"])


# ===========================================================================
#  Utilidades de escritura
# ===========================================================================
def simplifica(pts, d=0.03):
    out = [pts[0]]
    for p in pts[1:-1]:
        if (p[0] - out[-1][0]) ** 2 + (p[1] - out[-1][1]) ** 2 >= d * d:
            out.append(p)
    out.append(pts[-1])
    return out


def coords(pts, d=0.03):
    return " ".join("(%.3f,%.3f)" % (x, y) for x, y in simplifica(list(pts), d))


def escribe(nombre, cabecera, cuerpo):
    with open(os.path.join(SALIDA, nombre), "w", encoding="utf-8") as f:
        f.write("%% %s\n%% Generado por scripts/generar_datos.py (no editar a mano).\n"
                % cabecera)
        f.write("\n".join(cuerpo) + "\n")
    print("  ->", nombre, "(%d objetos)" % len(cuerpo))


def curvas_nivel(X, Y, V, niveles, mascara=None):
    if mascara is not None:
        V = np.ma.masked_where(mascara, V)
    fig = plt.figure()
    cs = plt.contour(X, Y, V, levels=sorted(niveles))
    cur = [seg for segs in cs.allsegs for seg in segs if len(seg) > 3]
    plt.close(fig)
    return cur


# ===========================================================================
#  Integración de líneas de campo (RK4 con paso de longitud de arco fijo)
# ===========================================================================
class Rejilla:
    """Tabla espacial para comprobar la separación mínima entre líneas."""

    def __init__(self, celda):
        self.c = celda
        self.cel = {}

    def _k(self, x, y):
        return (int(np.floor(x / self.c)), int(np.floor(y / self.c)))

    def libre(self, x, y, d):
        i, j = self._k(x, y)
        for a in (i - 1, i, i + 1):
            for b in (j - 1, j, j + 1):
                for (px, py) in self.cel.get((a, b), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < d * d:
                        return False
        return True

    def anade(self, pts):
        for x, y in pts:
            self.cel.setdefault(self._k(x, y), []).append((x, y))


def traza(E, x0, y0, sgn, para, h=0.01, nmax=4000):
    """Integra dr/ds = sgn E/|E| desde (x0, y0) hasta que para(x, y) es cierto."""
    def u(x, y):
        ex, ey = E(x, y)
        m = np.hypot(ex, ey)
        if not np.isfinite(m) or m < 1e-12:
            return None
        return sgn * ex / m, sgn * ey / m
    pts = [(x0, y0)]
    x, y = x0, y0
    for _ in range(nmax):
        k1 = u(x, y)
        if k1 is None:
            break
        k2 = u(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1])
        if k2 is None:
            break
        k3 = u(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1])
        if k3 is None:
            break
        k4 = u(x + h * k3[0], y + h * k3[1])
        if k4 is None:
            break
        x += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        y += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        pts.append((x, y))
        if para(x, y):
            break
    return pts


def flechas(lineas, E, fraccion=(0.5,), minimo=0.25, paso=None):
    """Puntas de flecha en las fracciones de longitud de arco indicadas (o cada
    `paso` unidades de longitud), orientadas según el campo."""
    out = []
    for pts in lineas:
        P = np.array(pts)
        if len(P) < 3:
            continue
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        L = s[-1]
        if L < minimo:
            continue
        fr = fraccion if paso is None else [(k + 0.5) / max(1, int(L / paso))
                                            for k in range(max(1, int(L / paso)))]
        for f in fr:
            i = min(max(int(np.searchsorted(s, f * L)), 1), len(P) - 1)
            x, y = P[i]
            ex, ey = E(x, y)
            out.append((x, y, np.degrees(np.arctan2(ey, ex))))
    return out


def reparte(contorno, peso, n_o_delta, cerrado=True):
    """Puntos de un contorno repartidos a incrementos iguales de la integral de
    `peso` (>= 0) a lo largo del arco. Devuelve índices de los puntos."""
    P = np.asarray(contorno)
    seg = np.hypot(*np.diff(np.vstack([P, P[:1]]) if cerrado else P, axis=0).T)
    w = np.asarray(peso)[:len(seg)] * seg
    acum = np.concatenate([[0], np.cumsum(w)])
    tot = acum[-1]
    if tot <= 0:
        return []
    if isinstance(n_o_delta, int):
        n = n_o_delta
    else:
        n = int(round(tot / n_o_delta))
    if n <= 0:
        return []
    obj = (np.arange(n) + 0.5) * tot / n
    return [min(int(np.searchsorted(acum, o)), len(P) - 1) for o in obj]


# ===========================================================================
#  Solver de Laplace 2D por diferencias finitas
# ===========================================================================
class Laplace2D:
    """Malla uniforme [x0, x1] x [y0, y1] con paso h. Los nodos de los
    conductores y del borde de la caja son de Dirichlet; los demás, incógnitas.
    Se factoriza una sola vez y se resuelve para varios segundos miembros."""

    def __init__(self, x0, x1, y0, y1, h, conductores):
        self.h = h
        self.x = np.arange(x0, x1 + h / 2, h)
        self.y = np.arange(y0, y1 + h / 2, h)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        ny, nx = self.X.shape
        self.nx, self.ny = nx, ny
        self.masc = []                    # máscara de cada conductor
        for c in conductores:
            self.masc.append(c.dentro(self.X, self.Y))
        cond = np.zeros_like(self.X, dtype=bool)
        for m in self.masc:
            cond |= m
        borde = np.zeros_like(cond)
        borde[0, :] = borde[-1, :] = borde[:, 0] = borde[:, -1] = True
        self.cond, self.borde = cond, borde
        libre = ~(cond | borde)
        self.libre = libre
        idx = -np.ones(cond.shape, dtype=int)
        idx[libre] = np.arange(libre.sum())
        self.idx = idx
        filas, cols, vals = [], [], []
        jj, ii = np.nonzero(libre)
        k = idx[jj, ii]
        filas.append(k); cols.append(k); vals.append(np.full(k.size, -4.0))
        self.vecinos = []
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            vj, vi = jj + dj, ii + di
            l = libre[vj, vi]
            filas.append(k[l]); cols.append(idx[vj[l], vi[l]]); vals.append(np.ones(l.sum()))
            self.vecinos.append((k[~l], vj[~l], vi[~l]))
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                          shape=(libre.sum(), libre.sum()))
        print("    factorizando %d incógnitas..." % libre.sum())
        self.lu = spla.splu(A)

    def resuelve(self, dirichlet):
        """`dirichlet`: array con los valores en los nodos de Dirichlet."""
        b = np.zeros(self.libre.sum())
        for k, vj, vi in self.vecinos:
            np.add.at(b, k, -dirichlet[vj, vi])
        V = dirichlet.copy().astype(float)
        V[self.libre] = self.lu.solve(b)
        return V

    def carga(self, V, i):
        """Carga (por unidad de longitud, eps0 = 1) del conductor i por el
        teorema de Gauss discreto: Q = -sum(lap_h V) h^2 sobre sus nodos."""
        lap = np.zeros_like(V)
        lap[1:-1, 1:-1] = (V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2]
                           - 4 * V[1:-1, 1:-1])
        return -lap[self.masc[i]].sum()

    def campo(self, V):
        """Interpolador bilineal de E = -grad V."""
        gy, gx = np.gradient(V, self.h)
        Ex, Ey = -gx, -gy
        x0, y0, h, nx, ny = self.x[0], self.y[0], self.h, self.nx, self.ny

        def E(x, y):
            fx, fy = (x - x0) / h, (y - y0) / h
            i, j = int(np.floor(fx)), int(np.floor(fy))
            if i < 0 or j < 0 or i >= nx - 1 or j >= ny - 1:
                return 0.0, 0.0
            tx, ty = fx - i, fy - j
            def b(F):
                return ((1 - tx) * (1 - ty) * F[j, i] + tx * (1 - ty) * F[j, i + 1]
                        + (1 - tx) * ty * F[j + 1, i] + tx * ty * F[j + 1, i + 1])
            return b(Ex), b(Ey)
        return E, np.hypot(Ex, Ey)


class Poligono:
    def __init__(self, vertices, nombre=""):
        self.v = np.asarray(vertices, dtype=float)
        self.path = Path(self.v, closed=False)
        self.nombre = nombre

    def dentro(self, X, Y):
        pts = np.column_stack([X.ravel(), Y.ravel()])
        return self.path.contains_points(pts, radius=1e-9).reshape(X.shape)

    def contiene(self, x, y):
        return self.path.contains_point((x, y))

    def contorno(self, ds=0.01):
        """Puntos densos del contorno (antihorario) y normales exteriores."""
        V = self.v if np.allclose(self.v[0], self.v[-1]) else np.vstack([self.v, self.v[:1]])
        area = 0.5 * np.sum(V[:-1, 0] * V[1:, 1] - V[1:, 0] * V[:-1, 1])
        if area < 0:
            V = V[::-1]
        P, N = [], []
        for a, b in zip(V[:-1], V[1:]):
            L = np.hypot(*(b - a))
            n = max(1, int(L / ds))
            t = (b - a) / L
            for k in range(n):
                P.append(a + (b - a) * k / n)
                N.append((t[1], -t[0]))
        return np.array(P), np.array(N)

    def tikz(self):
        return " -- ".join("(%.3f,%.3f)" % tuple(p) for p in self.v) + " -- cycle"


def circulo(cx, cy, r, n=240, nombre=""):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Poligono(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]), nombre)


def png(nombre, campo, masc, cmap, vmin, vmax):
    """Mapa de color en PNG (origen abajo); los conductores, transparentes."""
    t = np.clip((campo - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(t)
    rgba[masc, 3] = 0.0
    plt.imsave(os.path.join(SALIDA, nombre), rgba, origin="lower")
    print("  ->", nombre)


def semillas_superficie(sol, V, E, cond, delta, offset=0.035, signo=+1):
    """Semillas sobre la superficie de un conductor, a incrementos iguales de
    flujo saliente (signo=+1) o entrante (signo=-1)."""
    P, N = cond.contorno(0.008)
    Q = P + offset * N
    En = np.array([E(x, y)[0] * nx + E(x, y)[1] * ny for (x, y), (nx, ny) in zip(Q, N)])
    peso = np.clip(signo * En, 0, None)
    return [tuple(Q[i]) for i in reparte(P, peso, delta)], P, N, En


# ===========================================================================
#  1. Esfera conductora neutra en un campo uniforme (analítico)
# ===========================================================================
def esfera_campo_uniforme():
    print("[1] Esfera conductora en un campo uniforme")
    a, L = 1.0, 2.6
    x = np.linspace(-L, L, 801)
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = -(X) * (1 - a ** 3 / R ** 3)                 # E0 = 1 según +x
        psi = 0.5 * Y ** 2 * (1 + 2 * a ** 3 / R ** 3)     # función de Stokes
    fuera = R < a
    yk = np.arange(0.18, L + 0.01, 0.30)
    lc = curvas_nivel(X, Y, psi, [0.5 * v * v for v in yk], fuera)
    lc += [[(px, -py) for px, py in c] for c in lc]
    lc.append([(-L, 0.0), (-a, 0.0)])
    lc.append([(a, 0.0), (L, 0.0)])
    eq = curvas_nivel(X, Y, phi, list(np.arange(-2.4, 2.41, 0.4)), fuera)

    def E(px, py):
        r = np.hypot(px, py)
        c, s = px / r, py / r
        er = (1 + 2 * a ** 3 / r ** 3) * c
        et = -(1 - a ** 3 / r ** 3) * s
        return er * c - et * s, er * s + et * c
    # se orienta cada línea de izquierda a derecha
    lc = [c if c[-1][0] >= c[0][0] else c[::-1] for c in lc]
    escribe("esfera_uniforme_lc.tex", "Esfera en campo uniforme - líneas de campo",
            ["\\LC{%s}" % coords(c) for c in lc])
    escribe("esfera_uniforme_eq.tex", "Esfera en campo uniforme - equipotenciales",
            ["\\EQ{%s}" % coords(c) for c in eq])
    fl = []
    for c in lc:
        P = np.array(c)
        for xm in (-1.9, 1.9):
            if P[:, 0].min() < xm < P[:, 0].max():
                i = int(np.argmin(np.abs(P[:, 0] - xm)))
                ex, ey = E(*P[i])
                fl.append((P[i][0], P[i][1], np.degrees(np.arctan2(ey, ex))))
    escribe("esfera_uniforme_fl.tex", "Esfera en campo uniforme - flechas",
            ["\\FL{%.3f}{%.3f}{%.1f}" % f for f in fl])
    # cargas inducidas: sigma = 3 eps0 E0 cos(theta)
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    C = np.column_stack([1.13 * np.cos(t), 1.13 * np.sin(t)])
    sig = 3 * np.cos(t)
    marcas = []
    for i in reparte(C, np.clip(sig, 0, None), 9):
        marcas.append("\\CP{%.3f}{%.3f}" % tuple(C[i]))
    for i in reparte(C, np.clip(-sig, 0, None), 9):
        marcas.append("\\CN{%.3f}{%.3f}" % tuple(C[i]))
    escribe("esfera_uniforme_q.tex", "Esfera en campo uniforme - cargas inducidas", marcas)


# ===========================================================================
#  2. Conductores 2D: neutro en campo uniforme y cargado con punta
# ===========================================================================
def lineas_numericas(sol, E, conductores, semillas, ventana, h=0.012, dsep=0.0,
                     direcciones=(+1,)):
    x0, x1, y0, y1 = ventana

    def para(x, y):
        if x < x0 or x > x1 or y < y0 or y > y1:
            return True
        return any(c.contiene(x, y) for c in conductores)
    rej = Rejilla(max(dsep, 0.05))
    lineas = []
    for (sx, sy, sg) in semillas:
        if dsep > 0 and not rej.libre(sx, sy, dsep):
            continue
        pts = traza(E, sx, sy, sg, para, h=h)
        if len(pts) < 4:
            continue
        if sg < 0:
            pts = pts[::-1]
        # recorte suave al marco
        pts = [(min(max(px, x0), x1), min(max(py, y0), y1)) for px, py in pts]
        rej.anade(pts[::3])
        lineas.append(pts)
    return lineas


def conductor_neutro_uniforme():
    print("[2] Conductor neutro irregular en un campo uniforme (2D)")
    forma = Poligono([(-0.95, 0.55), (0.0, 0.85), (0.95, 0.62), (1.05, 0.35),
                      (0.35, -0.15), (0.05, -0.80), (-0.25, -0.78), (-0.55, -0.15),
                      (-1.10, 0.25)], "forma")
    # se redondean los vértices con un suavizado de Chaikin
    v = forma.v
    for _ in range(3):
        q = 0.75 * v + 0.25 * np.roll(v, -1, axis=0)
        r = 0.25 * v + 0.75 * np.roll(v, -1, axis=0)
        v = np.empty((2 * len(q), 2)); v[0::2] = q; v[1::2] = r
    forma = Poligono(v, "forma")
    sol = Laplace2D(-4.5, 4.5, -4.5, 4.5, 0.02, [forma])
    D0 = np.where(sol.borde, -sol.X, 0.0)                # conductor a 0, borde: -x
    D1 = np.where(sol.cond, 1.0, 0.0)                    # conductor a 1, borde a 0
    V0, V1 = sol.resuelve(D0), sol.resuelve(D1)
    fc = -sol.carga(V0, 0) / sol.carga(V1, 0)            # potencial del conductor neutro
    V = V0 + fc * V1
    print("    potencial del conductor neutro: %.4f ; carga final %.2e"
          % (fc, sol.carga(V, 0)))
    E, _ = sol.campo(V)
    ven = (-2.3, 2.3, -1.9, 1.9)
    sem = [(-2.3, yy, +1) for yy in np.arange(-1.85, 1.86, 0.205)]
    sem += [(2.3, yy, -1) for yy in np.arange(-1.85, 1.86, 0.205)]
    lin = lineas_numericas(sol, E, [forma], sem, ven, dsep=0.09)
    escribe("irregular_uniforme_lc.tex", "Conductor neutro en campo uniforme - líneas",
            ["\\LC{%s}" % coords(c) for c in lin])
    fl = flechas(lin, E, fraccion=(0.18, 0.82), minimo=0.6)
    escribe("irregular_uniforme_fl.tex", "Conductor neutro en campo uniforme - flechas",
            ["\\FL{%.3f}{%.3f}{%.1f}" % f for f in fl])
    m = (np.abs(sol.X) <= 2.35) & (np.abs(sol.Y) <= 1.95)
    eq = curvas_nivel(sol.X, sol.Y, np.where(m, V, np.nan),
                      [fc + k * 0.3 for k in range(-8, 9) if k != 0], sol.cond)
    escribe("irregular_uniforme_eq.tex", "Conductor neutro en campo uniforme - equipotenciales",
            ["\\EQ{%s}" % coords(c) for c in eq])
    _, P, N, En = semillas_superficie(sol, V, E, forma, 1.0)
    Q = P + 0.09 * N
    marcas = ["\\CP{%.3f}{%.3f}" % tuple(Q[i]) for i in reparte(P, np.clip(En, 0, None), 9)]
    marcas += ["\\CN{%.3f}{%.3f}" % tuple(Q[i]) for i in reparte(P, np.clip(-En, 0, None), 9)]
    escribe("irregular_uniforme_q.tex", "Conductor neutro en campo uniforme - cargas", marcas)
    escribe("irregular_forma.tex", "Contorno del conductor irregular",
            ["\\def\\formairregular{%s}" % forma.tikz()])


def gota(rc=0.9, cx=-0.6, punta=1.75, n=300):
    """Conductor en forma de «pera»: cuerpo circular de radio rc centrado en
    (cx, 0) unido por dos segmentos tangentes a una punta en (punta, 0); los
    vértices se redondean con dos pasadas de Chaikin."""
    d = punta - cx
    t0 = np.arccos(rc / d)
    t = np.linspace(t0, 2 * np.pi - t0, n)
    arco = np.column_stack([cx + rc * np.cos(t), rc * np.sin(t)])
    sube = np.linspace(arco[-1], [punta, 0.0], 80)[1:]
    baja = np.linspace([punta, 0.0], arco[0], 80)[1:-1]
    v = np.vstack([arco, sube, baja])
    for _ in range(2):
        q = 0.75 * v + 0.25 * np.roll(v, -1, axis=0)
        r = 0.25 * v + 0.75 * np.roll(v, -1, axis=0)
        v = np.empty((2 * len(q), 2)); v[0::2] = q; v[1::2] = r
    return Poligono(v, "gota")


def efecto_punta():
    print("[3] Conductor cargado con punta (efecto punta, 2D)")
    g = gota()
    sol = Laplace2D(-6, 6, -6, 6, 0.015, [g])
    V = sol.resuelve(np.where(sol.cond, 1.0, 0.0))
    Q = sol.carga(V, 0)
    E, Emod = sol.campo(V)
    print("    carga por unidad de longitud: %.4f" % Q)
    ven = (-2.6, 2.6, -2.0, 2.0)
    _, P, N, En = semillas_superficie(sol, V, E, g, 1.0, offset=0.03)
    # semillas: incrementos iguales de flujo (Q/26 por línea)
    idx = reparte(P, np.clip(En, 0, None), 30)
    sem = [(P[i][0] + 0.03 * N[i][0], P[i][1] + 0.03 * N[i][1], +1) for i in idx]
    lin = lineas_numericas(sol, E, [g], sem, ven)
    escribe("punta_lc.tex", "Conductor con punta - líneas de campo",
            ["\\LC{%s}" % coords(c) for c in lin])
    escribe("punta_fl.tex", "Conductor con punta - flechas",
            ["\\FL{%.3f}{%.3f}{%.1f}" % f for f in flechas(lin, E, fraccion=(0.62,), minimo=0.3)])
    m = (np.abs(sol.X) <= 2.65) & (np.abs(sol.Y) <= 2.05)
    eq = curvas_nivel(sol.X, sol.Y, np.where(m, V, np.nan),
                      [0.9, 0.8, 0.7, 0.6, 0.5, 0.42], sol.cond)
    escribe("punta_eq.tex", "Conductor con punta - equipotenciales",
            ["\\EQ{%s}" % coords(c) for c in eq])
    # superficie coloreada por sigma (escala logarítmica normalizada)
    Pc, Nc = g.contorno(0.02)
    sig = np.array([max(E(*(p + 0.03 * nn))[0] * nn[0] + E(*(p + 0.03 * nn))[1] * nn[1], 1e-6)
                    for p, nn in zip(Pc, Nc)])
    ls = np.log(sig)
    t = (ls - ls.min()) / (ls.max() - ls.min())
    seg = []
    for k in range(len(Pc)):
        a, b = Pc[k], Pc[(k + 1) % len(Pc)]
        seg.append("\\SG{%.3f}{%.3f}{%.3f}{%.3f}{%.3f}" % (a[0], a[1], b[0], b[1], t[k]))
    escribe("punta_sigma.tex", "Conductor con punta - densidad superficial", seg)
    escribe("punta_forma.tex", "Contorno de la gota",
            ["\\def\\formagota{%s}" % g.tikz()])
    # sigma frente a la abscisa x a lo largo del contorno superior
    with open(os.path.join(SALIDA, "punta_sigma.dat"), "w") as f:
        f.write("x sigma\n")
        # media móvil de 9 puntos: elimina el ruido de discretización de la malla
        sig_s = np.convolve(np.r_[sig[-4:], sig, sig[:4]], np.ones(9) / 9, mode="valid")
        sup = [(p[0], s) for p, s in zip(Pc, sig_s) if p[1] >= -1e-9]
        sup.sort()
        smin = min(s for _, s in sup)
        for x, s in sup[::2]:
            f.write("%.4f %.5f\n" % (x, s / smin))
    print("  -> punta_sigma.dat")
    return lin


# ===========================================================================
#  3. Corteza esférica con carga en la cavidad (imágenes)
# ===========================================================================
def corteza():
    print("[4] Corteza esférica con carga en la cavidad (imágenes)")
    R1, R2, D = 1.0, 2.0, 4.0
    ven = (-5.4, 3.4, -3.3, 3.3)

    def campo_cargas(lista):
        def E(x, y):
            ex = ey = 0.0
            for q, cx, cy in lista:
                dx, dy = x - cx, y - cy
                r3 = (dx * dx + dy * dy) ** 1.5
                ex += q * dx / r3
                ey += q * dy / r3
            return ex, ey
        return E

    casos = {
        "a": dict(q=1.0, s=0.0, Qe=0.0),
        "b": dict(q=1.0, s=0.45, Qe=0.0),
        "c": dict(q=1.0, s=0.45, Qe=1.6),
        "d": dict(q=0.0, s=0.0, Qe=1.6),
    }
    for clave, c in casos.items():
        q, s, Qe = c["q"], c["s"], c["Qe"]
        dentro = [(q, s, 0.0)]
        if q != 0 and s > 0:
            dentro.append((-q * R1 / s, R1 * R1 / s, 0.0))
        fuera = [(q + Qe * R2 / D, 0.0, 0.0)]
        if Qe != 0:
            fuera += [(Qe, -D, 0.0), (-Qe * R2 / D, -R2 * R2 / D, 0.0)]
        Ein, Eout = campo_cargas(dentro), campo_cargas(fuera)
        lineas = []
        # (i) líneas en la cavidad: de q a la cara interior
        if q != 0:
            for k in range(20):
                a = 2 * np.pi * (k + 0.5) / 20
                p = traza(Ein, s + 0.06 * np.cos(a), 0.06 * np.sin(a), +1,
                          lambda x, y: x * x + y * y >= R1 * R1, h=0.006)
                lineas.append(p)
        # (ii) líneas exteriores: desde la cara exterior (sigma > 0) y desde Qe
        t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
        C = np.column_stack([R2 * np.cos(t), R2 * np.sin(t)])
        Cn = np.column_stack([np.cos(t), np.sin(t)])
        sig = np.array([Eout(*(R2 * 1.0005 * n))[0] * n[0] + Eout(*(R2 * 1.0005 * n))[1] * n[1]
                        for n in Cn])
        x0, x1, y0, y1 = ven

        def para_out(x, y):
            return (x * x + y * y <= R2 * R2 or x < x0 or x > x1 or y < y0 or y > y1
                    or (Qe != 0 and (x + D) ** 2 + y * y < 0.003))
        # número de líneas que nacen en la cara exterior (sigma > 0)
        n_ext = {"a": 20, "b": 20, "c": 16, "d": 7}[clave]
        for i in reparte(C, np.clip(sig, 0, None), n_ext):
            p = traza(Eout, C[i][0] * 1.002, C[i][1] * 1.002, +1, para_out, h=0.012)
            lineas.append(p)
        if Qe != 0:
            for k in range(26):
                a = 2 * np.pi * (k + 0.5) / 26
                p = traza(Eout, -D + 0.07 * np.cos(a), 0.07 * np.sin(a), +1, para_out, h=0.012)
                lineas.append(p)
        lineas = [[(min(max(x, x0), x1), min(max(y, y0), y1)) for x, y in l] for l in lineas]
        escribe("corteza_%s_lc.tex" % clave, "Corteza (%s) - líneas de campo" % clave,
                ["\\LC{%s}" % coords(l, 0.02) for l in lineas])
        Eg = lambda x, y: Ein(x, y) if x * x + y * y < R1 * R1 else Eout(x, y)
        escribe("corteza_%s_fl.tex" % clave, "Corteza (%s) - flechas" % clave,
                ["\\FL{%.3f}{%.3f}{%.1f}" % f for f in flechas(lineas, Eg, fraccion=(0.55,), minimo=0.4)])
        # cargas inducidas en las dos caras
        marcas = []
        if q != 0:
            Ci = np.column_stack([R1 * np.cos(t), R1 * np.sin(t)])
            sin_ = np.array([-(Ein(*(R1 * 0.999 * n))[0] * n[0] + Ein(*(R1 * 0.999 * n))[1] * n[1])
                             for n in Cn])
            for i in reparte(Ci, np.clip(-sin_, 0, None), 10):
                marcas.append("\\CN{%.3f}{%.3f}" % tuple(0.86 * Ci[i]))
        for i in reparte(C, np.clip(sig, 0, None), 12 if q != 0 else 7):
            marcas.append("\\CP{%.3f}{%.3f}" % tuple(1.09 * C[i]))
        for i in reparte(C, np.clip(-sig, 0, None), 7 if Qe != 0 else 0):
            marcas.append("\\CN{%.3f}{%.3f}" % tuple(1.09 * C[i]))
        escribe("corteza_%s_q.tex" % clave, "Corteza (%s) - cargas inducidas" % clave, marcas)


# ===========================================================================
#  4. Sistema de cuatro conductores: coeficientes c_ij y p_ij
# ===========================================================================
def sistema():
    print("[5] Sistema de cuatro conductores en una caja a tierra")
    cuna = Poligono([(-4.25, -2.95), (-3.05, -3.95), (0.22, -0.18)], "cuña")
    rect = Poligono([(-2.75, 1.25), (-0.65, 1.25), (-0.65, 2.85), (-2.75, 2.85)], "rectángulo")
    disco = circulo(0.85, 0.35, 0.48, nombre="disco")
    barra = Poligono([(1.05, 1.32), (3.55, 1.32), (3.55, 1.68), (1.05, 1.68)], "barra")
    conds = [cuna, rect, disco, barra]
    sol = Laplace2D(-5, 5, -4.5, 4.5, 0.02, conds)
    U = []
    for j in range(4):
        D = np.zeros_like(sol.X)
        D[sol.masc[j]] = 1.0
        U.append(sol.resuelve(D))
    Cm = np.array([[sol.carga(U[j], i) for j in range(4)] for i in range(4)])
    Pm = np.linalg.inv(Cm)
    print("    c_ij (eps0 = 1):\n", np.array2string(Cm, precision=3))
    print("    asimetría máx. |c_ij - c_ji| / |c_ij|: %.2e"
          % np.max(np.abs(Cm - Cm.T) / np.abs(Cm)))
    Cs = 0.5 * (Cm + Cm.T)
    Ps = np.linalg.inv(Cs)
    nombres = ["A", "B", "C", "D"]
    out = []
    for i in range(4):
        for j in range(4):
            out.append("\\expandafter\\def\\csname cij%s%s\\endcsname{%s}"
                       % (nombres[i], nombres[j], ("%.2f" % Cs[i, j]).replace(".", "{,}")))
            out.append("\\expandafter\\def\\csname pij%s%s\\endcsname{%s}"
                       % (nombres[i], nombres[j], ("%.4f" % Ps[i, j]).replace(".", "{,}")))
        out.append("\\expandafter\\def\\csname sumc%s\\endcsname{%s}"
                   % (nombres[i], ("%.2f" % Cs[:, i].sum()).replace(".", "{,}")))
    escribe("sistema_coef.tex", "Sistema de cuatro conductores - coeficientes", out)
    escribe("sistema_formas.tex", "Contornos de los cuatro conductores",
            ["\\def\\formaA{%s}" % cuna.tikz(), "\\def\\formaB{%s}" % rect.tikz(),
             "\\def\\formaC{%s}" % disco.tikz(), "\\def\\formaD{%s}" % barra.tikz()])
    # mapas de potencial y de campo de los cuatro estados elementales
    for j in range(4):
        png("sistema_pot_%d.png" % (j + 1), U[j], sol.cond, AZUL_POT, 0.0, 1.0)
        _, Em = sol.campo(U[j])
        lE = np.log10(np.maximum(Em, 1e-4))
        png("sistema_campo_%d.png" % (j + 1), lE, sol.cond, AZUL_CAMPO, -1.6, 0.9)
        eq = curvas_nivel(sol.X, sol.Y, U[j], [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                          sol.cond)
        escribe("sistema_eq_%d.tex" % (j + 1), "Estado elemental %d - equipotenciales" % (j + 1),
                ["\\EQ{%s}" % coords(c, 0.04) for c in eq])
    # estado «disco cargado, los demás neutros»: phi = P e_C
    phis = Ps[:, 2] / Ps[2, 2]                # se normaliza phi_C = 1
    V = sum(phis[k] * U[k] for k in range(4))
    E, _ = sol.campo(V)
    print("    potenciales (disco cargado, resto neutros):", np.round(phis, 3))
    sem = []
    for k, c in enumerate(conds):
        s, P, N, En = semillas_superficie(sol, V, E, c, 1.0)
        idx = reparte(P, np.clip(En, 0, None), 0.11)       # flujo fijo por línea
        sem += [(P[i][0] + 0.035 * N[i][0], P[i][1] + 0.035 * N[i][1], +1) for i in idx]
    lin = lineas_numericas(sol, E, conds, sem, (-4.95, 4.95, -4.45, 4.45), h=0.012)
    escribe("sistema_libre_lc.tex", "Disco cargado, resto neutros - líneas de campo",
            ["\\LC{%s}" % coords(c) for c in lin])
    escribe("sistema_libre_fl.tex", "Disco cargado, resto neutros - flechas",
            ["\\FL{%.3f}{%.3f}{%.1f}" % f for f in flechas(lin, E, fraccion=(0.5,), minimo=0.35)])
    eq = curvas_nivel(sol.X, sol.Y, V, [0.08, 0.16, 0.24, 0.32, 0.4, 0.5, 0.6, 0.7, 0.8], sol.cond)
    escribe("sistema_libre_eq.tex", "Disco cargado, resto neutros - equipotenciales",
            ["\\EQ{%s}" % coords(c, 0.04) for c in eq])
    marcas = []
    for k, c in enumerate(conds):
        _, P, N, En = semillas_superficie(sol, V, E, c, 1.0)
        for i in reparte(P, np.clip(En, 0, None), 0.33):
            marcas.append("\\CP{%.3f}{%.3f}" % tuple(P[i] + 0.16 * N[i]))
        for i in reparte(P, np.clip(-En, 0, None), 0.33):
            marcas.append("\\CN{%.3f}{%.3f}" % tuple(P[i] + 0.16 * N[i]))
    escribe("sistema_libre_q.tex", "Disco cargado, resto neutros - cargas", marcas)
    with open(os.path.join(SALIDA, "sistema_libre_phi.tex"), "w") as f:
        f.write("%% potenciales del estado libre (phi_C = 1)\n")
        for n, v in zip(nombres, phis):
            f.write("\\expandafter\\def\\csname philibre%s\\endcsname{%s}\n"
                    % (n, ("%.2f" % v).replace(".", "{,}")))


if __name__ == "__main__":
    print("Generando datos en", os.path.normpath(SALIDA))
    esfera_campo_uniforme()
    conductor_neutro_uniforme()
    efecto_punta()
    corteza()
    sistema()
