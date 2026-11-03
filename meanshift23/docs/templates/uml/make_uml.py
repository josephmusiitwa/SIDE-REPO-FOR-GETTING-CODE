#!/usr/bin/env python3
"""Generates the UML diagrams for docs/templates as SVG files.

Run from anywhere:  python3 docs/templates/uml/make_uml.py
Then convert to PNG with any SVG viewer / browser "save as image", or with
the render command given in docs/templates/TEMPLATES.md.

Edit the boxes below if you change a class; each box lists its UML
compartments (name, attributes, operations) and an optional template
parameter box (the dashed rectangle in the top-right corner, standard UML
notation for a template class).
"""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent
CH = 7.4        # approx. width of one monospace character at FONT px
FONT = 12.5
LINE = 17       # line height
PAD = 8

COLORS = {
    "template": ("#e8f1fb", "#2f6db5"),
    "concept":  ("#f1eafb", "#6b3fb0"),
    "alias":    ("#f2f2f2", "#666666"),
    "kernel":   ("#e9f6ec", "#2e7d43"),
    "function": ("#fdf6e3", "#a07400"),
    "data":     ("#fdecea", "#b23b2e"),
}


class Box:
    def __init__(self, x, y, w, kind, stereo, name, attrs=(), ops=(), tparams=()):
        self.x, self.y, self.w = x, y, w
        self.kind, self.stereo, self.name = kind, stereo, name
        self.attrs, self.ops, self.tparams = list(attrs), list(ops), list(tparams)
        head = (1 if stereo else 0) + 1
        self.h_head = head * LINE + PAD * 2
        self.h_attrs = (len(self.attrs) * LINE + PAD * 2) if self.attrs else 0
        self.h_ops = (len(self.ops) * LINE + PAD * 2) if self.ops else 0
        self.h = self.h_head + self.h_attrs + self.h_ops

    def tbox(self):
        tw = max(len(t) for t in self.tparams) * CH + 16
        th = len(self.tparams) * LINE + 10
        return (self.x + self.w - tw + 18, self.y - th + 10, tw, th)

    def tparam_top(self, f=0.5):
        tx, ty, tw, _ = self.tbox()
        return (tx + tw * f, ty)

    # anchor points
    def top(self, f=0.5):    return (self.x + self.w * f, self.y)
    def bottom(self, f=0.5): return (self.x + self.w * f, self.y + self.h)
    def left(self, f=0.5):   return (self.x, self.y + self.h * f)
    def right(self, f=0.5):  return (self.x + self.w, self.y + self.h * f)

    def svg(self):
        fill, stroke = COLORS[self.kind]
        o = []
        o.append(f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" rx="3"/>')
        cy = self.y + PAD + LINE - 4
        cx = self.x + self.w / 2
        if self.stereo:
            o.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" class="stereo">{escape(self.stereo)}</text>')
            cy += LINE
        o.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" class="name">{escape(self.name)}</text>')
        yy = self.y + self.h_head
        for section in (self.attrs, self.ops):
            if not section:
                continue
            o.append(f'<line x1="{self.x}" y1="{yy}" x2="{self.x + self.w}" y2="{yy}" stroke="{stroke}"/>')
            ty = yy + PAD + LINE - 4
            for line in section:
                cls = "member"
                if line.startswith("{static}"):
                    line, cls = line[len("{static}"):], "member static"
                o.append(f'<text x="{self.x + 10}" y="{ty}" class="{cls}">{escape(line)}</text>')
                ty += LINE
            yy += (len(section) * LINE + PAD * 2)
        if self.tparams:
            tw = max(len(t) for t in self.tparams) * CH + 16
            th = len(self.tparams) * LINE + 10
            tx = self.x + self.w - tw + 18
            ty = self.y - th + 10
            o.append(f'<rect x="{tx}" y="{ty}" width="{tw}" height="{th}" fill="#ffffff" '
                     f'stroke="{stroke}" stroke-dasharray="5 3" stroke-width="1.3"/>')
            for i, t in enumerate(self.tparams):
                o.append(f'<text x="{tx + 8}" y="{ty + 5 + LINE * (i + 1) - 4}" class="tparam">{escape(t)}</text>')
        return "\n".join(o)


def arrow(points, kind="assoc", label=None, label_at=None, label_anchor="start"):
    """kind: assoc (open arrow), dep (dashed open arrow), real (dashed hollow
    triangle), compose (filled diamond at start, open arrow at end), bind."""
    pts = " ".join(f"{x},{y}" for x, y in points)
    dashed = kind in ("dep", "real", "bind")
    marker_end = {"assoc": "url(#open)", "dep": "url(#open)", "bind": "url(#open)",
                  "real": "url(#hollow)", "compose": "url(#open)"}[kind]
    marker_start = ' marker-start="url(#diamond)"' if kind == "compose" else ""
    color = {"bind": "#666", "real": "#2e7d43", "dep": "#6b3fb0"}.get(kind, "#333")
    s = (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"'
         f'{" stroke-dasharray=\"6 4\"" if dashed else ""} marker-end="{marker_end}"{marker_start}/>')
    if label:
        lx, ly = label_at if label_at else points[len(points) // 2]
        for i, part in enumerate(label.split("\n")):
            s += (f'\n<text x="{lx}" y="{ly + i * 15}" text-anchor="{label_anchor}" '
                  f'class="label" fill="{color}">{escape(part)}</text>')
    return s


def svg_doc(w, h, title, body, legend=True):
    defs = """
<defs>
  <marker id="open" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="11" markerHeight="11" orient="auto-start-reverse">
    <path d="M1,1 L11,6 L1,11" fill="none" stroke="context-stroke" stroke-width="1.5"/></marker>
  <marker id="hollow" viewBox="0 0 14 14" refX="13" refY="7" markerWidth="14" markerHeight="14" orient="auto-start-reverse">
    <path d="M1,1 L13,7 L1,13 Z" fill="#ffffff" stroke="context-stroke" stroke-width="1.4"/></marker>
  <marker id="diamond" viewBox="0 0 16 10" refX="1" refY="5" markerWidth="16" markerHeight="10" orient="auto-start-reverse">
    <path d="M1,5 L8,1 L15,5 L8,9 Z" fill="#333"/></marker>
</defs>
<style>
  text { white-space: pre; font-family: 'DejaVu Sans Mono', Consolas, Menlo, monospace; font-size: 12.5px; fill: #1d1d1d; }
  .name { font-weight: bold; font-size: 14px; }
  .stereo { font-style: italic; fill: #555; }
  .tparam { font-size: 12px; fill: #1d1d1d; }
  .static { text-decoration: underline; }
  .label { font-size: 11.5px; font-style: italic; }
  .title { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 20px; font-weight: bold; }
  .sub { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 13px; fill: #555; }
  .legend { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 12px; }
</style>"""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           defs, f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
           f'<text x="30" y="36" class="title">{escape(title[0])}</text>',
           f'<text x="30" y="58" class="sub">{escape(title[1])}</text>', body]
    if legend:
        lx, ly = w - 330, 26
        items = [("template", "class template"), ("concept", "concept (C++20)"),
                 ("kernel", "kernel policy (plain struct)"), ("alias", "alias / «bind» instance"),
                 ("function", "free function template")]
        out.append(f'<rect x="{lx - 10}" y="{ly - 8}" width="320" height="{len(items) * 18 + 86}" fill="#fff" stroke="#ccc"/>')
        for i, (k, t) in enumerate(items):
            f, s = COLORS[k]
            out.append(f'<rect x="{lx}" y="{ly + i * 18}" width="22" height="12" fill="{f}" stroke="{s}"/>')
            out.append(f'<text x="{lx + 30}" y="{ly + i * 18 + 11}" class="legend">{t}</text>')
        y0 = ly + len(items) * 18 + 4
        rows = [("compose", "composition (owns)"), ("real", "realizes concept"),
                ("dep", "constrained by / uses"), ("bind", "«bind» template instance")]
        for i, (k, t) in enumerate(rows):
            yy = y0 + i * 18 + 6
            xx = lx
            out.append(arrow([(xx, yy), (xx + 30, yy)], k))
            out.append(f'<text x="{xx + 38}" y="{yy + 4}" class="legend">{t}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Diagram 1: the class diagram
# ---------------------------------------------------------------------------
def class_diagram():
    numeric = Box(30, 220, 330, "concept", "«concept»", "Numeric<T>",
                  ops=["is_arithmetic_v<T>", "&& !same_as<T, bool>", "&& !CharacterType<T>"])
    real = Box(400, 220, 250, "concept", "«concept»", "RealNumber<T>",
               ops=["std::floating_point<T>"])
    dflt = Box(690, 220, 330, "function", "«alias template»", "default_real_t<T : Numeric>",
               ops=["floating T  ->  T", "integral T  ->  double"])
    kfun = Box(1150, 220, 380, "concept", "«concept»", "KernelFunction<K, R>",
               ops=["RealNumber<R>", "&& default_initializable<K>",
                    "&& k(d, h) -> convertible_to<R>"])

    dataset = Box(30, 500, 470, "template", "", "BasicDataset",
                  attrs=["- data_ : BasicData<T>"],
                  ops=["+ BasicDataset()",
                       "+ BasicDataset(data : BasicData<T>)",
                       "+ loadFromCSV(file : string,",
                       "              hasHeader : bool = true) : bool",
                       "+ getData() : const BasicData<T>&",
                       "+ standardize()  requires RealNumber<T>",
                       "+ normalize()    requires RealNumber<T>",
                       "+ cast<U : Numeric>() : BasicDataset<U>"],
                  tparams=["T : Numeric"])
    ms = Box(560, 500, 560, "template", "", "BasicMeanShift",
             attrs=["- bandwidth_ : R", "- epsilon_ : R", "- maxIterations_ : int",
                    "- kernel_ : K", "- clusters_ : vector<BasicCluster<R>>",
                    "- labels_ : vector<int>"],
             ops=["+ BasicMeanShift(bandwidth : R = -1,",
                  "                 epsilon : R = 1e-3, maxIter : int = 300)",
                  "{static}+ euclideanDistance<A, B>(a, b) : R",
                  "+ gaussianKernel(d : R) : R",
                  "+ kernelWeight(d : R) : R",
                  "+ getBandwidth() : R",
                  "+ estimateBandwidth(data : BasicData<T>)",
                  "+ fit(data : BasicData<T>)",
                  "+ shiftPoint<U : Numeric>(p : BasicPoint<U>,",
                  "                data : BasicData<T>) : BasicPoint<R>",
                  "+ getClusters() : const vector<BasicCluster<R>>&",
                  "+ getLabels() : const vector<int>&",
                  "+ evaluate(data : BasicData<T>) : R"],
             tparams=["T : Numeric", "R : RealNumber = default_real_t<T>",
                      "K : KernelFunction<R> = GaussianKernel"])

    gk = Box(1200, 550, 330, "kernel", "«struct»", "GaussianKernel",
             ops=["+ operator()<R>(d : R, h : R) : R", "  exp(-d² / 2h²)"])
    fk = Box(1200, gk.y + gk.h + 22, 330, "kernel", "«struct»", "FlatKernel",
             ops=["+ operator()<R>(d : R, h : R) : R", "  d <= h ? 1 : 0"])
    ek = Box(1200, fk.y + fk.h + 22, 330, "kernel", "«struct»", "EpanechnikovKernel",
             ops=["+ operator()<R>(d : R, h : R) : R", "  1 - (d/h)²  if d <= h"])

    cluster_y = ms.y + ms.h + 90
    cluster = Box(640, cluster_y, 360, "template", "«struct»", "BasicCluster",
                  attrs=["+ centroid : BasicPoint<R>", "+ pointIndices : vector<size_t>"],
                  tparams=["R : RealNumber"])
    pdata = Box(30, dataset.y + dataset.h + 90, 470, "data", "«alias templates»", "BasicPoint<T> / BasicData<T>",
                ops=["BasicPoint<T : Numeric> = vector<T>",
                     "BasicData<T : Numeric>  = vector<BasicPoint<T>>",
                     "parseNumber<T>(text, out) : bool   (from_chars)"])
    ed = Box(1150, cluster_y, 380, "function", "«function template»", "euclideanDistance",
             ops=["<R : RealNumber, A : Numeric, B : Numeric>",
                  "(a : BasicPoint<A>, b : BasicPoint<B>) : R",
                  "casts to R before subtracting"])

    alias_y = max(cluster.y + cluster.h, pdata.y + pdata.h, ed.y + ed.h) + 70
    a_ds = Box(30, alias_y, 300, "alias", "«bind» T → double", "Dataset",
               ops=["= BasicDataset<double>"])
    a_pt = Box(350, alias_y, 300, "alias", "«bind» T → double", "Point / Data",
               ops=["= BasicPoint<double>", "= BasicData<double>"])
    a_ms = Box(670, alias_y, 350, "alias", "«bind» T, R → double; K → Gaussian", "MeanShift",
               ops=["= BasicMeanShift<double>"])
    a_cl = Box(1040, alias_y, 300, "alias", "«bind» R → double", "Cluster",
               ops=["= BasicCluster<double>"])

    boxes = [numeric, real, dflt, kfun, dataset, ms, gk, fk, ek, cluster, pdata, ed,
             a_ds, a_pt, a_ms, a_cl]
    H = alias_y + max(a_ms.h, a_pt.h) + 40
    arrows = []
    # concept constraints (dependency, dashed purple) -> the template-parameter box
    tp = ms.tbox()
    ptop = tp[1]
    arrows.append(arrow([numeric.bottom(0.3), (numeric.bottom(0.3)[0], dataset.tparam_top(0.5)[1] - 40),
                         (dataset.tparam_top(0.5)[0], dataset.tparam_top(0.5)[1] - 40), dataset.tparam_top(0.5)], "dep",
                        "«constrains T»", (numeric.bottom(0.3)[0] + 8, numeric.y + numeric.h + 30)))
    arrows.append(arrow([numeric.bottom(0.8), (numeric.bottom(0.8)[0], ptop - 20), (ms.tparam_top(0.08)[0], ptop - 20),
                         ms.tparam_top(0.08)], "dep", "«constrains T»", (numeric.bottom(0.8)[0] + 8, ptop - 26)))
    arrows.append(arrow([real.bottom(0.5), (real.bottom(0.5)[0], ptop - 45), (ms.tparam_top(0.25)[0], ptop - 45),
                         ms.tparam_top(0.25)], "dep", "«constrains R»", (real.bottom(0.5)[0] + 8, ptop - 51)))
    arrows.append(arrow([dflt.bottom(0.75), (dflt.bottom(0.75)[0], ptop - 70), (ms.tparam_top(0.55)[0], ptop - 70),
                         ms.tparam_top(0.55)], "dep", "«default for R»", (dflt.bottom(0.75)[0] + 8, dflt.y + dflt.h + 22)))
    arrows.append(arrow([kfun.bottom(0.15), (kfun.bottom(0.15)[0], ptop - 20),
                         (ms.tparam_top(0.85)[0], ptop - 20), ms.tparam_top(0.85)], "dep",
                        "«constrains K»", (kfun.bottom(0.15)[0] + 8, kfun.y + kfun.h + 22)))
    # kernels realize the concept
    for k in (gk, fk, ek):
        arrows.append(arrow([k.right(0.5), (1555, k.right(0.5)[1]), (1555, kfun.bottom(0.9)[1] + 40),
                             (kfun.bottom(0.9)[0], kfun.bottom(0.9)[1] + 40), kfun.bottom(0.9)], "real"))
    arrows.append(f'<text x="{1385}" y="{kfun.y + kfun.h + 30}" class="label" fill="#2e7d43">«realize»</text>')
    # MeanShift owns kernel_ and clusters_
    arrows.append(arrow([ms.right(0.15), gk.left(0.4)], "compose", "kernel_  1", (ms.x + ms.w + 6, ms.right(0.15)[1] - 8)))
    arrows.append(arrow([ms.bottom(0.5), cluster.top(0.5)], "compose", "clusters_\n0..*",
                        (ms.bottom(0.5)[0] + 10, ms.y + ms.h + 30)))
    # MeanShift uses BasicData<T> and euclideanDistance
    arrows.append(arrow([ms.bottom(0.08), (ms.bottom(0.08)[0], pdata.top(0.8)[1] - 30), (pdata.top(0.8)[0], pdata.top(0.8)[1] - 30), pdata.top(0.8)],
                        "dep", "«uses» data : BasicData<T>", (ms.x - 10, pdata.y - 38), "end"))
    arrows.append(arrow([dataset.bottom(0.3), pdata.top(0.3)], "compose", "data_", (dataset.bottom(0.3)[0] + 10, dataset.y + dataset.h + 40)))
    arrows.append(arrow([ms.bottom(0.92), (ms.bottom(0.92)[0], ed.left(0.3)[1]), ed.left(0.3)], "dep", "«calls»",
                        (ms.bottom(0.92)[0] + 8, ms.y + ms.h + 30)))
    # bind relations
    arrows.append(arrow([a_ds.top(0.4), (a_ds.top(0.4)[0], alias_y - 30), (dataset.x - 12, alias_y - 30), (dataset.x - 12, dataset.y + 40), (dataset.x, dataset.y + 40)], "bind"))
    arrows.append(arrow([a_pt.top(0.2), (a_pt.top(0.2)[0], pdata.y + pdata.h)], "bind"))
    arrows.append(arrow([a_ms.top(0.15), (a_ms.top(0.15)[0], alias_y - 40), (540, alias_y - 40), (540, ms.y + 60), (ms.x, ms.y + 60)], "bind"))
    arrows.append(arrow([a_cl.top(0.3), (a_cl.top(0.3)[0], cluster.right(0.5)[1]), cluster.right(0.5)], "bind"))

    body = "\n".join(arrows) + "\n" + "\n".join(b.svg() for b in boxes)
    title = ("MeanShift library: class templates and concepts",
             "Group 21 · namespace meanshift · dashed box at the top-right = template parameters, each constrained by a concept")
    return svg_doc(1600, H, title, body)


# ---------------------------------------------------------------------------
# Diagram 2: how one type flows through the templates (T = int example)
# ---------------------------------------------------------------------------
def type_flow():
    y = 120
    b1 = Box(30, y, 210, "data", "input", "CSV text", ops=["Name,Math,Science", "John,85,92", "Sarah,88,94"])
    b2 = Box(330, y, 290, "function", "«function template»", "parseNumber<int>",
             ops=["from_chars(text, int&)", "\"85\"   -> 85", "\"John\" -> skipped", "\"2.5\"  -> rejected",
                  "\"9e9\"  -> rejected (range)"])
    b3 = Box(720, y, 300, "template", "«bind» T → int", "BasicDataset<int>",
             ops=["getData() : BasicData<int>", "standardize()  ✗ not available", "cast<double>() ✓"])
    b4 = Box(1120, y, 460, "template", "«bind» T → int, R → double, K → Gaussian", "BasicMeanShift<int>",
             ops=["fit(BasicData<int>)", "  shiftedPoints : vector<BasicPoint<double>>",
                  "  distance: static_cast<double>(a[i]) - ...",
                  "evaluate(...) : double"])
    b5 = Box(1120, b4.y + b4.h + 80, 460, "template", "«bind» R → double", "BasicCluster<double>",
             ops=["centroid = {86.5, 93.0}   (real-valued mean)", "pointIndices = {0, 1}"])
    b6 = Box(720, b3.y + b3.h + 80, 300, "template", "«bind» T → double", "BasicDataset<double>",
             ops=["standardize() ✓", "normalize()   ✓"])
    b7 = Box(30, b6.y + b6.h + 90, 990, "concept", "compile-time decisions (no run-time cost)", "What the compiler checks for T = int",
             ops=["Numeric<int>               ✓  -> BasicDataset<int>, BasicMeanShift<int> are allowed",
                  "RealNumber<int>            ✗  -> standardize()/normalize() removed from BasicDataset<int>",
                  "default_real_t<int>        =  double  -> centroids, distances, evaluate() use double",
                  "KernelFunction<Gaussian, double>  ✓",
                  "Numeric<std::string>       ✗  -> BasicMeanShift<std::string> is a compile error naming the concept"])
    boxes = [b1, b2, b3, b4, b5, b6, b7]
    def mid(a, b): return ((a.x + a.w + b.x) / 2, a.right(0.5)[1] - 10)
    arrows = [
        arrow([b1.right(0.5), b2.left(0.5)], "assoc", "each cell", mid(b1, b2), "middle"),
        arrow([b2.right(0.5), b3.left(0.5)], "assoc", "loadFromCSV", mid(b2, b3), "middle"),
        arrow([b3.right(0.5), b4.left(0.5)], "assoc", "getData()", mid(b3, b4), "middle"),
        arrow([b4.bottom(0.5), b5.top(0.5)], "compose", "getClusters()", (b4.bottom(0.5)[0] + 10, b4.y + b4.h + 44)),
        arrow([b3.bottom(0.5), b6.top(0.5)], "bind", "cast<double>()", (b3.bottom(0.5)[0] + 10, b3.y + b3.h + 44)),
        arrow([b6.right(0.5), (1070, b6.right(0.5)[1]), (1070, b4.y + b4.h - 12), (b4.x, b4.y + b4.h - 12)], "assoc",
              "or: BasicMeanShift<double>\non the scaled data", (b6.x + b6.w + 8, b6.y + b6.h + 22)),
    ]
    H = b7.y + b7.h + 40
    body = "\n".join(b.svg() for b in boxes) + "\n" + "\n".join(arrows)
    title = ("Type flow: int data through the templates",
             "Which instantiation is produced at each step, and what the concepts allow or forbid")
    return svg_doc(1600, H, title, body, legend=False)


if __name__ == "__main__":
    (OUT / "class-diagram.svg").write_text(class_diagram(), encoding="utf-8")
    (OUT / "type-flow.svg").write_text(type_flow(), encoding="utf-8")
    print("wrote", OUT / "class-diagram.svg", "and", OUT / "type-flow.svg")
