"""
VAJAX Logo Generator — Signal Topology Philosophy (v3 - Polished)

Creates:
1. Square icon (512x512 SVG) — filled "V" mark with circuit traces
2. Horizontal wordmark (SVG) — icon + VAJAX text
3. PDF brand sheet with both versions
"""

from pathlib import Path
import math

from reportlab.lib.colors import Color, HexColor
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import (
    Drawing, Rect, Line, Circle, String, PolyLine, Polygon,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
FONTS = Path("/Users/roberttaylor/.claude/plugins/cache/anthropic-agent-skills/"
             "document-skills/3d5951151859/skills/canvas-design/canvas-fonts")
OUT = HERE

# ---------------------------------------------------------------------------
# Register fonts
# ---------------------------------------------------------------------------
for name, filename in [
    ("JetBrainsMono", "JetBrainsMono-Regular.ttf"),
    ("JetBrainsMono-Bold", "JetBrainsMono-Bold.ttf"),
    ("Jura-Light", "Jura-Light.ttf"),
    ("Jura-Medium", "Jura-Medium.ttf"),
    ("GeistMono", "GeistMono-Regular.ttf"),
    ("GeistMono-Bold", "GeistMono-Bold.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))

# ---------------------------------------------------------------------------
# Colour palette — Signal Topology
# ---------------------------------------------------------------------------
DEEP_INDIGO = HexColor("#0D1B2A")
CIRCUIT_BLUE = HexColor("#1B4965")
SIGNAL_AMBER = HexColor("#E8A924")
AMBER_GLOW = HexColor("#F5C842")
BRIGHT_TEAL = HexColor("#41B3A3")
WHITE_CLEAN = HexColor("#F0F4F8")
GRID_LINE = HexColor("#162D47")
NODE_GREEN = HexColor("#5CDB95")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def draw_grid(g, w, h, spacing=16, color=GRID_LINE, sw=0.3, opacity=0.2):
    """Subtle substrate grid."""
    for i in range(0, int(w) + 1, spacing):
        g.add(Line(i, 0, i, h, strokeColor=color, strokeWidth=sw,
                   strokeOpacity=opacity))
    for i in range(0, int(h) + 1, spacing):
        g.add(Line(0, i, w, i, strokeColor=color, strokeWidth=sw,
                   strokeOpacity=opacity))


def draw_dot(g, x, y, r=3.0, color=NODE_GREEN, opacity=0.85):
    """Circuit node dot."""
    g.add(Circle(x, y, r, fillColor=color, strokeColor=None,
                 fillOpacity=opacity))


def _flat(pts):
    """Flatten list of (x,y) tuples to [x1,y1,x2,y2,...]."""
    out = []
    for p in pts:
        out.extend(p)
    return out


def draw_trace(g, pts, color=CIRCUIT_BLUE, sw=1.8, opacity=0.55):
    """Draw a single circuit trace polyline."""
    g.add(PolyLine(_flat(pts), strokeColor=color, strokeWidth=sw,
                   fillColor=Color(0, 0, 0, 0), strokeOpacity=opacity,
                   strokeLineCap=1, strokeLineJoin=1))


# ---------------------------------------------------------------------------
# V mark — filled polygon with glow
# ---------------------------------------------------------------------------
def _v_polygon_points(cx, cy, size):
    """
    Return polygon points for a filled V shape with parallel arm edges.

    6-point polygon. The inner edges are exactly parallel to the outer edges.
    This is achieved by computing notch_y from the geometry so that:
      inner_slope == outer_slope for each arm.

    Layout (reportlab: y increases upward):

        5 _________ 3       <- inner-left, inner-right (top_y)
         \\       //
          \\  4  //          <- inner notch (notch_y, computed)
           \\ | //
      0 ----\\|//---- 2     <- outer-left, outer-right (top_y)
              1              <- apex (apex_y)
    """
    # V dimensions as fractions of size
    spread = size * 0.24        # half-distance from center to arm center at top
    arm_half_w = size * 0.048   # half arm width (horizontal) at top
    top_y = cy + size * 0.16    # y of arm tops
    apex_y = cy - size * 0.22   # y of apex tip

    # For parallel edges: the inner edge has direction (half_top_outer, apex_y - top_y)
    # starting from half_top_inner. The two inner edges meet at x=cx when:
    #   t = half_top_inner / half_top_outer
    # So notch_y = top_y + t * (apex_y - top_y)
    half_top_outer = spread + arm_half_w
    half_top_inner = spread - arm_half_w
    t_notch = half_top_inner / half_top_outer
    notch_y = top_y + t_notch * (apex_y - top_y)

    # 6-point polygon (clockwise in visual space):
    #   outer-left -> apex -> outer-right -> inner-right -> notch -> inner-left
    pts = [
        (cx - half_top_outer, top_y),   # 0: outer left
        (cx,                  apex_y),  # 1: apex
        (cx + half_top_outer, top_y),   # 2: outer right
        (cx + half_top_inner, top_y),   # 3: inner right
        (cx,                  notch_y), # 4: inner notch
        (cx - half_top_inner, top_y),   # 5: inner left
    ]
    return pts, top_y, apex_y, spread, arm_half_w


def draw_v_filled(g, cx, cy, size):
    """Draw the V as a filled polygon with soft glow."""
    pts, top_y, apex_y, spread, arm_half_w = _v_polygon_points(cx, cy, size)

    # Glow layer (scaled outward for halo)
    glow_s = 1.12
    glow_pts = [(cx + (px - cx) * glow_s, cy + (py - cy) * glow_s)
                for (px, py) in pts]
    g.add(Polygon(_flat(glow_pts), fillColor=SIGNAL_AMBER,
                  strokeColor=None, fillOpacity=0.08))

    # Main fill
    g.add(Polygon(_flat(pts), fillColor=SIGNAL_AMBER,
                  strokeColor=None, fillOpacity=0.95))

    # Highlight along outer spine (pts 0 -> 1 -> 2 = outer-left -> apex -> outer-right)
    outer_spine = [pts[0], pts[1], pts[2]]
    g.add(PolyLine(_flat(outer_spine), strokeColor=AMBER_GLOW,
                   strokeWidth=size * 0.004, fillColor=Color(0, 0, 0, 0),
                   strokeLineCap=1, strokeLineJoin=1,
                   strokeOpacity=0.35))



# ---------------------------------------------------------------------------
# Circuit traces
# ---------------------------------------------------------------------------
def draw_circuit_traces(g, cx, cy, size, color=CIRCUIT_BLUE, sw=1.8,
                        clip_radius=None):
    """
    Draw circuit traces routing from the V's three endpoints
    (left arm tip, right arm tip, apex) outward toward the edges.

    If clip_radius is set, traces are clipped to stay within that radius
    of (cx, cy) — used for the wordmark circle.
    """
    pts_v, top_y, apex_y, spread, arm_half_w = _v_polygon_points(cx, cy, size)

    # V endpoint positions
    v_left = (cx - spread - arm_half_w, top_y)
    v_right = (cx + spread + arm_half_w, top_y)
    v_apex = (cx, apex_y)

    # Extent limits
    if clip_radius:
        far = clip_radius * 0.85
    else:
        far = size * 0.42

    # Traces emerge from the V endpoints and route outward like PCB traces.
    # All routing goes outward/downward from the V — nothing goes above the V.

    # --- Left arm: route left then down ---
    lx = cx - far
    trace_l1 = [
        v_left,
        (lx, top_y),                  # route left at arm level
        (lx, top_y - far * 0.50),     # then down
    ]
    # Second trace from left arm, routes left-down at a different level
    lx2 = cx - far * 0.75
    arm_mid_y = top_y - (top_y - apex_y) * 0.30  # partway down left arm
    arm_mid_x = cx - spread * 0.70                # corresponding x on arm
    trace_l2 = [
        (arm_mid_x, arm_mid_y),
        (lx2, arm_mid_y),             # route left
        (lx2, arm_mid_y - far * 0.35),  # then down
    ]

    # --- Right arm: route right then down ---
    rx = cx + far
    trace_r1 = [
        v_right,
        (rx, top_y),
        (rx, top_y - far * 0.50),
    ]
    rx2 = cx + far * 0.75
    trace_r2 = [
        (cx + spread * 0.70, arm_mid_y),
        (rx2, arm_mid_y),
        (rx2, arm_mid_y - far * 0.35),
    ]

    # --- Apex: route down then branch ---
    branch_y = apex_y - far * 0.22
    trace_a1 = [
        v_apex,
        (cx, branch_y),
    ]
    trace_a2 = [
        (cx, branch_y),
        (cx - far * 0.40, branch_y),
    ]
    trace_a3 = [
        (cx, branch_y),
        (cx + far * 0.40, branch_y),
        (cx + far * 0.40, branch_y - far * 0.18),
    ]

    all_traces = [trace_l1, trace_l2, trace_r1, trace_r2,
                  trace_a1, trace_a2, trace_a3]

    for t_pts in all_traces:
        draw_trace(g, t_pts, color=color, sw=sw, opacity=0.45)

    # Via dots at trace endpoints
    dot_r = size * 0.016
    endpoints = [
        (lx, top_y - far * 0.50),
        (lx2, arm_mid_y - far * 0.35),
        (rx, top_y - far * 0.50),
        (rx2, arm_mid_y - far * 0.35),
        (cx - far * 0.40, branch_y),
        (cx + far * 0.40, branch_y - far * 0.18),
        (cx, branch_y),
    ]
    for ex, ey in endpoints:
        draw_dot(g, ex, ey, r=dot_r, color=NODE_GREEN, opacity=0.80)


# ---------------------------------------------------------------------------
# Decorative elements
# ---------------------------------------------------------------------------
def draw_chevrons(g, x, y, count=4, spacing=6, sz=4,
                  color=BRIGHT_TEAL, opacity=0.4, sw=1.0):
    """Row of small chevrons — GPU parallel lanes."""
    for i in range(count):
        cx = x + i * spacing
        pts = [(cx - sz / 2, y + sz / 2),
               (cx, y - sz / 2),
               (cx + sz / 2, y + sz / 2)]
        g.add(PolyLine(_flat(pts), strokeColor=color, strokeWidth=sw,
                       fillColor=Color(0, 0, 0, 0), strokeOpacity=opacity))


def draw_waveform(g, x0, yc, w, amp, periods=3,
                  color=SIGNAL_AMBER, sw=1.2, opacity=0.7):
    """Sine waveform decoration."""
    pts = []
    steps = 100
    for i in range(steps + 1):
        t = i / steps
        x = x0 + t * w
        raw = math.sin(t * periods * 2 * math.pi)
        y = yc + max(-0.85, min(0.85, raw * 1.2)) * amp
        pts.extend([x, y])
    g.add(PolyLine(pts, strokeColor=color, strokeWidth=sw,
                   fillColor=Color(0, 0, 0, 0), strokeOpacity=opacity))


# =========================================================================
# ICON (square)
# =========================================================================
def create_icon(size=512):
    """Square icon — the V mark with circuit trace routing."""
    d = Drawing(size, size)
    cx, cy = size / 2, size / 2

    # Background
    d.add(Rect(0, 0, size, size, fillColor=DEEP_INDIGO,
               strokeColor=None, rx=size * 0.08, ry=size * 0.08))

    # Substrate grid
    draw_grid(d, size, size, spacing=int(size / 20), color=GRID_LINE,
              sw=0.7, opacity=0.18)

    # Shift V slightly up from center for better visual balance
    vy = cy + size * 0.02

    # Circuit traces (behind V) — thick wires
    draw_circuit_traces(d, cx, vy, size, color=CIRCUIT_BLUE,
                        sw=size * 0.012)

    # Filled V mark
    draw_v_filled(d, cx, vy, size)

    # Corner fiducials
    mk = size * 0.04
    mo = size * 0.055
    for mx, my in [(mo, size - mo), (size - mo, size - mo),
                   (mo, mo), (size - mo, mo)]:
        d.add(Line(mx - mk / 2, my, mx + mk / 2, my,
                   strokeColor=WHITE_CLEAN, strokeWidth=size * 0.003,
                   strokeOpacity=0.28))
        d.add(Line(mx, my - mk / 2, mx, my + mk / 2,
                   strokeColor=WHITE_CLEAN, strokeWidth=size * 0.003,
                   strokeOpacity=0.28))

    return d


# =========================================================================
# WORDMARK (horizontal)
# =========================================================================
def create_wordmark(width=1200, height=400):
    """Horizontal wordmark — icon circle + VAJAX text."""
    d = Drawing(width, height)
    margin = height * 0.08
    icon_size = height * 0.82
    icon_x = margin
    icon_y = (height - icon_size) / 2

    # Background
    d.add(Rect(0, 0, width, height, fillColor=DEEP_INDIGO,
               strokeColor=None, rx=10, ry=10))

    # Subtle grid
    draw_grid(d, width, height, spacing=20, color=GRID_LINE, sw=0.5,
              opacity=0.15)

    # --- Circular icon ---
    icon_cx = icon_x + icon_size / 2
    icon_cy = icon_y + icon_size / 2
    circle_r = icon_size * 0.44

    # Circle bg
    d.add(Circle(icon_cx, icon_cy, circle_r,
                 fillColor=HexColor("#0F2236"), strokeColor=CIRCUIT_BLUE,
                 strokeWidth=2.0, strokeOpacity=0.35))

    # Icon internals — scale to fit within circle
    inner_size = icon_size * 0.72
    vy = icon_cy + inner_size * 0.02
    draw_circuit_traces(d, icon_cx, vy, inner_size,
                        color=CIRCUIT_BLUE, sw=2.8,
                        clip_radius=circle_r)
    draw_v_filled(d, icon_cx, vy, inner_size)

    # --- Text area ---
    text_x = icon_x + icon_size + margin * 1.5

    # Subtitle: "Verilog-A x JAX" — light, above main
    sub_y = height * 0.72
    fs_sub = height * 0.075
    d.add(String(text_x, sub_y, "Verilog-A",
                 fontName="Jura-Light", fontSize=fs_sub,
                 fillColor=BRIGHT_TEAL, fillOpacity=0.70))

    # Use font metrics for positioning with explicit gaps
    from reportlab.pdfbase.pdfmetrics import stringWidth
    va_w = stringWidth("Verilog-A", "Jura-Light", fs_sub)
    gap = fs_sub * 0.5  # explicit gap on each side of ×
    cross_w = stringWidth("\u00d7", "Jura-Light", fs_sub)
    cross_x = text_x + va_w + gap + cross_w / 2
    d.add(String(cross_x - cross_w / 2, sub_y, "\u00d7",
                 fontName="Jura-Light", fontSize=fs_sub,
                 fillColor=Color(1, 1, 1, 0.25)))
    jax_x = cross_x + cross_w / 2 + gap
    d.add(String(jax_x, sub_y, "JAX",
                 fontName="Jura-Light", fontSize=fs_sub,
                 fillColor=SIGNAL_AMBER, fillOpacity=0.70))

    # Main: "VAJAX"
    main_y = height * 0.32
    d.add(String(text_x, main_y, "VAJAX",
                 fontName="GeistMono-Bold", fontSize=height * 0.34,
                 fillColor=WHITE_CLEAN))

    # Descriptor — below, separated
    desc_y = height * 0.16
    d.add(String(text_x, desc_y,
                 "GPU-ACCELERATED CIRCUIT SIMULATION",
                 fontName="JetBrainsMono", fontSize=height * 0.042,
                 fillColor=WHITE_CLEAN, fillOpacity=0.22))

    # Subtle waveform at very bottom
    draw_waveform(d, text_x, height * 0.055, width - text_x - margin * 2,
                  height * 0.012, periods=8, color=SIGNAL_AMBER, sw=1.5,
                  opacity=0.25)

    return d


# =========================================================================
# PDF brand sheet
# =========================================================================
def create_pdf():
    """PDF with both logo versions and colour palette."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas as PDFCanvas

    pdf_path = OUT / "vajax_logo.pdf"
    c = PDFCanvas(str(pdf_path), pagesize=A4)
    pw, ph = A4

    # Background
    c.setFillColor(DEEP_INDIGO)
    c.rect(0, 0, pw, ph, fill=1, stroke=0)

    # Title
    c.setFont("GeistMono", 9)
    c.setFillColor(WHITE_CLEAN, 0.3)
    c.drawString(30, ph - 35, "VAJAX \u2014 BRAND IDENTITY")

    # Icon
    icon_d = create_icon(280)
    ix = (pw - 280) / 2
    iy = ph - 350
    renderPDF.draw(icon_d, c, ix, iy)
    c.setFont("JetBrainsMono", 7)
    c.setFillColor(WHITE_CLEAN, 0.25)
    c.drawString(ix, iy - 15, "ICON  512 \u00d7 512")

    # Wordmark
    ww = pw - 60
    wh = ww / 3
    wm_d = create_wordmark(ww, wh)
    wy = iy - wh - 45
    renderPDF.draw(wm_d, c, 30, wy)
    c.setFont("JetBrainsMono", 7)
    c.setFillColor(WHITE_CLEAN, 0.25)
    c.drawString(30, wy - 15, "WORDMARK  1200 \u00d7 400")

    # Swatches
    sy = wy - 75
    swatches = [
        (DEEP_INDIGO, "Substrate", "#0D1B2A"),
        (CIRCUIT_BLUE, "Trace", "#1B4965"),
        (SIGNAL_AMBER, "Signal", "#E8A924"),
        (BRIGHT_TEAL, "Compute", "#41B3A3"),
        (WHITE_CLEAN, "Clean", "#F0F4F8"),
        (NODE_GREEN, "Node", "#5CDB95"),
    ]
    sp = (pw - 60) / len(swatches)
    for i, (col, name, hx) in enumerate(swatches):
        sx = 30 + i * sp
        c.setFillColor(col)
        c.roundRect(sx, sy, 28, 28, 4, fill=1, stroke=0)
        c.setFont("JetBrainsMono", 6)
        c.setFillColor(WHITE_CLEAN, 0.5)
        c.drawString(sx, sy - 12, name)
        c.setFillColor(WHITE_CLEAN, 0.25)
        c.drawString(sx, sy - 22, hx)

    # Footer
    c.setFont("Jura-Light", 7)
    c.setFillColor(WHITE_CLEAN, 0.15)
    c.drawString(30, 30, "Design Philosophy: Signal Topology")
    c.drawRightString(pw - 30, 30, "VAJAX \u00b7 ChipFlow")

    c.save()
    print(f"PDF saved: {pdf_path}")


# =========================================================================
# SVG export
# =========================================================================
def _fix_svg(path):
    """Fix reportlab SVG quirk: 'fill: None' (Python) -> 'fill: none' (SVG)."""
    text = path.read_text()
    text = text.replace("fill: None", "fill: none")
    path.write_text(text)


def create_svgs():
    """Export icon and wordmark as SVGs."""
    from reportlab.graphics import renderSVG

    icon_d = create_icon(512)
    icon_svg = OUT / "vajax_icon_512.svg"
    renderSVG.drawToFile(icon_d, str(icon_svg))
    _fix_svg(icon_svg)
    print(f"Icon SVG saved: {icon_svg}")

    wm_d = create_wordmark(1200, 400)
    wm_svg = OUT / "vajax_wordmark.svg"
    renderSVG.drawToFile(wm_d, str(wm_svg))
    _fix_svg(wm_svg)
    print(f"Wordmark SVG saved: {wm_svg}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    create_svgs()
    create_pdf()
    print("\nDone.")
