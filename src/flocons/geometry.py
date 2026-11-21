"""Snowflake geometry: ink segmentation, dihedral symmetry, polygon fitting, SVG.

Coordinates are (x, y) in pixels, y pointing down (image and SVG convention).
"""

import io
from importlib.resources import files

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize, minimize_scalar
from skimage.draw import polygon as draw_polygon
from skimage.filters import threshold_multiotsu
from skimage.measure import approximate_polygon, find_contours

# ---------------------------------------------------------------- image and ink


def load_preview() -> np.ndarray:
    """The scanned snowflake as an RGB float array in [0, 1] (shipped in the package)."""
    data = files("flocons").joinpath("data/preview.webp").read_bytes()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=float) / 255


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def segment(rgb: np.ndarray) -> dict:
    """Split the image into ink, yellow and paper; keep the largest ink component.

    Returns the soft ink fraction (0 = paper, 1 = ink, linear in luminance, restricted to the
    neighbourhood of the main component), the binary black mask, the thresholds and the colours.
    """
    g = luminance(rgb)
    t_ink, t_paper = threshold_multiotsu(g, classes=3)
    ink_g, paper_g = np.median(g[g < t_ink]), np.median(g[g > t_paper])
    lab, n = ndimage.label(g < t_ink, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    black = lab == sizes.argmax()
    near = ndimage.binary_dilation(black, iterations=3)
    soft = np.clip((paper_g - g) / (paper_g - ink_g), 0, 1) * near
    return {
        "soft": soft.astype(np.float32),
        "black": black,
        "n_components": n,
        "thresholds": (float(t_ink), float(t_paper)),
        "ink_rgb": np.median(rgb[black], axis=0),
        "paper_rgb": np.median(rgb[g > t_paper], axis=0),
    }


def outside_and_holes(black: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Background connected to the border, and the label image of the holes in `black`."""
    lab, n = ndimage.label(~black)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    outside = np.isin(lab, border[border > 0])
    holes, n_holes = ndimage.label(~black & ~outside)
    return outside, holes, n_holes


# ---------------------------------------------------------------- symmetry


def rotation(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def reflection(b: float) -> np.ndarray:
    """Reflection across the line through the origin at angle `b`."""
    c, s = np.cos(2 * b), np.sin(2 * b)
    return np.array([[c, s], [s, -c]])


def dihedral(n: int, theta: float) -> list[np.ndarray]:
    """The 2n elements of D_n: rotations k*2pi/n, then the n mirrors at theta + k*pi/n."""
    rots = [rotation(2 * np.pi * k / n) for k in range(n)]
    return rots + [reflection(theta + np.pi * k / n) for k in range(n)]


def apply(A: np.ndarray, pts: np.ndarray, c) -> np.ndarray:
    """Apply the linear map A about the centre c to points (..., 2)."""
    return (np.asarray(pts) - c) @ A.T + c


def warp(img: np.ndarray, A: np.ndarray, c, order: int = 1) -> np.ndarray:
    """The image moved by the isometry p -> c + A (p - c) (A orthogonal)."""
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    q = apply(A.T, np.stack([x, y], axis=-1), c)  # inverse map: source of each pixel
    return ndimage.map_coordinates(img, [q[..., 1], q[..., 0]], order=order, cval=0.0)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Soft intersection over union of two images in [0, 1]."""
    return float(np.minimum(a, b).sum() / np.maximum(a, b).sum())


def block_mean(img: np.ndarray, k: int) -> np.ndarray:
    h, w = img.shape
    return img[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def find_symmetry(soft: np.ndarray, n: int = 6, k: int = 3) -> dict:
    """Centre and mirror angle of a D_n symmetry, by maximising the overlap of the image with
    its rotated and reflected copies (on a k-times smaller image for speed).

    theta is the mirror through the arm that reaches furthest (an arm axis), in radians,
    measured from +x towards +y (clockwise on screen).
    """
    small = block_mean(soft, k)
    yy, xx = np.nonzero(small > 0.5)
    c0 = np.array([xx.mean(), yy.mean()])
    rots = [rotation(2 * np.pi * j / n) for j in (1, 2)]  # these two fix the centre

    def rot_loss(c):
        return -np.mean([overlap(small, warp(small, A, c)) for A in rots])

    c = minimize(rot_loss, c0, method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-5}).x

    def mirror_loss(t):
        return -overlap(small, warp(small, reflection(t), c))

    step = np.pi / n
    grid = np.linspace(0, step, 61)[:-1]
    t0 = grid[np.argmin([mirror_loss(t) for t in grid])]
    d = grid[1] - grid[0]
    t = minimize_scalar(mirror_loss, bounds=(t0 - d, t0 + d), method="bounded").x
    c = c * k + (k - 1) / 2  # back to full-resolution pixel coordinates

    # the two mirror families (t and t + pi/n): the arm axis is the one with ink further out
    def reach(a):
        r = np.arange(0, max(soft.shape))
        x, y = c[0] + r * np.cos(a), c[1] + r * np.sin(a)
        inside = (x >= 0) & (x < soft.shape[1] - 1) & (y >= 0) & (y < soft.shape[0] - 1)
        v = ndimage.map_coordinates(soft, [y[inside], x[inside]], order=1)
        return r[inside][v > 0.5].max(initial=0)

    reaches = [np.mean([reach(a + 2 * np.pi * j / n) for j in range(n)]) for a in (t, t + step)]
    theta = (t if reaches[0] >= reaches[1] else t + step) % (2 * step)
    group = dihedral(n, theta)
    scores = [overlap(soft, warp(soft, A, c)) for A in group]
    return {"center": c, "theta": float(theta), "group": group, "scores": scores}


def symmetrize_image(soft: np.ndarray, group, c) -> np.ndarray:
    """Average of the image over the group: exactly symmetric up to resampling."""
    return np.mean([warp(soft, A, c) for A in group], axis=0)


# ---------------------------------------------------------------- polygons


def area(poly: np.ndarray) -> float:
    """Signed area (shoelace); positive for counterclockwise in x-right, y-up axes."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    return np.array(
        [((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]
    ) / (6 * a)


def contours_xy(img: np.ndarray, level: float = 0.5) -> list[np.ndarray]:
    """Closed iso-contours as (x, y) arrays without the repeated last point."""
    padded = np.pad(img, 1)
    return [cnt[:-1, ::-1] - 1 for cnt in find_contours(padded, level)]


def seg_dist(pts: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance of each point to each edge of the closed polygon: (n_points, n_edges), and the
    position along the edge in [0, 1]."""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None]
    u = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    d = np.linalg.norm(ap - u[..., None] * ab, axis=-1)
    return d, u


def _line_fit(p: np.ndarray):
    """Total least squares line: a point and a unit direction."""
    m = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - m, full_matrices=False)
    return m, vt[0]


def _intersect(p1, d1, p2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 0.05:  # nearly parallel: no stable corner
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def refine(poly: np.ndarray, pts: np.ndarray, trim: float = 2.0) -> np.ndarray:
    """One round of: assign the points to their nearest edge, fit a straight line to each edge
    (leaving out the points within `trim` px of its corners, where a raster rounds them), and
    put each vertex at the intersection of its two lines. Corners stay sharp."""
    d, u = seg_dist(pts, poly)
    edge = d.argmin(axis=1)
    n = len(poly)
    lines = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        length = np.linalg.norm(b - a)
        cut = min(trim, 0.25 * length) / max(length, 1e-9)
        sel = (edge == i) & (u[:, i] > cut) & (u[:, i] < 1 - cut)
        if sel.sum() >= 4:
            lines.append(_line_fit(pts[sel]))
        else:
            lines.append((a, (b - a) / max(length, 1e-9)))
    out = poly.copy()
    for i in range(n):
        p = _intersect(*lines[i - 1], *lines[i])
        if p is not None and np.linalg.norm(p - poly[i]) < 4 * trim + 2:
            out[i] = p
    return out


def prune(poly: np.ndarray, min_turn: float = 12.0, min_edge: float = 2.5) -> np.ndarray:
    """Drop vertices where the outline barely turns, and merge the two ends of tiny edges."""
    changed = True
    while changed and len(poly) > 3:
        changed = False
        prev, nxt = np.roll(poly, 1, axis=0), np.roll(poly, -1, axis=0)
        v1, v2 = poly - prev, nxt - poly
        turn = np.degrees(
            np.abs(np.arctan2(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0], (v1 * v2).sum(-1)))
        )
        if turn.min() < min_turn:
            poly = np.delete(poly, turn.argmin(), axis=0)
            changed = True
            continue
        lengths = np.linalg.norm(v2, axis=1)
        if lengths.min() < min_edge:
            i = lengths.argmin()
            j = (i + 1) % len(poly)
            poly[i] = (poly[i] + poly[j]) / 2
            poly = np.delete(poly, j, axis=0)
            changed = True
    return poly


def fit_polygon(
    pts: np.ndarray, tol: float = 1.5, n_iter: int = 6, trim: float = 2.0
) -> np.ndarray:
    """A polygon with sharp corners fitted to an ordered closed contour: Douglas-Peucker for the
    corners, then edges re-fitted as straight lines and corners put at their intersections."""
    poly = approximate_polygon(np.vstack([pts, pts[:1]]), tol)[:-1]
    for _ in range(n_iter):
        poly = prune(refine(poly, pts, trim))
    return refine(poly, pts, trim)


def fit_error(poly: np.ndarray, pts: np.ndarray) -> float:
    """RMS distance (px) from the contour points to the polygon."""
    return float(np.sqrt((seg_dist(pts, poly)[0].min(axis=1) ** 2).mean()))


# ---------------------------------------------------------------- symmetric polygons


def stabilizer(poly: np.ndarray, group, c, tol: float = 2.0) -> list[np.ndarray]:
    """The group elements that map the polygon's centroid onto itself."""
    m = centroid(poly)
    return [A for A in group if np.linalg.norm(apply(A, m, c) - m) < tol]


def mirror_angle(A: np.ndarray) -> float | None:
    """Angle of the mirror line of a reflection matrix, None for a rotation."""
    if np.linalg.det(A) > 0:
        return None
    return float(np.arctan2(A[1, 0], A[0, 0]) / 2)


def unfold_symmetric(
    poly: np.ndarray, pts: np.ndarray, sym, c, theta: float, snap: float = 1.5
) -> np.ndarray:
    """An exactly invariant polygon: keep the vertices in one fundamental wedge of the
    stabilizer `sym` (|sym| = 2: one side of the mirror; 12: 30 degrees from the arm axis
    `theta`), put those within `snap` px of a wedge edge onto that mirror, and copy them by
    every element of `sym`. Vertices are ordered along the contour `pts`."""
    mirrors = [b for b in map(mirror_angle, sym) if b is not None]
    b0 = theta if len(sym) == 12 else (mirrors[0] if mirrors else theta)
    width = 2 * np.pi / len(sym)
    v = poly - c
    r = np.hypot(v[:, 0], v[:, 1])
    phi = (np.arctan2(v[:, 1], v[:, 0]) - b0) % (2 * np.pi)
    phi = np.where(phi > 2 * np.pi - np.pi / 2, phi - 2 * np.pi, phi)  # (-90, 270] degrees
    keep = []
    for edge in (0.0, width):
        dist = r * np.abs(np.sin(phi - edge))
        near = (dist < snap) & (np.abs(phi - edge) < np.pi / 2)
        if mirrors:  # project onto the mirror line
            d = np.array([np.cos(b0 + edge), np.sin(b0 + edge)])
            keep += [c + d * float(np.dot(p - c, d)) for p in poly[near]]
        phi = np.where(near, np.nan, phi)
    keep += list(poly[(phi > 0) & (phi < width)])
    cloud = np.vstack([apply(A, np.array(keep), c) for A in sym])
    out = []
    for p in cloud:
        if all(np.linalg.norm(p - q) > 1e-6 for q in out):
            out.append(p)
    out = np.array(out)
    idx = np.linalg.norm(pts[None] - out[:, None], axis=-1).argmin(axis=1)
    return out[np.argsort(idx, kind="stable")]


def is_invariant(poly: np.ndarray, sym, c, tol: float = 1e-6) -> bool:
    """Is the vertex set mapped onto itself by every element of `sym`?"""
    for A in sym:
        img = apply(A, poly, c)
        d = np.linalg.norm(poly[:, None] - img[None], axis=-1)
        if d.min(axis=1).max() > tol:
            return False
    return True


def fit_symmetric(
    pts: np.ndarray, group, c, theta: float, tol: float = 1.5
) -> tuple[np.ndarray, int]:
    """fit_polygon, then made exactly invariant under the part of the group that fixes it.
    Returns the polygon and the size of its stabilizer."""
    poly = fit_polygon(pts, tol)
    sym = stabilizer(poly, group, c)
    if len(sym) > 1:
        poly = unfold_symmetric(poly, pts, sym, c, theta)
    return poly, len(sym)


def orbit(poly: np.ndarray, group, c, tol: float = 2.0) -> list[np.ndarray]:
    """The distinct images of a polygon under the group."""
    out, seen = [], []
    for A in group:
        q = apply(A, poly, c)
        m = centroid(q)
        if all(np.linalg.norm(m - s) > tol for s in seen):
            seen.append(m)
            out.append(q)
    return out


def in_wedge(p, c, theta: float, n: int = 6, eps: float = 1.0) -> bool:
    """Is the point in the fundamental wedge [0, pi/n] from the arm axis theta (eps in px)?"""
    v = np.asarray(p) - c
    r = np.hypot(*v)
    if r < 2 * eps:
        return True
    phi = np.arctan2(v[1], v[0]) - theta
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    margin = eps / r
    return -margin <= phi <= np.pi / n + margin


# ---------------------------------------------------------------- output


def rasterize(rings: list[np.ndarray], shape) -> np.ndarray:
    """Even-odd fill of the rings (pixel centres)."""
    out = np.zeros(shape, bool)
    for r in rings:
        rr, cc = draw_polygon(r[:, 1], r[:, 0], shape)
        m = np.zeros(shape, bool)
        m[rr, cc] = True
        out ^= m
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / (a | b).sum())


def to_svg(
    rings: list[np.ndarray],
    fill: str = "#3a3633",
    background: str | None = None,
    margin: float = 10.0,
    digits: int = 2,
) -> str:
    """One path, even-odd fill: the outer outline and its holes. Straight segments only, so the
    corners are exactly the fitted ones (miter, no rounding)."""
    allp = np.vstack(rings)
    x0, y0 = allp.min(axis=0) - margin
    x1, y1 = allp.max(axis=0) + margin
    w, h = x1 - x0, y1 - y0
    d = " ".join(
        "M" + " L".join(f"{x - x0:.{digits}f},{y - y0:.{digits}f}" for x, y in r) + " Z"
        for r in rings
    )
    bg = f'<rect width="100%" height="100%" fill="{background}"/>' if background else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.{digits}f} {h:.{digits}f}" '
        f'width="{w:.0f}" height="{h:.0f}">{bg}'
        f'<path fill="{fill}" fill-rule="evenodd" stroke="none" d="{d}"/></svg>\n'
    )


def hex_color(rgb) -> str:
    return "#" + "".join(f"{int(round(255 * v)):02x}" for v in rgb)


# ---------------------------------------------------------------- test shapes and pipeline


def render_polygon(
    poly: np.ndarray, shape, k: int = 8, blur: float = 0.8, noise: float = 0.03, seed: int = 0
) -> np.ndarray:
    """A polygon drawn like a scan: area coverage of each pixel (k x k supersampling), a little
    optical blur and noise. Values in [0, 1], 1 inside."""
    h, w = shape
    rr, cc = draw_polygon(
        (poly[:, 1] + 0.5) * k - 0.5, (poly[:, 0] + 0.5) * k - 0.5, (h * k, w * k)
    )
    fine = np.zeros((h * k, w * k))
    fine[rr, cc] = 1
    img = ndimage.gaussian_filter(block_mean(fine, k), blur)
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 1)


def vertex_error(fit: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Distance from each true vertex to the nearest fitted vertex (px)."""
    return np.linalg.norm(truth[:, None] - fit[None], axis=-1).min(axis=1)


def vote_vertices(cloud: np.ndarray, n_copies: int, radius: float = 4.0) -> tuple[np.ndarray, int]:
    """Vertices most copies agree on: clusters of the folded vertices (complete linkage, at most
    `radius` px across) holding at least half the copies; returns their means and the number of
    clusters dropped."""
    labels = fcluster(linkage(cloud, "complete"), t=radius, criterion="distance")
    keep, dropped = [], 0
    for lab in np.unique(labels):
        members = cloud[labels == lab]
        if len(members) >= n_copies / 2:
            keep.append(members.mean(axis=0))
        else:
            dropped += 1
    return np.array(keep), dropped


def order_along(points: np.ndarray, contour: np.ndarray) -> np.ndarray:
    idx = np.linalg.norm(contour[None] - points[:, None], axis=-1).argmin(axis=1)
    return points[np.argsort(idx, kind="stable")]


def fit_flake(
    soft: np.ndarray,
    sym_img: np.ndarray,
    group,
    c,
    theta: float,
    min_area: float = 10.0,
    match: float = 5.0,
) -> dict:
    """The snowflake as exactly D6-symmetric polygons.

    Every component of the scan (`soft`) is fitted on its own (fit_polygon). The averaged image
    `sym_img` gives the outline and one representative hole per orbit (in the fundamental
    wedge). For each representative, the group elements that carry a scanned hole onto it fold
    its fitted vertices there: 12 copies per orbit. Vertices most copies agree on are kept
    (vote_vertices), ordered along the representative, made exactly invariant under its
    stabilizer (unfold_symmetric), and copied around the orbit. Folded holes are shifted onto
    the representative first: their centroids scatter by ~1 px (the scan is not perfectly
    symmetric), their tips by more.
    """

    def contours(img):
        cs = [q for q in contours_xy(img) if abs(area(q)) > min_area]
        return sorted(cs, key=lambda q: -abs(area(q)))

    sym_cs, raw_cs = contours(sym_img), contours(soft)
    raw_fit = [fit_polygon(q) for q in raw_cs]
    raw_m = np.array([centroid(q) for q in raw_cs])
    targets = [sym_cs[0]] + [q for q in sym_cs[1:] if in_wedge(centroid(q), c, theta)]
    polys, rows = [], []
    for t, target in enumerate(targets):
        m = centroid(target)
        pairs = (
            [(0, A) for A in group]
            if t == 0
            else [
                (i, A)
                for i in range(1, len(raw_cs))
                for A in group
                if np.linalg.norm(apply(A, raw_m[i], c) - m) < match
            ]
        )
        # each folded hole is shifted onto the representative (its centroid can be ~1 px off)
        shift = [np.zeros(2) if t == 0 else m - apply(A, raw_m[i], c) for i, A in pairs]
        cloud = np.vstack(
            [apply(A, raw_fit[i], c) + d for (i, A), d in zip(pairs, shift, strict=True)]
        )
        verts, dropped = vote_vertices(cloud, len(pairs))
        poly = order_along(verts, target)
        sym = stabilizer(poly, group, c)
        if len(sym) > 1:
            poly = unfold_symmetric(poly, target, sym, c, theta)
        raw_rms = np.mean(
            [
                fit_error(poly, apply(A, raw_cs[i], c) + d)
                for (i, A), d in zip(pairs, shift, strict=True)
            ]
        )
        v = m - c
        rows.append(
            {
                "r": float(np.hypot(*v)),
                "phi": float((np.degrees(np.arctan2(v[1], v[0]) - theta) + 180) % 360 - 180),
                "vertices": len(poly),
                "stabilizer": len(sym),
                "copies": 12 // len(sym),
                "folded": len(pairs),
                "raw_vertices": sorted({len(raw_fit[i]) for i, _ in pairs}),
                "dropped": dropped,
                "area": abs(area(poly)),
                "rms": float(raw_rms),
            }
        )
        polys.append(poly)
    rings = [polys[0]] + [p for rep in polys[1:] for p in orbit(rep, group, c)]
    return {
        "outer": polys[0],
        "reps": polys[1:],
        "rows": rows,
        "rings": rings,
        "raw_contours": raw_cs,
        "raw_fits": raw_fit,
        "n_sym_contours": len(sym_cs),
    }
