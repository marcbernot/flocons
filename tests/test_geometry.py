import numpy as np
import pytest

from flocons import geometry as G

DART = np.array([[50, 12], [78, 80], [50, 58], [22, 80]], float)


@pytest.mark.parametrize("scale, shape", [(1.0, (95, 100)), (0.4, (40, 42))])
def test_dart_corners_recovered(scale, shape):
    truth = DART * scale + 0.37
    cnt = max(G.contours_xy(G.render_polygon(truth, shape)), key=len)
    fit = G.fit_polygon(cnt)
    assert len(fit) == 4
    assert G.vertex_error(fit, truth).max() < 0.6


def test_group_is_closed():
    group = G.dihedral(6, 0.3)
    for A in group:
        for B in group:
            assert min(np.abs(A @ B - C).max() for C in group) < 1e-12


def test_unfold_gives_invariant_polygon():
    c = np.array([10.0, 20.0])
    group = G.dihedral(6, 0.0)
    # a hexagram-ish star outline, slightly perturbed so the fit is not symmetric by itself
    ang = np.arange(12) * np.pi / 6
    rad = np.where(np.arange(12) % 2 == 0, 9.0, 5.0)
    star = c + np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    noisy = star + np.random.default_rng(1).normal(0, 0.2, star.shape)
    out = G.unfold_symmetric(noisy, star, group, c, 0.0)
    assert len(out) == 12
    assert G.is_invariant(out, group, c)


def test_svg_has_one_evenodd_path():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    svg = G.to_svg([square, square / 2 + 2.5])
    assert svg.count("<path") == 1 and 'fill-rule="evenodd"' in svg and svg.count("Z") == 2
