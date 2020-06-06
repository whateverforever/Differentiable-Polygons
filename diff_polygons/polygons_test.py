import numpy as np

from hypothesis import given
from hypothesis.strategies import floats, lists

from .primitives import Point, Param
from .polygons import Polygon, MultiPolygon

reals = floats(
    allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000
)

def pts_from_random_list(vals):
    if len(vals) % 2 != 0:
        vals = vals[:-1]

    xys = list(zip(vals, vals[::-1]))[:len(vals)//2]
    pts = [Point(x,y) for (x,y) in xys]

    xs = [x for (x,y) in xys]
    ys = [y for (x,y) in xys]

    return pts, xs, ys

class TestPolygon:
    def test_init(self):
        pt1 = Point(0, 0)
        pt2 = Point(1, 0)
        pt3 = Point(0.5, 1.7)

        hole1 = Point(0.1, 0.1)
        hole2 = Point(0.9, 0)
        hole3 = Point(0.5, 1.6)

        poly = Polygon([pt1, pt2, pt3])

        assert poly.points[0] == pt1
        assert poly.points[1] == pt2
        assert poly.points[2] == pt3
        assert len(poly.holes) == 0

        poly2 = Polygon([pt1, pt2, pt3], [hole1, hole2, hole3])

        assert poly2.holes[0] == hole1
        assert poly2.holes[1] == hole2
        assert poly2.holes[2] == hole3

    def test_as_numpy(self):
        points = [Point(0, 0), Point(1, 2), Point(3, 4), Point(-1, 2)]

        poly = Polygon(points)
        pts, holes = poly.as_numpy()

        assert np.allclose(pts, [[0, 0], [1, 2], [3, 4], [-1, 2]])
        assert np.allclose(holes, [])

        poly = Polygon(None, [points])
        pts, holes = poly.as_numpy()

        assert np.allclose(pts, [])
        assert np.allclose(holes, [[0, 0], [1, 2], [3, 4], [-1, 2]])

    @given(lists(reals, min_size=6, max_size=100))
    def test_bounding_box(self, vals):
        pts, xs, ys = pts_from_random_list(vals)
        poly = Polygon(pts)

        bb = poly.bounding_box

        assert bb["minx"] == np.min(xs)
        assert bb["maxx"] == np.max(xs)

        assert bb["miny"] == np.min(ys)
        assert bb["maxy"] == np.max(ys)

    @given(lists(reals, min_size=6, max_size=100), lists(lists(reals, min_size=6, max_size=100)))
    def test_copy(self, ptvals, holes):
        pts, _, _ = pts_from_random_list(ptvals)
        holes = [pts_from_random_list(holevals) for holevals in holes]

        poly = Polygon(pts, holes)
        poly2 = poly.copy()

        assert poly.points == poly2.points
        assert poly.holes == poly2.holes

        assert poly.points is not poly2.points
        assert poly.holes is not poly2.holes


