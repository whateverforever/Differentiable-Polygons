import numpy as np

from .primitives import Point, Param
from .polygons import Polygon, MultiPolygon


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
        points = [
            Point(0,0),
            Point(1,2),
            Point(3,4),
            Point(-1,2)
        ]

        poly = Polygon(points)
        pts, holes = poly.as_numpy()

        assert np.allclose(pts, [[0,0],[1,2],[3,4],[-1,2]])
        assert np.allclose(holes, [])