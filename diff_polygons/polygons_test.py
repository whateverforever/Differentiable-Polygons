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
