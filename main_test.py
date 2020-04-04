import unittest as ut
import numpy as np  # type:ignore
from main import translate, Param, Point, Line  # type:ignore


class TestTranslate(ut.TestCase):
    def test_parameter_translate(self):
        l = Param("l", 1.0)
        theta = Param("theta", np.radians(60))

        pt = Point(0, 0)
        pt2 = translate(pt, [l, 0])

        assert pt2.x == l.value
        assert np.shape(pt2.grads["l"]) == (2, 1)


class TestLine(ut.TestCase):
    def test_from_points(self):
        l = Param("l", 1.0)
        theta = Param("theta", np.radians(60))

        pt = Point(1, 1)
        pt2 = translate(pt, [l, 2])

        line = Line.make_from_points(pt2, pt)

        assert np.shape(line.grads["l"]) == (2, 1)

        # temporary
        assert np.allclose(line.grads["p1"], [[-2, 1], [2, -1]])
        assert np.allclose(line.grads["p2"], [[2, -1], [-4, 2]])
