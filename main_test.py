import unittest as ut
import numpy as np  # type:ignore
from main import (
    translate,
    rotate_param,
    diffvec,
    norm,
    Param,
    Point,
    Line,
    make_param,
)  # type:ignore


class TestParameter(ut.TestCase):
    def test_make_param(self):
        l = make_param("l", 2.0)

        assert "l" in l.grads
        assert np.isclose(l.grads["l"], [[2.0]])

    def test_param_and_point(self):
        print(">> Shoop di woop")
        l = make_param("l", 2.0)

        pt = Point(0, l)

        assert "l" in pt.grads
        assert np.allclose(pt.grads["l"], [[0], [2.0]])


"""
class TestOverall(ut.TestCase):
    def test_norm_optimization(self):
        from scipy import optimize  # type: ignore

        def parametric_pt(l=2.0, theta=np.radians(60)):
            l = Param("l", l)
            theta = Param("theta", theta)

            pt = Point(0, 0)

            pt2 = translate(pt, [l, 0])
            pt3 = rotate_param(pt2, pt, theta)
            pt4 = translate(pt3, [2 * l, 0])

            diff_vec = diffvec(pt4, Point(8, 2))

            length = norm(diff_vec)

            return length, length.grads

        def f(x):
            l, theta = x
            scl, _ = parametric_pt(l, theta)
            dist = scl.value

            return dist

        def jac(x):
            l, theta = x
            _, pt_grads = parametric_pt(l, theta)

            grads = []
            for param in ["l", "theta"]:
                grads.append(pt_grads[param])

            return np.squeeze(grads)

        x0 = [2.0, np.radians(60)]
        xs = []

        res = optimize.minimize(f, x0, method="L-BFGS-B", jac=jac)
        length_reached, _ = parametric_pt(*res.x)

        assert np.isclose(length_reached.value, 0)


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
