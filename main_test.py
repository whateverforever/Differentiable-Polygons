import unittest as ut
import numpy as np  # type:ignore
from main import (
    translate,
    rotate_param,
    diffvec,
    norm,
    Scalar,
    Point,
    Vector,
    Line,
)  # type:ignore


class TestParameter(ut.TestCase):
    def test_make_param(self):
        l = Scalar.Param("l", 2.0)

        assert "l" in l.grads
        assert np.isclose(l.grads["l"], [[1.0]])

    def test_param_and_point(self):
        l = Scalar.Param("l", 2.0)
        pt = Point(0, l)

        assert "l" in pt.grads
        assert np.allclose(pt.grads["l"], [[0], [1.0]])

    def test_multiplied_param(self):
        l = Scalar.Param("l", 2.0)

        assert isinstance(2 * l, Scalar)
        assert (2 * l).value == 4.0

        assert set((2 * l).grads.keys()) == {"l"}
        assert (2 * l).grads["l"].shape == (1, 1)
        assert np.isclose((2 * l).grads["l"], [[2.0]])

        pt = Point(0, 2 * l)
        assert np.allclose(pt.grads["l"], [[0.0], [2.0]])

    def test_recursive_params(self):
        l = Scalar.Param("l", 2.0)
        l2 = Scalar.Param("l2", 0.5 * l)

        assert np.allclose(l2.grads["l2"], [[1]])
        assert np.allclose(l2.grads["l"], [[0.5]])


class TestIntegration(ut.TestCase):
    def test_TRT(self):
        l = Scalar.Param("l", 2.0)
        # TODO: l2 = Scalar.Param("l2", 0.15 * l)
        # TODO: l3 = Scalar.Param("l3", l2 + 0.5 * l)
        l2 = Scalar.Param("l2", 3.5)
        theta = Scalar.Param("theta", np.radians(45))

        pt = Point(0, 0)
        pt2 = translate(pt, Vector(l, 0))
        pt3 = rotate_param(pt2, pt, theta)
        pt4 = translate(pt3, Vector(0, 0.5 * l))
        pt5 = translate(pt4, Vector(4 * l2, 0))

        assert np.allclose(pt5.x, l.value * np.cos(theta.value) + 4 * l2.value)
        assert np.allclose(pt5.y, l.value * np.sin(theta.value) + 0.5 * l.value)

        assert "l2" in pt5.grads
        assert np.allclose(
            pt5.grads["l"], [[np.cos(theta.value)], [np.sin(theta.value) + 0.5]]
        )
        assert np.allclose(
            pt5.grads["theta"],
            [[-l.value * np.sin(theta.value)], [l.value * np.cos(theta.value)]],
        )
        assert np.allclose(pt5.grads["l2"], [[4], [0]])

    def test_norm_optimization(self):
        from scipy import optimize  # type: ignore

        def parametric_pt(l=2.0, theta=np.radians(60)):
            l = Scalar.Param("l", l)
            theta = Scalar.Param("theta", theta)

            pt = Point(0, 0)
            pt2 = translate(pt, Vector(l, 0))
            pt3 = rotate_param(pt2, pt, theta)
            pt4 = translate(pt3, Vector(2 * l, 0))

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
        l = Scalar.Param("l", 1.0)
        theta = Scalar.Param("theta", np.radians(60))

        pt = Point(0, 0)
        pt2 = translate(pt, Point(l, 0))

        assert pt2.x == l.value
        assert np.shape(pt2.grads["l"]) == (2, 1)


class TestRotation(ut.TestCase):
    def test_rotation(self):
        pt1 = Point(2, 0)
        origin = Point(0, 0)
        angle = np.radians(45)

        pt2 = rotate_param(pt1, origin, angle)

        assert np.isclose(np.sqrt(2), pt2.x)
        assert np.isclose(np.sqrt(2), pt2.y)

    def test_rotation_parametric_angle(self):
        pt1 = Point(2, 0)
        origin = Point(0, 0)
        angle_param = Scalar.Param("theta", np.radians(45))

        pt2 = rotate_param(pt1, origin, angle_param)

        assert pt2.grads["theta"].shape == (2, 1)

    def test_rotation_parametric_pt(self):
        pt1 = Point(Scalar.Param("l", 2.0), 0)
        origin = Point(0, 0)
        angle_param = np.radians(45)

        pt2 = rotate_param(pt1, origin, angle_param)

        assert pt2.grads["l"].shape == (2, 1)


class TestLine(ut.TestCase):
    def test_from_points(self):
        l = Scalar.Param("l", 2.0)
        theta = Scalar.Param("theta", np.radians(60))

        pt = Point(1, 1)
        pt2 = translate(pt, Point(l, 0))

        line = Line.make_from_points(pt2, pt)

        assert np.shape(line.grads["l"]) == (2, 1)
        assert line.b == 1
        assert line.m == 0

    def test_translation(self):
        l = Scalar.Param("l", 2.0)
        theta = Scalar.Param("theta", np.radians(60))

        pt = Point(1, 1)
        pt2 = translate(pt, Point(l, 0))

        line = Line.make_from_points(pt2, pt)
        line2 = line.translate(Vector(l, l))

        assert line2.b == l.value + 1
        assert line2.m == 0

    def test_from_const(self):
        line = Line(0.5, 0)

        assert line.m == 0.5
        assert line.b == 0

        assert line.grads == {}

    def test_from_params(self):
        param_m = Scalar.Param("param_m", 0.5)
        param_b = Scalar.Param("param_b", 0.5)

        line = Line(param_m, param_b)

        assert "param_m" in line.grads
        assert "param_b" in line.grads

        assert line.grads["param_m"].shape == (2, 1)
        assert line.grads["param_b"].shape == (2, 1)

        assert np.allclose(line.grads["param_m"], [[1], [0]])
        assert np.allclose(line.grads["param_b"], [[0], [1]])
