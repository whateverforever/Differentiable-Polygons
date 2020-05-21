from typing import List

from hypothesis import given
from hypothesis.strategies import floats

import numpy as np  # type:ignore
from .primitives import (
    Scalar,
    Param,
    Point,
    Vector,
    Line,
    Line2,
    update_grads,
)  # type:ignore

reals = floats(allow_infinity=False, allow_nan=False)
reals2 = lambda **kwargs: floats(allow_infinity=False, allow_nan=False, **kwargs)


def test_update_grads():
    l = a = Scalar.Param("l", 2.0)

    b = a.value ** 2 + 1.2 * l.value

    inputs = {"a": a, "l": l}
    db_dinputs = {"a": [[2 * a.value]], "l": [[1.2]]}
    b = Scalar(b).with_grads_from_previous(inputs, db_dinputs)
    db_dparams = update_grads(inputs, db_dinputs)

    assert np.allclose(db_dparams["l"], [[2 * a.value + 1.2]])

    c = b.value ** 2 + a.value ** 2 + np.cos(l.value)
    inputs = {"a": a, "b": b, "l": l}
    dc_dinputs = {
        "a": [[2 * a.value]],
        "b": [[2 * b.value]],
        "l": [[-np.sin(l.value)]],
    }
    c = Scalar(c).with_grads_from_previous(inputs, dc_dinputs)

    dc_dparams = update_grads(inputs, dc_dinputs)

    assert np.allclose(
        dc_dparams["l"],
        [[2 * b.value * (2 * l.value + 1.2) + 2 * a.value * 1 - np.sin(l.value)]],
    )


class TestScalar:
    @given(reals)
    def test_coercion(self, val):
        a = Param("a", val)
        b = Scalar(a)
        c = Scalar(val)

        assert b.value == a.value
        assert b.value == c.value
        assert b.grads == a.grads

    def test_make_param(self):
        l = Scalar.Param("l", 2.0)

        assert "l" in l.grads
        assert np.isclose(l.grads["l"], [[1.0]])

        l = Param("abc", 1.23)

        assert isinstance(l, Scalar)

    @given(reals)
    def test_eq(self, real1):
        a = Param("a", real1)
        b = Param("a", real1)

        assert a == a
        assert a == b
        assert b == a
        
        assert a == real1
        assert real1 == a

    @given(
        reals2(min_value=-1000, max_value=1000), reals2(min_value=-1000, max_value=1000)
    )
    def test_add(self, real1, real2):
        s1 = Scalar.Param("s1", real1)
        s2 = Scalar.Param("s2", real2)

        f = lambda x: x[0] + x[1]

        # Scalar - Scalar
        assert f([s1, s2]).value == real1 + real2
        check_all_grads(f, [s1, s2])

        # Scalar - float
        assert (s1 + real2).value == real1 + real2
        # float - Scalar
        assert (real1 + s2).value == real1 + real2
    
    @given(reals2(min_value=1,max_value=10), reals2(min_value=2, max_value=5))
    def test_pow(self, base, power):
        p_base = Scalar.Param("base", base)
        p_power = Scalar.Param("power", power)

        f = lambda x: x[0] ** x[1]

        # Scalar - Scalar
        assert f([p_base, p_power]).value == base ** power
        # Higher tolerance because powers are so sensitive
        check_all_grads(f, [p_base, p_power],  tol=1e-3)

        # Scalar - float
        assert (p_base ** power).value == base ** power
        # float - Scalar
        assert (base ** p_power).value == base ** power

    @given(
        reals2(min_value=-1000, max_value=1000), reals2(min_value=-1000, max_value=1000)
    )
    def test_sub(self, real1, real2):
        s1 = Scalar.Param("s1", real1)
        s2 = Scalar.Param("s2", real2)

        f = lambda x: x[0] - x[1]
        s3 = f([s1, s2])

        # Scalar - Scalar
        assert s3.value == real1 - real2
        check_all_grads(f, [s1, s2])

        # Scalar - float
        assert (s1 - real2).value == real1 - real2
        # float - Scalar
        assert (real1 - s2).value == real1 - real2

    @given(reals2(min_value=-100, max_value=100), reals2(min_value=-100, max_value=100))
    def test_mul(self, real1, real2):
        s1 = Scalar.Param("s1", real1)
        s2 = Scalar.Param("s2", real2)

        f = lambda x: x[0] * x[1]
        s3 = f([s1, s2])

        # Scalar - Scalar
        assert s3.value == real1 * real2
        check_all_grads(f, [s1, s2])

        # Scalar - float
        assert (s1 * real2).value == real1 * real2
        # float - Scalar
        assert (real1 * s2).value == real1 * real2

    @given(reals2(min_value=-1000, max_value=1000))
    def test_neg(self, real1):
        f = lambda x: -x[0]

        p1 = Param("param1", real1)
        result = f([p1])

        assert result.value == -real1
        check_all_grads(f, [p1])

    @given(reals)
    def test_recursive_params(self, some_real):
        l = Scalar.Param("l", some_real)
        l2 = Scalar.Param("l2", 0.5 * l)

        assert np.allclose(l2.grads["l2"], [[1]])
        assert np.allclose(l2.grads["l"], [[0.5]])


class TestIntegration:
    def test_TRT(self):
        l = Scalar.Param("l", 2.0)
        # TODO: l2 = Scalar.Param("l2", 0.15 * l)
        # TODO: l3 = Scalar.Param("l3", l2 + 0.5 * l)
        l2 = Scalar.Param("l2", 3.5)
        theta = Scalar.Param("theta", np.radians(45))

        pt = Point(0, 0)
        pt2 = pt.translate(Vector(l, 0))
        pt3 = pt2.rotate(pt, theta)
        pt4 = pt3.translate(Vector(0, 0.5 * l))
        pt5 = pt4.translate(Vector(4 * l2, 0))

        assert np.allclose(pt5.x.value, l.value * np.cos(theta.value) + 4 * l2.value)
        assert np.allclose(pt5.y.value, l.value * np.sin(theta.value) + 0.5 * l.value)

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
            pt2 = pt.translate(Vector(l, 0))
            pt3 = pt2.rotate(pt, theta)
            pt4 = pt3.translate(Vector(2 * l, 0))

            # should be reached for l=2.0, theta=30deg
            const_target = Vector(2 * 2.0 + 2.0 * np.sqrt(3) / 2, 2.0 * 0.5)
            diff_vec = pt4 - const_target
            length = diff_vec.norm()

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

        x0 = [1.0, np.radians(10)]

        res = optimize.minimize(f, x0, method="CG", jac=jac)
        length_reached, _ = parametric_pt(*res.x)

        assert res.success == True


class TestAPI:
    @given(reals, reals, reals, reals)
    def test_static_and_member_fun(self, x, y, shift_x, shift_y):
        a = Point(x, y)
        shift_vec = Vector(Scalar.Param("sx", shift_x), Scalar.Param("sy", shift_y))

        version1 = a.translate(shift_vec)
        version2 = Point.translate(a, shift_vec)

        assert np.allclose(version1.as_numpy(), version2.as_numpy())
        assert np.allclose(version1.grads["sx"], version2.grads["sx"])
        assert np.allclose(version1.grads["sy"], version2.grads["sy"])


class TestPoint:
    # very variable behaviour, sometimes passes, sometimes doesnt
    @given(reals2(min_value=-1, max_value=1), reals2(min_value=-100, max_value=100))
    def test_mirror_axis(self, some_m, some_b):
        def f(x):
            m, b = x
            pt = Point(3, 3)
            line = Line(m, b)
            pt_mirr = pt.mirror_across_line(line)
            return pt_mirr

        m = Param("m", some_m)
        b = Param("b", some_b)

        assert np.allclose(f([0, 2]).as_numpy(), [[3], [1]])
        check_all_grads(f, [m, b])

    def test_parameter_translate(self):
        l = Scalar.Param("l", 3.0)

        pt = Point(0, 0)
        pt2 = pt.translate(Point(l, 0))

        assert pt is not pt2
        assert pt2.x == l.value
        assert np.shape(pt2.grads["l"]) == (2, 1)
        assert np.allclose(pt2.grads["l"], [[1], [0]])

    def test_rotation(self):
        pt1 = Point(2, 0)
        origin = Point(0, 0)
        angle = np.radians(45)

        pt2 = pt1.rotate(origin, angle)

        assert pt2 is not pt1
        assert np.isclose(np.sqrt(2), pt2.x.value)
        assert np.isclose(np.sqrt(2), pt2.y.value)

    @given(
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-720, max_value=720),
    )
    def test_rotation_grad(self, x1, y1, x2, y2, angle):
        if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 1:
            return

        def f(x):
            x1, y1, x2, y2, angle = x

            pt1 = Point(x1, y1)
            pt2 = Point(x2, y2)

            return pt1.rotate(pt2, angle)

        x1 = Param("x1", x1)
        y1 = Param("y1", y1)
        x2 = Param("x2", x2)
        y2 = Param("y2", y2)
        angle = Param("angle", np.radians(angle))
        check_all_grads(f, [x1, y1, x2, y2, angle])

    @given(reals, reals, reals, reals)
    def test_subtraction(self, x1, y1, x2, y2):
        pt1 = Point(x1, y1)
        pt2 = Point(x2, y2)

        diff_vec = pt1 - pt2

        assert diff_vec.x == x1 - x2
        assert diff_vec.y == y1 - y2

    @given(
        reals,
        reals,
        floats(allow_infinity=False, allow_nan=False, min_value=-1e10, max_value=1e10),
    )
    def test_div(self, x, y, s):
        x = Param("x", x)
        y = Param("y", y)
        s = Param("s", s)

        pt = Point(x, y)

        try:
            res = pt / s

            assert pt.x / s.value == res.x
            assert pt.y / s.value == res.y
        except ZeroDivisionError:
            pass

    @given(
        reals2(min_value=-1000, max_value=1000),
        reals2(min_value=-1000, max_value=1000),
        reals2(min_value=-1000, max_value=1000),
        reals2(min_value=-1000, max_value=1000),
    )
    def test_add(self, x1, y1, x2, y2):
        x1 = Param("x1", x1)
        y1 = Param("y1", y1)

        x2 = Param("x2", x2)
        y2 = Param("y2", y2)

        pt1 = Point(x1, y1)
        pt2 = Point(x2, y2)

        # TODO: Remove .value by allowing Scalar
        # to be compared to a float (ignoring the gradient information)
        assert (pt1 + pt2).x == (x1 + x2).value
        assert (pt1 + pt2).y == (y1 + y2).value
        assert pt1 + pt2 == pt2 + pt1

    @given(reals, reals, reals)
    def test_mul(self, x, y, scalar):
        xx = Param("xx", x)
        yy = Param("yy", y)

        pt = Point(xx, yy)
        s = Param("s", scalar)

        # assert (s * pt).x == (pt * s).x
        res = pt * s

        assert scalar * x == res.x
        assert scalar * y == res.y

        assert np.allclose(res.grads["s"], pt.as_numpy())
        assert np.allclose(res.grads["s"], pt.as_numpy())

        assert np.allclose(res.grads["xx"], [[s.value], [0.0]])
        assert np.allclose(res.grads["yy"], [[0.0], [s.value]])


class TestLine2:
    def test_from_points(self):
        pt1 = Point(1, 1)
        pt2 = pt1.translate(Point(1.23, 0))
        line = Line2.make_from_points(pt2, pt1)

    def test_intersection(self):
        line1 = Line2.make_from_points(Point(0, 1), Point(1, 2))
        line2 = Line2.make_from_points(Point(4, 0), Point(3, 2))

        intersect = line1.intersect(line2)

        assert np.isclose(intersect.x.value, 2.3333333333)
        assert np.isclose(intersect.y.value, 3.3333333333)

        h = Param("h", 2.0)
        line_a = Line2.make_from_points(Point(0, 0), Point(1, 1))
        line_horiz = Line2.make_from_points(Point(0, h), Point(5, h))

        intersect = line_a.intersect(line_horiz)
        intersect_2 = line_horiz.intersect(line_a)
        assert np.allclose(intersect.as_numpy(), [[2], [2]])
        assert np.allclose(intersect.as_numpy(), intersect_2.as_numpy())


class TestLine:
    def test_from_points(self):
        l = Scalar.Param("l", 2.0)
        theta = Scalar.Param("theta", np.radians(60))

        pt = Point(1, 1)
        pt2 = pt.translate(Point(l, 0))

        line = Line.make_from_points(pt2, pt)

        assert np.shape(line.grads["l"]) == (2, 1)
        assert line.b == 1
        assert line.m == 0

        # TODO: Check grad values

    # TODO: Replace this shitty line parameterization with singularities everywhere
    @given(reals2(min_value=-88, max_value=88), reals)
    def test_rotation(self, real_angle, real_b):
        theta = Param("theta", np.radians(real_angle))

        def f(x):
            (theta,) = x

            line = Line(0, real_b)
            line2 = line.rotate_ccw(theta)

            return line2

        assert np.isclose(f([theta]).m.value, np.tan(theta.value))
        assert f([theta]).b == real_b
        # check_all_grads(f, [theta])

    # TODO: With new parameterized line: Take 0 into account and negative vals
    @given(reals2(min_value=0.1, max_value=100))
    def test_translation(self, l_val):
        def f(x):
            (l,) = x

            pt = Point(1, 1)
            pt2 = pt.translate(Point(l, 0))

            line = Line.make_from_points(pt2, pt)
            line2 = line.translate(Vector(l, l))
            return line2

        l = Param("l", l_val)
        line2 = f([l])

        assert line2.b == l_val + 1
        assert line2.m == 0
        check_all_grads(f, [l])

    def test_intersection(self):
        line1 = Line.make_from_points(Point(0, 1), Point(1, 2))
        line2 = Line.make_from_points(Point(4, 0), Point(3, 2))

        intersect = line1.intersect(line2)

        assert np.isclose(intersect.x.value, 2.33333)
        assert np.isclose(intersect.y.value, 3.33333)

        h = Param("h", 2.0)
        line_a = Line.make_from_points(Point(0, 0), Point(1, 1))
        line_horiz = Line.make_from_points(Point(0, h), Point(5, h))

        intersect = line_a.intersect(line_horiz)
        assert np.allclose(intersect.as_numpy(), [[2], [2]])
        assert np.allclose(intersect.grads["h"], [[1], [1]])

        intersect_2 = line_horiz.intersect(line_a)
        assert np.allclose(intersect.as_numpy(), intersect_2.as_numpy())
        assert np.allclose(intersect.grads["h"], intersect_2.grads["h"])

    @given(
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
    )
    def test_intersection_grad(self, m1, m2, b1, b2):
        # For nearly colinear lines, the intersection is too sensitive
        # for comparison of the numerical gradient
        if abs(np.arctan(m1) - np.arctan(m2)) <= np.radians(10):
            return

        def f(x) -> Point:
            m1, m2, b1, b2 = x

            line_a = Line(m1, b1)
            line_b = Line(m2, b2)
            intersect2 = line_a.intersect(line_b)

            return intersect2

        m1 = Param("m1", m1)
        m2 = Param("m2", m2)
        b1 = Param("b1", b1)
        b2 = Param("b2", b2)

        check_all_grads(f, [m1, m2, b1, b2])

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


def central_diff(fun, x, epsilon):
    res = (fun(x + epsilon) - fun(x - epsilon)) / (2 * epsilon)

    if isinstance(res, Scalar):
        return res.value

    return res


def check_grad_c(fun, gradfun, x, epsilon):
    grad_numerical = central_diff(fun, x, epsilon)
    grad_analytic = gradfun(x)

    return abs(grad_analytic - grad_numerical)


def check_all_grads(fun, x: List[Param], tol=1e-5, eps=1e-6):
    """
    Takes a GradientCarrier object and checks all its gradients against the 
    numerical equivalent.
    """

    n_notparams = sum(
        [1 for xi in x if not (isinstance(xi, Scalar) and hasattr(xi, "name"))]
    )
    if n_notparams > 0:
        raise TypeError("Can't use anything else than `Param` for check_all_grads")

    gradients = [param.name for param in x]
    xx = np.array(x)

    for iout, output in enumerate(fun(x)._params):
        for igrad, grad_name in enumerate(gradients):

            def fun_m(x_scalar):
                x_full = xx.copy()
                x_full[igrad] = x_scalar[0]

                return fun(x_full)._params[iout]

            def grad_m(x_scalar):
                x_full = xx.copy()
                x_full[igrad] = x_scalar[0]

                return float(fun(x_full).grads[grad_name][iout])

            # Need to coerce this multivariate function into a function of one
            # variable in order for check_grad to compare the correct entries
            partial_x = np.array([xx[igrad]])

            try:
                assert check_grad_c(fun_m, grad_m, partial_x, epsilon=eps) < tol
            except Exception as e:
                gradient_numerical = central_diff(fun_m, partial_x, epsilon=eps)
                gradient_analytic = grad_m(partial_x)

                print(f"In output {iout}, failing in grad `{grad_name}`")
                print(f"Grad numerical:  {gradient_numerical}")
                print(f"Grad analytical: {gradient_analytic}")

                raise e

