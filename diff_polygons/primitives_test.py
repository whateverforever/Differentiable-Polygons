from typing import List
from math import sqrt, isclose, pi

from hypothesis import given, settings
from hypothesis.strategies import floats
 
settings.register_profile("old_pc", deadline=1000)
settings.load_profile("old_pc")

import numpy as np  # type:ignore
from .primitives import (
    Scalar,
    Param,
    Point,
    Vector,
    Line,
    update_grads,
)  # type:ignore

reals = floats(allow_infinity=False, allow_nan=False)
reals2 = lambda **kwargs: floats(allow_infinity=False, allow_nan=False, **kwargs)
# reasonable reals
rreals = floats(allow_infinity=False, allow_nan=False, min_value=-100000, max_value=100000)

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

        check_all_grads(f, [s1, s2])

        # Scalar - Scalar
        assert f([s1, s2]).value == real1 + real2
        # Scalar - float
        assert (s1 + real2).value == real1 + real2
        # float - Scalar
        assert (real1 + s2).value == real1 + real2

    @given(reals2(min_value=1, max_value=10), reals2(min_value=2, max_value=5))
    def test_pow(self, base, power):
        p_base = Scalar.Param("base", base)
        p_power = Scalar.Param("power", power)

        f = lambda x: x[0] ** x[1]

        # Higher tolerance because powers are so sensitive
        check_all_grads(f, [p_base, p_power], tol=1e-3)

        # Scalar - Scalar
        assert f([p_base, p_power]).value == base ** power
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

        check_all_grads(f, [s1, s2])

        # Scalar - Scalar
        assert s3.value == real1 - real2
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

        check_all_grads(f, [s1, s2])

        # Scalar - Scalar
        assert s3.value == real1 * real2
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
    @given(rreals.filter(lambda x: abs(x) > 0.5), rreals.filter(lambda x: abs(x) > 0.5))
    def test_mirror_axis(self, dx, dy):
        def f(x):
            dx, dy = x
            pt = Point(3, 3)
            line = Line(0, 0, dx, dy)
            pt_mirr = line.mirror_pt(pt)
            return pt_mirr

        dir_len = sqrt(dx**2 + dy**2)
        dx = Param("dx", dx/dir_len) 
        dy = Param("dy", dy/dir_len)

        assert np.allclose(f([1, 0]).as_numpy(), [[3], [-3]])
        check_all_grads(f, [dx, dy])

    def test_translate(self):
        pt = Point(0, 0)
        f = lambda x: pt.translate(Point(x[0], 0))

        l = Scalar.Param("l", 3.0)
        check_all_grads(f, [l])

        pt2 = f([l])

        assert pt is not pt2
        assert pt2.x == l.value

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

    @given(
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
        reals2(min_value=-100, max_value=100),
    )
    def test_subtraction(self, x1, y1, x2, y2):
        def f(x):
            x1, y1, x2, y2 = x

            pt1 = Point(x1, y1)
            pt2 = Point(x2, y2)

            return pt1 - pt2

        x1 = Param("x1", x1)
        y1 = Param("y1", y1)

        x2 = Param("x2", x2)
        y2 = Param("y2", y2)

        diff_vec = f([x1, y1, x2, y2])

        assert diff_vec.x == x1 - x2
        assert diff_vec.y == y1 - y2

        check_all_grads(f, [x1, y1, x2, y2])

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

        def f(x):
            x1, y1, x2, y2 = x
            pt1 = Point(x1, y1)
            pt2 = Point(x2, y2)

            return pt1 + pt2

        check_all_grads(f, [x1, y1, x2, y2])

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


class TestLine:
    @given(
    rreals,
    rreals,
    rreals.filter(lambda x: abs(x)>0.5),
    rreals.filter(lambda x: abs(x)>0.5),
    )
    def test_init_grads(self, ox, oy, dx, dy):
        def f(x):
            ox, oy, dx, dy = x

            return Line(ox, oy, dx, dy)

        ox = Param("oxx", ox)
        oy = Param("oyy", oy)

        dir_len = sqrt(dx ** 2 + dy ** 2)
        dx = Param("dx", dx / dir_len)
        dy = Param("dy", dy / dir_len)

        check_all_grads(f, [ox, oy, dx, dy])
    
    def test_mirror_pt(self):
        pt = Point(3,3)

        line_horiz = Line(0,0,1,0)
        pt2 = line_horiz.mirror_pt(pt)

        assert pt2.x == 3
        assert pt2.y == -3

        line_vert = Line(0,0,0,1)
        pt3 = line_vert.mirror_pt(pt)

        assert pt3.x == -3
        assert pt3.y == 3

    def test_from_points(self):
        pt1 = Point(1, 1)
        pt2 = pt1.translate(Point(1.23, 0))
        line = Line.make_from_points(pt2, pt1)

    def test_intersection(self):
        line1 = Line.make_from_points(Point(0, 1), Point(1, 2))
        line2 = Line.make_from_points(Point(4, 0), Point(3, 2))

        intersect = line1.intersect(line2)

        assert np.isclose(intersect.x.value, 2.3333333333)
        assert np.isclose(intersect.y.value, 3.3333333333)

        h = Param("h", 2.0)
        line_a = Line.make_from_points(Point(0, 0), Point(1, 1))
        line_horiz = Line.make_from_points(Point(0, h), Point(5, h))

        intersect = line_a.intersect(line_horiz)
        intersect_2 = line_horiz.intersect(line_a)
        assert np.allclose(intersect.as_numpy(), [[2], [2]])
        assert np.allclose(intersect.as_numpy(), intersect_2.as_numpy())

    @given(
        rreals,
        rreals,
        rreals.filter(lambda x: abs(x)>0.5),
        rreals.filter(lambda x: abs(x)>0.5),
    )
    def test_intersection_grad(self, x1, y1, dx, dy):
        dir_ = Vector(dx, dy)
        dir_length = dir_.norm()

        dir_ /= dir_length
        dx = dir_.x
        dy = dir_.y

        # check if colinear with line against we check
        if abs(np.dot([dx.value, dy.value], [np.sqrt(0.5), np.sqrt(0.5)])) > 0.8:
            return

        def f(x) -> Point:
            ox, oy, dx, dy = x

            line_a = Line(ox, oy, dx, dy)
            line_b = Line(0, 0, np.sqrt(0.5), np.sqrt(0.5))
            intersect2 = line_a.intersect(line_b)

            return intersect2

        x1 = Param("x1", x1)
        y1 = Param("y1", y1)
        dx = Param("dx", dx)
        dy = Param("dy", dy)

        check_all_grads(f, [x1, y1, dx, dy])

    @given(reals)
    def test_translation(self, offset):
        line_horiz = Line(0, 0, 1, 0)
        line_horiz = line_horiz.translate(Vector(0, offset))
        assert line_horiz.oy == offset

        line_vert = Line(0,0,0, 1)
        line_vert = line_vert.translate(Vector(offset, 0))
        assert line_vert.ox == offset

    @given(
        rreals,
        rreals,
        rreals,
        rreals,
        rreals,
        rreals,
    )
    def tests_translation_grads(self, ox, oy, dx, dy, tx, ty):
        def f(x):
            ox, oy, dx, dy, tx, ty = x

            line = Line(ox, oy, dx, dy)
            return line.translate(Vector(tx, ty))

        ox = Param("ox", ox)
        oy = Param("oy", oy)

        dir_len = sqrt(dx ** 2 + dy ** 2)
        if dir_len < 0.5:
            return

        dx = Param("dx", dx / dir_len)
        dy = Param("dy", dy / dir_len)

        tx = Param("tx", tx)
        ty = Param("ty", ty)

        check_all_grads(f, [ox, oy, dx, dy, tx, ty])
    
    def test_rotation(self):
        line = Line(0,0,1,0)
        line = line.rotate_ccw(pi/2, Point(-1,-1))

        assert np.isclose(line.dx.value, 0.0)
        assert np.isclose(line.dy.value, 1.0)

        assert np.isclose(line.ox.value, -2.0)
        assert np.isclose(line.oy.value, 0.0)
    
    @given(rreals, rreals, rreals)
    def test_rotation_grad(self, angle, px, py):
        def f(x):
            angle, px, py = x

            line = Line(1.23, 3.45, 4.56/8.17, 6.78/8.17)
            return line.rotate_ccw(angle, pivot=Point(px, py))
        
        angle = Param("angle", angle)
        px = Param("px", px)
        py = Param("py", py)

        check_all_grads(f, [angle, px, py])


def central_diff(fun, x, epsilon):
    res = (fun(x + epsilon) - fun(x - epsilon)) / (2 * epsilon)

    if isinstance(res, Scalar):
        return res.value

    return res


def check_grad_c(fun, gradfun, x, epsilon):
    grad_numerical = central_diff(fun, x, epsilon)
    grad_analytic = gradfun(x)

    err_abs = abs(grad_analytic - grad_numerical)
    err_rel = err_abs / (grad_numerical + epsilon)

    return err_abs, err_rel


def check_all_grads(fun, x: List[Param], tol=1e-5, rtol=0.01, eps=1e-6):
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

            err_abs, err_rel = check_grad_c(fun_m, grad_m, partial_x, epsilon=eps)

            try:
                assert (err_abs < tol) or (err_rel < rtol)
            except Exception as e:
                gradient_numerical = central_diff(fun_m, partial_x, epsilon=eps)
                gradient_analytic = grad_m(partial_x)

                print(f"In output {iout}, failing in grad `{grad_name}`")
                print(f"Grad numerical:  {gradient_numerical}")
                print(f"Grad analytical: {gradient_analytic}")

                raise e
