from hypothesis import given, reject
from hypothesis.strategies import integers, text, floats

import numpy as np  # type:ignore
from scipy.optimize import check_grad  # type:ignore
from main import (
    diffvec,
    Scalar,
    Param,
    Point,
    Vector,
    Line,
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


def test_make_param():
    l = Scalar.Param("l", 2.0)

    assert "l" in l.grads
    assert np.isclose(l.grads["l"], [[1.0]])

    l = Param("abc", 1.23)

    assert isinstance(l, Scalar)


def test_param_and_point():
    l = Scalar.Param("l", 2.0)
    pt = Point(0, l)

    assert "l" in pt.grads
    assert np.allclose(pt.grads["l"], [[0], [1.0]])


def test_multiplied_param():
    l = Scalar.Param("l", 2.0)

    assert isinstance(2 * l, Scalar)
    assert (2 * l).value == 4.0

    assert set((2 * l).grads.keys()) == {"l"}
    assert (2 * l).grads["l"].shape == (1, 1)
    assert np.isclose((2 * l).grads["l"], [[2.0]])

    pt = Point(0, 2 * l)
    assert np.allclose(pt.grads["l"], [[0.0], [2.0]])


def test_recursive_params():
    l = Scalar.Param("l", 2.0)
    l2 = Scalar.Param("l2", 0.5 * l)

    assert np.allclose(l2.grads["l2"], [[1]])
    assert np.allclose(l2.grads["l"], [[0.5]])


def test_TRT():
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


def test_norm_optimization():
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
        diff_vec = diffvec(pt4, const_target)
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


def test_mirror_axis():
    pt = Point(3, 3)
    line = Line(0, 2)

    pt_mirr = pt.mirror_across_line(line)

    assert np.allclose(pt_mirr.as_numpy(), [[3], [1]])
    # TODO: Gradient


@given(reals, reals, reals, reals)
def test_subtraction(x1, y1, x2, y2):
    pt1 = Point(x1, y1)
    pt2 = Point(x2, y2)

    diff_vec = pt1 - pt2

    assert diff_vec.x == x1 - x2
    assert diff_vec.y == y1 - y2

    assert diff_vec.x == diffvec(pt1, pt2).x
    assert diff_vec.y == diffvec(pt1, pt2).y


@given(
    reals,
    reals,
    floats(allow_infinity=False, allow_nan=False, min_value=-1e10, max_value=1e10),
)
def test_div(x, y, s):
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


@given(reals, reals, reals)
def test_mul(x, y, scalar):
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


@given(reals, reals, reals, reals)
def test_static_and_member_fun(x, y, shift_x, shift_y):
    a = Point(x, y)
    shift_vec = Vector(Scalar.Param("sx", shift_x), Scalar.Param("sy", shift_y))

    version1 = a.translate(shift_vec)
    version2 = Point.translate(a, shift_vec)

    assert np.allclose(version1.as_numpy(), version2.as_numpy())
    assert np.allclose(version1.grads["sx"], version2.grads["sx"])
    assert np.allclose(version1.grads["sy"], version2.grads["sy"])


@given(reals, reals, reals, reals)
def test_diffvec(x1, y1, x2, y2):
    diff_vec = diffvec(Point(x1, y1), Point(x2, y2))

    assert diff_vec.x == x1 - x2
    assert diff_vec.y == y1 - y2


@given(reals, reals, reals, reals)
def test_diffvec_param(x1, y1, x2, y2):
    x1 = Scalar.Param("x1", x1)
    y1 = Scalar.Param("y1", y1)

    x2 = Scalar.Param("x2", x2)
    y2 = Scalar.Param("y2", y2)

    diff_vec = diffvec(Point(x1, y1), Point(x2, y2))

    assert np.allclose(diff_vec.grads["x1"], [[1], [0]])
    assert np.allclose(diff_vec.grads["y1"], [[0], [1]])
    assert np.allclose(diff_vec.grads["x2"], [[-1], [0]])
    assert np.allclose(diff_vec.grads["y2"], [[0], [-1]])


def test_parameter_translate():
    l = Scalar.Param("l", 3.0)

    pt = Point(0, 0)
    pt2 = pt.translate(Point(l, 0))

    assert pt is not pt2
    assert pt2.x == l.value
    assert np.shape(pt2.grads["l"]) == (2, 1)
    assert np.allclose(pt2.grads["l"], [[1], [0]])


def test_rotation():
    pt1 = Point(2, 0)
    origin = Point(0, 0)
    angle = np.radians(45)

    pt2 = pt1.rotate(origin, angle)

    assert pt2 is not pt1
    assert np.isclose(np.sqrt(2), pt2.x)
    assert np.isclose(np.sqrt(2), pt2.y)


@given(reals2(min_value=-np.radians(720), max_value=np.radians(720)))
def test_rotation_parametric_angle(angle):
    theta = Scalar.Param("theta", angle)

    origin = Point(0, 0)
    pt1 = Point(2, 0)
    pt2 = pt1.rotate(origin, theta)

    assert pt2.grads["theta"].shape == (2, 1)
    assert np.allclose(pt2.grads["theta"], [[-2 * np.sin(angle)], [2 * np.cos(angle)]],)

    def fun(angle):
        theta = Scalar.Param("theta", angle)
        origin = Point(0, 0)
        pt1 = Point(2, 0)
        pt2 = pt1.rotate(origin, theta)

        return pt2

    params = [theta]

    h = 0.01
    for param in params:
        x = param.value

        xs = [x - h, x, x + h]
        fs = [fun(x).as_numpy() for x in xs]

        grad_analyt = fun(x).grads[param.name]
        grad_numeric = np.gradient(fs, xs, axis=0)[1]

        eps = 1e-10
        rel_err = np.abs((grad_analyt - grad_numeric) / (grad_analyt + eps))

        assert np.all(rel_err < 1e-4)


@given(reals, reals)
def test_rotation_parametric_pt(l, angle):
    # TODO replace random with Hypothesis
    l = Scalar.Param("l", l)
    angle = np.radians(angle)
    theta = Scalar.Param("theta", angle)

    pt1 = Point(l, 0)
    origin = Point(0, 0)
    pt2 = pt1.rotate(origin, theta)

    assert pt2.grads["l"].shape == (2, 1)
    assert np.allclose(pt2.grads["l"], [[np.cos(angle)], [np.sin(angle)]])
    assert np.allclose(
        pt2.grads["theta"], [[-l.value * np.sin(angle)], [l.value * np.cos(angle)]]
    )


def test_line_from_points():
    l = Scalar.Param("l", 2.0)
    theta = Scalar.Param("theta", np.radians(60))

    pt = Point(1, 1)
    pt2 = pt.translate(Point(l, 0))

    line = Line.make_from_points(pt2, pt)

    assert np.shape(line.grads["l"]) == (2, 1)
    assert line.b == 1
    assert line.m == 0

    # TODO: Check grad values


@given(reals2(min_value=-720, max_value=720), reals)
def test_line_rotation(real_angle, real_b):
    real_angle = np.around(real_angle, decimals=4)

    from scipy.optimize import check_grad  # type:ignore

    def f(angle):
        theta_val = float(angle)

        line = Line(0, real_b)
        theta = Scalar.Param("theta", np.radians(theta_val))
        line2 = line.rotate_ccw(theta)

        return line2

    assert np.isclose(f(real_angle).m, np.tan(np.radians(real_angle)))
    assert f(real_angle).b == real_b

    assert (
        check_grad(
            lambda x: f(x).m,
            lambda x: float(f(x).grads["theta"][0]),
            np.array([real_angle]),
        )
        < 1e-4
    )


# TODO: With new parameterized line: Take 0 into account and negative vals
@given(reals2(min_value=0.1, max_value=100))
def test_translation(l_val):
    l = Scalar.Param("l", l_val)

    pt = Point(1, 1)
    pt2 = pt.translate(Point(l, 0))

    line = Line.make_from_points(pt2, pt)
    line2 = line.translate(Vector(l, l))

    assert line2.b == l.value + 1
    assert line2.m == 0

    assert "l" in line2.grads
    # TODO: Check grad values

    def f(x):
        l = float(x)

        l = Scalar.Param("l", l)
        pt = Point(1, 1)
        pt2 = pt.translate(Point(l, 0))

        line = Line.make_from_points(pt2, pt)
        line2 = line.translate(Vector(l, l))

        return line2.b

    def grads(x):
        l = float(x)

        l = Scalar.Param("l", l)
        pt = Point(1, 1)
        pt2 = pt.translate(Point(l, 0))

        line = Line.make_from_points(pt2, pt)
        line2 = line.translate(Vector(l, l))

        return line2.grads["l"][1][0]

    assert check_grad(f, grads, np.array([l_val])) < 1e-4


def test_from_const():
    line = Line(0.5, 0)

    assert line.m == 0.5
    assert line.b == 0

    assert line.grads == {}


def test_from_params():
    param_m = Scalar.Param("param_m", 0.5)
    param_b = Scalar.Param("param_b", 0.5)

    line = Line(param_m, param_b)

    assert "param_m" in line.grads
    assert "param_b" in line.grads

    assert line.grads["param_m"].shape == (2, 1)
    assert line.grads["param_b"].shape == (2, 1)

    assert np.allclose(line.grads["param_m"], [[1], [0]])
    assert np.allclose(line.grads["param_b"], [[0], [1]])


def test_intersection():
    line1 = Line.make_from_points(Point(0, 1), Point(1, 2))
    line2 = Line.make_from_points(Point(4, 0), Point(3, 2))

    intersect = line1.intersect(line2)

    assert np.isclose(intersect.x, 2.33333)
    assert np.isclose(intersect.y, 3.33333)

    h = Param("h", 2.0)
    line_a = Line.make_from_points(Point(0, 0), Point(1, 1))
    line_horiz = Line.make_from_points(Point(0, h), Point(5, h))

    intersect = line_a.intersect(line_horiz)
    assert np.allclose(intersect.as_numpy(), [[2], [2]])
    assert np.allclose(intersect.grads["h"], [[1], [1]])

    intersect_2 = line_horiz.intersect(line_a)
    assert np.allclose(intersect.as_numpy(), intersect_2.as_numpy())
    assert np.allclose(intersect.grads["h"], intersect_2.grads["h"])

    # Imagine a line like a clock dial
    m = Param("m", 1.23)
    b = 0.0
    line_dial = Line(m, b)

    # As we wiggle its slope, the x-coordinate shifts, while the
    # y-coordinate stays untouched
    intersect2 = line_dial.intersect(line_horiz)

    assert np.allclose(intersect2.grads["m"][0], [(b - h.value) / (m.value ** 2)])
    assert np.allclose(intersect2.grads["m"][1], [0.0])
