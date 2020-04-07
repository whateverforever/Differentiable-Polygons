from __future__ import annotations

import copy
import typing as ty

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from numbers import Number

from scipy import optimize  # type: ignore


class GradientCarrier:
    def __init__(self):
        self.gradients = {}

    @property
    def grads(self):
        return copy.deepcopy(self.gradients)

    def with_grads(self, grads):
        self_copy = copy.deepcopy(self)
        self_copy.gradients = grads
        return self_copy

    def with_grads_from_previous(self, inputs, local_grads):
        """
        Takes the inputs into the function and the local gradients that matter
        inside of the function, calculates the new gradients wrt. to the original parameters
        (by utilizing the local gradients like dself_dprevpt or dself_dparam).

        inputs: GradientCarriers that that get transformed in the operation. Their
                gradients will be used for chaining new ones onto.

        local_grads: Gradients of this particular operation wrt the inputs. Must use
                     same names.

        returns: an instance of this particular GradientCarrier with the gradient set
        """
        self_copy = copy.deepcopy(self)
        assert self_copy.gradients == {}

        self_copy.gradients = update_grads(inputs, local_grads)
        return self_copy


class Scalar(GradientCarrier):
    def __init__(self, value):
        super().__init__()

        # TODO: Change, inelegant. Lookup how coercion is usually done
        if isinstance(value, Scalar):
            self.value = value.value
            self.gradients = value.gradients
            return

        self.value = value

    def __repr__(self):
        return "Scalar({:.4f})".format(self.value)

    def __rmul__(self, scalar2):
        scalar1 = copy.deepcopy(self)

        scalar2 = Scalar(scalar2)
        inputs = {"scalar1": scalar1, "scalar2": scalar2}
        local_grads = {"scalar2": [[scalar1.value]], "scalar1": [[scalar2.value]]}

        return Scalar(scalar1.value * scalar2.value).with_grads_from_previous(
            inputs, local_grads
        )

    @staticmethod
    def Param(name, value):
        grads = {name: [[1.0]]}
        # TODO: This is a bit too much all over the place. How to harmonize?
        if isinstance(value, Scalar):
            grads.update(value.grads)
        return Scalar(value).with_grads(grads)


class Point(GradientCarrier):
    def __init__(self, x, y):
        super().__init__()

        x = Scalar(x)
        y = Scalar(y)

        inputs = {"_x": x, "_y": y}
        local_grads = {"_x": [[1], [0]], "_y": [[0], [1]]}

        self.x = x.value
        self.y = y.value
        self.gradients = update_grads(inputs, local_grads)

    def as_numpy(self):
        return np.array([[self.x], [self.y]])

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x, self.y)

    def __neg__(self):
        x1 = self.x
        y1 = self.y

        x2 = -x1
        y2 = -y1

        inputs = {"_x": x1, "_y": y1}
        grads = {"_x": [[-1], [0]], "_y": [[0], [-1]]}

        return Point(x2, y2).with_grads_from_previous(inputs, grads)

    def translate(pt: Point, vec: Point) -> Point:
        x2 = pt.x + vec.x
        y2 = pt.y + vec.y

        inputs = {"pt": pt, "vec": vec}
        _grads = {}
        _grads["pt"] = np.array([[1, 0], [0, 1]])
        _grads["vec"] = np.array([[1, 0], [0, 1]])

        pt2 = Point(x2, y2).with_grads_from_previous(inputs, _grads)
        return pt2

    def norm(pt: Point):
        eps = 1e-13

        l2_norm = np.sqrt(eps + pt.x ** 2 + pt.y ** 2)

        grad_pt = [[pt.x / l2_norm, pt.y / l2_norm]]

        inputs = {"pt": pt}
        _grads = {}
        _grads["pt"] = grad_pt

        return Scalar(l2_norm).with_grads_from_previous(inputs, _grads)

    def rotate(pt: Point, origin: Point, angle_param: Scalar) -> Point:
        # TODO: Same for points, coercion
        angle_param = Scalar(angle_param)

        x1 = pt.x
        y1 = pt.y

        ox = origin.x
        oy = origin.y

        angle = angle_param.value

        x2 = (x1 - ox) * np.cos(angle) - (y1 - oy) * np.sin(angle) + ox
        y2 = (x1 - ox) * np.sin(angle) + (y1 - oy) * np.cos(angle) + oy

        d_dpt = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        d_dorigin = [
            [-np.cos(angle) + 1, np.sin(angle)],
            [-np.sin(angle), -np.cos(angle) + 1],
        ]
        d_dangle = [
            [(x1 - ox) * (-np.sin(angle) - (y1 - oy) * np.cos(angle))],
            [(x1 - ox) * np.cos(angle) + (y1 - oy) * (-np.sin(angle))],
        ]

        inputs = {"pt": pt, "origin": origin, "angle": angle_param}

        _grads = {}
        _grads["pt"] = d_dpt
        _grads["origin"] = d_dorigin
        _grads["angle"] = d_dangle

        pt2 = Point(x2, y2).with_grads_from_previous(inputs, _grads)

        return pt2


Vector = Point
Param = Scalar.Param


def update_grads(
    inputs: ty.Dict[str, GradientCarrier],
    local_grads: ty.Dict[str, ty.Union[ty.List[Number], np.ndarray]],
):
    incoming_parameters = []
    for input_name, input_obj in inputs.items():
        # TODO: Can We get rid of this? Maybe if everything is coerced into GradientCarrier
        if not isinstance(input_obj, GradientCarrier):
            continue

        incoming_parameters.extend(list(input_obj.grads.keys()))
    incoming_parameters = list(set(incoming_parameters))

    # Parameters that previous operations don't know anything about
    # I.e. maybe we did translations on `l` before, and now a rotation
    # on new parameter `theta`
    own_parameters = [
        param
        for param in local_grads.keys()
        if param not in inputs.keys() and param not in incoming_parameters
    ]

    out_grads = {}
    for param in incoming_parameters + own_parameters:
        grads = []

        # If we have inputs that depended on parameters
        for input_name, input_obj in inputs.items():
            # TODO: Same as above
            if not isinstance(input_obj, GradientCarrier):
                continue
            # If one of the inputs doesn't depend on the parameter, we simply
            # ignore it. No gradient information in there!
            if param in input_obj.grads:
                dself_dinput = np.array(local_grads[input_name])
                dinput_dparam = np.array(input_obj.grads[param])

                grads.append(dself_dinput @ dinput_dparam)

        # TODO: Think about name clashes; How to prevent a global Param like
        # 'm' or 'x' being overriden (added twice) when a local gradient
        # with the same name exists?! :O

        out_grads[param] = np.sum(grads, axis=0)
    return out_grads


class Line(GradientCarrier):
    """ Simple class representing a line, used to construct the unit cell """

    # TODO: Replace by better representation with no singularities
    def __init__(self, m, b):
        super().__init__()

        m = Scalar(m)
        b = Scalar(b)

        inputs = {"_m": m, "_b": b}
        local_grads = {"_m": [[1], [0]], "_b": [[0], [1]]}

        self.m = m.value
        self.b = b.value
        self.gradients = update_grads(inputs, local_grads)

    def __repr__(self):
        return f"Line(m={self.m:.4f}, b={self.b:.4f})"

    @staticmethod
    def make_from_points(p1: Point, p2: Point):
        return Line.make_from_points_({"p1": p1, "p2": p2})

    @staticmethod
    def make_from_points_(inputs: ty.Dict[str, Point]):
        """ Returns line that goes through p1 and p2 """
        p1 = inputs["p1"]
        p2 = inputs["p2"]

        x1 = p1.x
        y1 = p1.y

        x2 = p2.x
        y2 = p2.y

        b = (y1 * x2 - y2 * x1) / (x2 - x1)
        m = (y2 - b) / x2

        dm_dp1 = [[(-y1 + y2) / (x1 - x2) ** 2, 1 / (x1 - x2)]]
        dm_dp2 = [[(y1 - y2) / (x1 - x2) ** 2, 1 / (-x1 + x2)]]

        db_dp1 = [[(x2 * (y1 - y2)) / (x1 - x2) ** 2, x2 / (-x1 + x2)]]
        db_dp2 = [[(x1 * (-y1 + y2)) / (x1 - x2) ** 2, x1 / (x1 - x2)]]

        local_grads = {}
        local_grads["p1"] = d_dpt1 = np.vstack([dm_dp1, db_dp1])
        local_grads["p2"] = d_dpt2 = np.vstack([dm_dp2, db_dp2])

        new_line = Line(m, b).with_grads_from_previous(inputs, local_grads)
        return new_line

    def translate(self, vec: Vector):
        line_old = copy.deepcopy(self)

        m = line_old.m
        b = line_old.b + vec.y - line_old.m * vec.x

        inputs = {"vec": vec, "line": line_old}
        grads = {}
        grads["vec"] = [[0, 0], [-line_old.m, 1]]
        grads["line"] = [[1, 0], [-vec.x, 1]]

        new_line = Line(m, b).with_grads_from_previous(inputs, grads)

        return new_line

    def rotate_ccw(self, theta: Scalar, pivot: Point = None):
        if pivot is None:
            pivot = Point(0, 0)

        line_centered = self.translate(-pivot)

        m = line_centered.m
        b = line_centered.b

        m2 = np.tan(np.arctan(m) + theta.value)
        b2 = b

        sec = lambda x: 1 / np.cos(x)

        local_grads = {}
        local_grads["theta"] = [
            [sec(theta.value + np.arctan(m)) ** 2],
            [0],
        ]
        local_grads["line_centered"] = [
            [sec(theta.value + np.arctan(m) ** 2) / (1 + m ** 2), 0],
            [0, 1],
        ]

        inputs = {"theta": theta, "pivot": pivot, "line_centered": line_centered}

        rotated_line = Line(m2, b).with_grads_from_previous(inputs, local_grads)

        new_line = rotated_line.translate(pivot)

        return new_line

    def intersect(line_1: Line, line_2: Line) -> Point:
        m1 = line_1.m
        b1 = line_1.b

        m2 = line_2.m
        b2 = line_2.b

        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

        # dx_dline meaning [dx_dm, dx_db]
        dx_dline1 = [[(b1 - b2) / (m1 - m2) ** 2, 1 / (-m1 + m2)]]
        dx_dline2 = [[(-b1 + b2) / (m1 - m2) ** 2, 1 / (m1 - m2)]]

        dy_dline1 = [[((b1 - b2) * m2) / (m1 - m2) ** 2, -(m2 / (m1 - m2))]]
        dy_dline2 = [[((-b1 + b2) * m1) / (m1 - m2) ** 2, m1 / (m1 - m2)]]

        inputs = {"line_1": line_1, "line_2": line_2}
        local_grads = {
            "line_1": np.vstack([dx_dline1, dy_dline1]),
            "line_2": np.vstack([dx_dline2, dy_dline2]),
        }

        return Point(x, y).with_grads_from_previous(inputs, local_grads)

    """
    def plot(self, ax=plt, lims=(-20, 20, 10)):
        x = np.linspace(*lims)
        y = self.m * x + self.b

        ax.plot(x, y)
    """


def diffvec(p1: Point, p2: Point):
    diff_vec = [p1.x - p2.x, p1.y - p2.y]

    inputs = {"p1": p1, "p2": p2}
    _grads = {}
    _grads["p1"] = np.array([[1, 0], [0, 1]])
    _grads["p2"] = np.array([[-1, 0], [0, -1]])

    pt_new = Point(*diff_vec).with_grads_from_previous(inputs, _grads)
    return pt_new


def parametric_pt(l=2.0, theta=np.radians(10)):
    l = Scalar.Param("l", l)
    theta = Scalar.Param("theta", theta)

    pt = Point(0, 0)
    pt2 = pt.translate(Vector(l, 0))
    pt3 = pt2.rotate(pt, theta)
    pt4 = pt3.translate(Vector(2 * l, 0))

    target = Vector(2 * 2.0 + 2.0 * np.sqrt(3) / 2, 2.0 * 0.5)
    diff_vec = diffvec(pt4, target)
    length = diff_vec.norm()
    return length, length.grads


def main():
    print("Go ####################\n\n")

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

        # print("grad={}, norm={}".format(np.squeeze(grads), np.linalg.norm(np.squeeze(grads))))
        return np.squeeze(grads)

    x0 = [1.0, np.radians(10)]
    xs = []

    def reporter(xk):
        xs.append(xk)

    # with jac: succ, nfev=74, nit=8
    # without jac: no succ, nfev=252, nit=7
    res = optimize.minimize(f, x0, method="CG", jac=jac, callback=reporter)
    length_reached, _ = parametric_pt(*res.x)

    xs = np.array(xs)
    fig, axes = plt.subplots(ncols=3)

    xxs, yys = np.meshgrid(
        np.linspace(np.min(xs[:, 0]), np.max(xs[:, 0]), 50),
        np.linspace(np.min(xs[:, 1]), np.max(xs[:, 1]), 50),
    )
    zzs = np.zeros_like(xxs)
    jjs = np.zeros((xxs.shape[0], xxs.shape[1], 2))
    for ix, x in enumerate(np.linspace(np.min(xs[:, 0]), np.max(xs[:, 0]), 50)):
        for iy, y in enumerate(np.linspace(np.min(xs[:, 1]), np.max(xs[:, 1]), 50)):
            zzs[iy, ix] = f([x, y])
            jjs[iy, ix] = jac([x, y])

    a = axes[0].contourf(xxs, yys, zzs, levels=50)
    axes[0].contour(xxs, yys, zzs, levels=20, colors="k", linewidths=0.5)
    axes[0].plot(xs[:, 0], xs[:, 1], "-o")
    axes[0].quiver(xxs[:, ::6], yys[:, ::6], jjs[:, ::6, 0], jjs[:, ::6, 1], scale=20)
    plt.colorbar(a)

    axes[0].set_title("Solution Space")
    axes[0].set_xlabel("l")
    axes[0].set_ylabel("theta")

    axes[1].plot(range(len(xs)), [f(x) for x in xs])
    axes[1].set_title("Convergence Plot")
    axes[1].set_ylabel("Objective Fun.")
    axes[1].set_xlabel("Iteration #")

    print("Example jacobian of beginning: {}, end: {}".format(jac(xs[0]), jac(xs[-1])))

    axes[2].plot(range(len(xs)), [jac(x)[1] for x in xs])
    axes[2].set_title("Infty Norm of Jacobian")
    axes[2].set_ylabel("Norm of Jac.")
    axes[2].set_xlabel("Iteration #")

    plt.tight_layout()
    plt.show()

    print(res)
    print("x initial: {}".format(x0))
    print("x final: {}".format(res.x))

    print("Jac at 2.9/0.6: {}".format(jac([2.9, 0.6])))

    print("Initial distance: {}".format(f(x0)))
    print(
        "Final   distance: {}, gradient norm: l={:.2f}, theta={:.2f}".format(
            length_reached.value,
            np.linalg.norm(length_reached.grads["l"]),
            np.linalg.norm(length_reached.grads["theta"]),
        )
    )
    print("Jac at goal", jac(res.x))

    # TODO: Gradient norms wayyyy to large (norm d_dl=2.48 and norm d_dtheta=2.46)
    # Problem appears when I rotate a parametrized vector. Gradients are zero at minimum,
    # if I rotate fixed vec, and afterwards translate by parameter


if __name__ == "__main__":
    main()
