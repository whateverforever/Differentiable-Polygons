import copy

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from numbers import Number

from scipy import optimize  # type: ignore


class Param:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.factor = 1
        self.power = 1

    def __rmul__(self, other):
        self_copy = copy.deepcopy(self)

        if isinstance(other, Number):
            self_copy.factor = other

        return self_copy

    def __repr__(self):
        return "Param({}*{}^{}={})".format(
            self.factor, self.name, self.power, self.value
        )

    def compute(self):
        return self.factor * self.value ** self.power

    def grad(self):
        return self.factor * self.power * self.value ** (self.power - 1)


class GradientCarrier:
    def __init__(self):
        self.gradients = {}

    @property
    def grads(self):
        return copy.deepcopy(self.gradients)

    def update_grads(self, new_grads):
        """
        Chains new gradients onto the current ones
        """
        new_scalar = copy.deepcopy(self)
        old_grads = new_scalar.gradients
        updated_grads = {}

        incoming_params = set(old_grads.keys()).union(set(new_grads.keys()))

        for param in incoming_params:
            if param == "d_dprevpt":
                continue

            # we already have a trace to this parameter
            if param in old_grads:
                # we take our gradient towards last point
                # and the gradient of old point towards parameter
                # d_dl = d_dprev @ dprev_dl
                updated_grads[param] = (
                    np.array(new_grads["d_dprevpt"]) @ old_grads[param]
                )

            # we don't have a trace yet, meaning we start one
            else:
                # d_dl = XXX
                updated_grads[param] = new_grads[param]

        new_scalar.gradients = updated_grads
        return new_scalar


class Scalar(GradientCarrier):
    def __init__(self, value):
        self.value = value
        super().__init__()

    def __repr__(self):
        return "Scalar({:.4f})".format(self.value)

    @staticmethod
    def from_point(pt):
        aa = Scalar(0)
        aa.value = pt.x
        aa.gradients = copy.deepcopy(pt.gradients)

        return aa


class Point(GradientCarrier):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__()

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x, self.y)


Vector = Point


class Line(GradientCarrier):
    """ Simple class representing a line, used to construct the unit cell """

    # TODO: Replace by better representation with no singularities
    def __init__(self, m, b):
        self.m = m
        self.b = b
        super().__init__()

    @staticmethod
    def make_from_points(p1: Point, p2: Point):
        return Line.make_from_points_({"p1": p1, "p2": p2})

    @staticmethod
    def make_from_points_(args: dict):
        """ Returns line that goes through p1 and p2 """
        p1 = args["p1"]
        p2 = args["p2"]

        x1 = p1.x
        y1 = p1.y

        x2 = p2.x
        y2 = p2.y

        b = (y1 * x2 - y2 * x1) / (x2 - x1)
        m = (y2 - b) / x2

        db_dp1 = [
            [
                -(y2 / (-x1 + x2)) + (x2 * y1 - x1 * y2) / (-x1 + x2) ** 2,
                x2 / (-x1 + x2),
            ]
        ]
        db_dp2 = [
            [
                y1 / (-x1 + x2) - (x2 * y1 - x1 * y2) / (-x1 + x2) ** 2,
                -(x1 / (-x1 + x2)),
            ]
        ]

        dm_dp1 = [
            [
                (y2 / (-x1 + x2) - (x2 * y1 - x1 * y2) / (-x1 + x2) ** 2) / x2,
                -(1 / (-x1 + x2)),
            ]
        ]
        dm_dp2 = [
            [
                (-(y1 / (-x1 + x2)) + (x2 * y1 - x1 * y2) / (-x1 + x2) ** 2) / x2
                - (y2 - (x2 * y1 - x1 * y2) / (-x1 + x2)) / x2 ** 2,
                (1 + x1 / (-x1 + x2)) / x2,
            ]
        ]

        local_grads = {}
        local_grads["p1"] = d_dpt1 = np.vstack([dm_dp1, db_dp1])
        local_grads["p2"] = d_dpt2 = np.vstack([dm_dp2, db_dp2])
        local_grads["abc"] = [[0], [10]]

        # the signature for the following gradient computation:
        # inputs, local_grads --> out_grads

        incoming_parameters = []
        for input_name, input_obj in args.items():
            incoming_parameters.extend(list(input_obj.grads.keys()))
        incoming_parameters = list(set(incoming_parameters))

        # Parameters that previous operations don't know anything about
        # I.e. maybe we did translations on `l` before, and now a rotation
        # on new parameter `theta`
        own_parameters = [
            param
            for param in local_grads.keys()
            if param not in args.keys() and param not in incoming_parameters
        ]

        out_grads = {}
        for param in incoming_parameters + own_parameters:
            grads = []

            # If we have inputs that depended on parameters
            for input_name, input_obj in args.items():
                # If one of the inputs doesn't depend on the parameter, we simply
                # ignore it. No gradient information in there!
                if param in input_obj.grads:
                    dself_dinput = local_grads[input_name]
                    dinput_dparam = input_obj.grads[param]

                    grads.append(dself_dinput @ dinput_dparam)

            # If we got directly injected a parameter as input (i.e. rotate(pt, theta))
            if param in local_grads:
                dself_dparam = local_grads[param]

                grads.append(dself_dparam)

            out_grads[param] = np.sum(grads, axis=0)

        new_line = Line(m, b)
        new_line.gradients = out_grads

        print("Gradients:", new_line.grads)

        return new_line

    """
    def plot(self, ax=plt, lims=(-20, 20, 10)):
        x = np.linspace(*lims)
        y = self.m * x + self.b

        ax.plot(x, y)

    def translate(self, dx, dy):
        new_line = copy.deepcopy(self)
        new_line.b += dy
        new_line.b -= new_line.m * dx

        return new_line

    def rotate_ccw(self, theta, pivot=None):
        if pivot is None:
            pivot = [0, 0]

        new_line = copy.deepcopy(self)
        new_line = new_line.translate(-pivot[0], -pivot[1])
        new_line.m = np.tan(np.arctan(new_line.m) + theta)
        new_line = new_line.translate(pivot[0], pivot[1])

        return new_line

    def intersect(self, other_line):
        x = (other_line.b - self.b) / (self.m - other_line.m)
        y = self.m * x + self.b
        return np.array((x, y))
    """


def translate(pt, vec):
    deltax = vec[0].compute() if isinstance(vec[0], Param) else vec[0]
    deltay = vec[1].compute() if isinstance(vec[1], Param) else vec[1]

    x2 = pt.x + deltax
    y2 = pt.y + deltay

    _grads = {}
    if isinstance(vec[0], Param):
        _grads[vec[0].name] = [[vec[0].grad()], [0]]

    if isinstance(vec[1], Param):
        _grads[vec[1].name] = [[0], [vec[1].grad()]]

    d_dprevpt = np.array([[1, 0], [0, 1]])

    _grads["d_dprevpt"] = d_dprevpt

    pt2 = pt.update_grads(_grads)
    pt2.x = x2
    pt2.y = y2

    return pt2


def norm(pt):
    params = list(pt.grads.keys())

    l2_norm = np.sqrt(pt.x ** 2 + pt.y ** 2)

    grad_pt = [
        [pt.x / np.sqrt(pt.x ** 2 + pt.y ** 2), pt.y / np.sqrt(pt.x ** 2 + pt.y ** 2)]
    ]

    grads = {}
    grads["d_dprevpt"] = grad_pt

    # TODO: Point isnt the right class. Should Differentiable or Scalar or Thing or sth
    length = Scalar.from_point(pt)  # pt.update_grads(grads)
    length = length.update_grads(grads)
    length.value = l2_norm

    return length


def rotate_param(pt, origin, angle_param):
    x1 = pt.x
    y1 = pt.y

    ox = origin.x
    oy = origin.y

    angle = angle_param.compute()

    x2 = (x1 - ox) * np.cos(angle) - (y1 - oy) * np.sin(angle) + ox
    y2 = (x1 - ox) * np.sin(angle) + (y1 - oy) * np.cos(angle) + oy

    dpt = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    dorigin = [
        [-np.cos(angle) + 1, np.sin(angle)],
        [-np.sin(angle), -np.cos(angle) + 1],
    ]
    dangle = [
        [(x1 - ox) * (-np.sin(angle) - (y1 - oy) * np.cos(angle))],
        [(x1 - ox) * np.cos(angle) + (y1 - oy) * (-np.sin(angle))],
    ]

    _grads = {}
    _grads[angle_param.name] = dangle
    _grads["d_dprevpt"] = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    pt2 = pt.update_grads(_grads)
    pt2.x = x2
    pt2.y = y2

    return pt2


def diffvec(p1, p2):
    diff_vec = [p1.x - p2.x, p1.y - p2.y]

    _grads = {}
    _grads["d_dprevpt"] = np.array([[1, 0], [0, 1]])

    pt_new = p1.update_grads(_grads)
    pt_new.x = diff_vec[0]
    pt_new.y = diff_vec[1]

    return pt_new


def parametric_pt(l=2.0, theta=np.radians(60)):
    l = Param("l", l)
    theta = Param("theta", theta)

    pt = Point(0, 0)

    # TODO: This vector has to have a gradient too, so `incoming_parameters` realizes
    # that there has been a parameter injected
    pt2 = translate(pt, [l, 0])
    pt3 = rotate_param(pt2, pt, theta)
    pt4 = translate(pt3, [2 * l, 0])

    diff_vec = diffvec(pt4, Point(8, 2))

    length = norm(diff_vec)

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

    x0 = [2.0, np.radians(60)]
    xs = []

    def reporter(xk):
        xs.append(xk)

    # with jac: succ, nfev=74, nit=8
    # without jac: no succ, nfev=252, nit=7
    res = optimize.minimize(f, x0, method="BFGS", jac=jac, callback=reporter)
    length_reached, _ = parametric_pt(*res.x)

    xs = np.array(xs)
    fig, axes = plt.subplots(ncols=2)

    xxs, yys = np.meshgrid(
        np.linspace(np.min(xs[:, 0]), np.max(xs[:, 0]), 50),
        np.linspace(np.min(xs[:, 1]), np.max(xs[:, 1]), 50),
    )
    zzs = np.zeros_like(xxs)
    for ix, x in enumerate(np.linspace(np.min(xs[:, 0]), np.max(xs[:, 0]), 50)):
        for iy, y in enumerate(np.linspace(np.min(xs[:, 1]), np.max(xs[:, 1]), 50)):
            z = f([x, y])
            zzs[ix, iy] = z
    axes[0].contourf(xxs, yys, zzs, levels=50)
    axes[0].plot(xs[:, 0], xs[:, 1], "-o")
    axes[0].set_title("Solution Space")
    axes[0].set_xlabel("l")
    axes[0].set_ylabel("theta")

    axes[1].plot(range(len(xs)), [f(x) for x in xs])
    axes[1].set_title("Convergence Plot")
    axes[1].set_ylabel("Objective Fun.")
    axes[1].set_xlabel("Iteration #")

    plt.tight_layout()
    plt.show()

    print(res)
    print("x initial: {}".format(x0))
    print("x final: {}".format(res.x))

    print("Initial distance: {}".format(f(x0)))
    print("Final   distance: {}".format(length_reached))


if __name__ == "__main__":
    main()
