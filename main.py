import copy
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from numbers import Number

from scipy import optimize  # type: ignore


class Param:
    """
    Class used so that geometrical operations can access named parameters, and
    can change their gradients (from only const). Allows for a factor in front
    and a power in the back. But not more atm.
    """

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


class Differentiable:
    def __init__(self):
        pass


class Scalar:
    def __init__(self, value):
        self.value = value
        self.gradients = {}

    def __repr__(self):
        return "Scalar({:.4f})".format(self.value)

    @property
    def grads(self):
        return copy.deepcopy(self.gradients)

    @staticmethod
    def from_point(pt):
        aa = Scalar(0)
        aa.value = pt.x
        aa.gradients = copy.deepcopy(pt.gradients)

        return aa

    # copied from Point
    def update_grads(self, new_grads):
        """
        Chains new gradients onto the current ones
        """
        new_scalar = copy.deepcopy(self)
        old_grads = new_scalar.gradients
        updated_grads = {}

        params = set(old_grads.keys()).union(set(new_grads.keys()))

        for param in params:
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


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gradients = {}

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x, self.y)

    @property
    def grads(self):
        return copy.deepcopy(self.gradients)

    def update_grads(self, new_grads):
        """
        Chains new gradients onto the current ones
        """
        pt2 = copy.deepcopy(self)
        old_grads = pt2.gradients
        updated_grads = {}

        params = set(old_grads.keys()).union(set(new_grads.keys()))

        for param in params:
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

        pt2.gradients = updated_grads
        return pt2


Vector = Point


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
