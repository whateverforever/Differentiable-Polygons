import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from scipy import optimize  # type: ignore

from primitives import Point, Param, Vector


def parametric_pt(l, theta):
    """
    Outputs the distance of the end of a kinematic chain from a predetermined target
    point. Has parameters l and theta. Analytical solution is for l=1.23, theta=34deg
    """

    l = Param("l", l)
    theta = Param("theta", theta)

    shift_right = Vector(l, 0)

    origin = Point(0, 0)
    new_pt = origin.translate(shift_right).rotate(origin, theta).translate(shift_right)

    target = Point(2.24972, 0.6878)
    dist = (new_pt - target).norm()

    return dist


def main():
    print("Go ####################\n\n")

    def f(x):
        l, theta = x
        dist = parametric_pt(*x)

        return dist.value

    def jac(x):
        l, theta = x
        dist = parametric_pt(l, theta)

        grads = []
        for param in ["l", "theta"]:
            grads.append(dist.grads[param])

        # print("grad={}, norm={}".format(np.squeeze(grads), np.linalg.norm(np.squeeze(grads))))
        return np.squeeze(grads)

    x0 = [1.0, np.radians(40)]
    xs = []

    def reporter(xk):
        xs.append(xk)

    res = optimize.minimize(f, x0, method="CG", jac=jac, callback=reporter)
    length_reached = parametric_pt(*res.x)

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


if __name__ == "__main__":
    main()
