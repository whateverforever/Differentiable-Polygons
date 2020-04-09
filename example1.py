import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore

from main import Line, Vector, Point, Scalar, Param


def main():
    s = Param("s", 0.05)  # width of cuts
    t = Param("t", 0.15)
    l = Param("l", 2.0)
    theta = Param("theta", np.radians(10))
    lower_angles = Param("lower_angles", np.radians(60))

    corner_left = Point(0, 0)
    corner_right = Point(l, 0)

    line_horiz = Line.make_from_points(corner_left, corner_right)
    line_left = line_horiz.rotate_ccw(lower_angles)
    line_right = line_horiz.rotate_ccw(-lower_angles).translate(Vector(l, 0))

    corner_top = line_left.intersect(line_right)

    fig, ax = plt.subplots()
    ax.plot([corner_top.x], [corner_top.y], "o")

    lims = (0, l.value, 10)
    line_horiz.plot(ax=ax, lims=lims)
    line_left.plot(ax=ax, lims=lims)
    line_right.plot(ax=ax, lims=lims)

    assert np.allclose(corner_top.grads["lower_angles"][0], [0.0])
    assert corner_top.grads["lower_angles"][1][0] > 0

    vec_left_up = (corner_top - corner_left) / (corner_top - corner_left).norm()
    vec_bottom_left = (corner_left - corner_right) / (corner_left - corner_right).norm()
    vec_right_down = (corner_right - corner_top) / (corner_right - corner_top).norm()

    pt1 = corner_left + vec_left_up * t
    pt2 = corner_right + vec_bottom_left * t
    pt3 = corner_top + vec_right_down * t

    ax.plot(pt1.x, pt1.y, "o")

    assert pt1.grads["lower_angles"][0][0] < 0
    assert pt1.grads["lower_angles"][1][0] > 0

    # TODO: Check if gradients for d_dpt1 and d_dpt2 correct (pivot!)
    cut_lower = line_horiz.rotate_ccw(theta, pivot=corner_left).translate(
        vec_left_up * t
    )
    cut_lower2 = line_horiz.rotate_ccw(theta, pivot=corner_left).translate(
        vec_left_up * (t + s)
    )

    cut_right = line_right.rotate_ccw(theta, pivot=corner_right).translate(
        vec_bottom_left * t
    )
    cut_right2 = line_right.rotate_ccw(theta, pivot=corner_right).translate(
        vec_bottom_left * (t + s)
    )

    cut_top = line_left.rotate_ccw(theta, pivot=corner_top).translate(
        vec_right_down * t
    )
    cut_top2 = line_left.rotate_ccw(theta, pivot=corner_top).translate(
        vec_right_down * (t + s)
    )

    cut_lower.plot(ax=ax, lims=lims, label="lower")
    cut_lower2.plot(ax=ax, lims=lims, label="lowe2")
    cut_right.plot(ax=ax, lims=lims, label="right")
    cut_right2.plot(ax=ax, lims=lims, label="right2")
    cut_top.plot(ax=ax, lims=lims, label="top")
    cut_top2.plot(ax=ax, lims=lims, label="top2")

    # pt3 = cut_lower.intersect(line_right)
    # ax.plot(pt3.x, pt3.y, "o")

    # print("pt3 grads", pt3.grads)

    plt.legend()
    plt.xlim((-0.05, 2.05))
    plt.ylim((-0.05, 1.8))
    plt.show()


if __name__ == "__main__":
    main()
