import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore

from main import Line, Vector, Point, Scalar, Param


def main():
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

    pt1 = corner_left + vec_left_up * t
    pt2 = corner_right + vec_bottom_left * t
    ax.plot(pt1.x, pt1.y, "o")

    assert pt1.grads["lower_angles"][0][0] < 0
    assert pt1.grads["lower_angles"][1][0] > 0

    # TODO: Check if gradients for d_dpt1 and d_dpt2 correct (pivot!)
    cut_lower = line_horiz.translate(vec_left_up * t).rotate_ccw(theta, pivot=pt1)
    cut_right = line_right.translate(vec_bottom_left * t).rotate_ccw(theta, pivot=pt2)

    cut_lower.plot(ax=ax, lims=lims)
    cut_right.plot(ax=ax, lims=lims)

    pt3 = cut_lower.intersect(line_right)
    ax.plot(pt3.x, pt3.y, "o")

    print("pt3 grads", pt3.grads)

    plt.show()


if __name__ == "__main__":
    main()
