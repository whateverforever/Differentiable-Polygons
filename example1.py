import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore

from main import Line, Vector, Point, Scalar, Param


def main():
    t = Param("t", 0.15)
    l = Param("l", 2.0)
    theta = Param("theta", np.radians(10))
    lower_angles = Param("lower_angles", np.radians(60))

    origin = Point(0, 0)
    pt1 = Point(l, 0)

    line_horiz = Line.make_from_points(origin, pt1)
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

    vec_left_up = (corner_top - origin) / (corner_top - origin).norm()
    pt2 = vec_left_up * t
    ax.plot(pt2.x, pt2.y, "o")

    assert pt2.grads["lower_angles"][0][0] < 0
    assert pt2.grads["lower_angles"][1][0] > 0

    line_cut = line_horiz.translate(pt2).rotate_ccw(theta)
    line_cut.plot(ax=ax, lims=lims)

    pt3 = line_cut.intersect(line_right)
    ax.plot(pt3.x, pt3.y, "o")

    print("pt3 grads", pt3.grads)

    plt.show()


if __name__ == "__main__":
    main()
