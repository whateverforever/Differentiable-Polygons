import copy
import typing as ty

import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from matplotlib.path import Path  # type:ignore
import matplotlib.patches as patches  # type:ignore

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
    pt1s = corner_left + vec_left_up * (t + s)
    pt2 = corner_right + vec_bottom_left * t
    pt2s = corner_right + vec_bottom_left * (t + s)
    pt3 = corner_top + vec_right_down * t
    pt3s = corner_top + vec_right_down * (t + s)

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

    assert np.allclose(cut_lower.grads["t"], cut_lower2.grads["t"])

    tri_lr = cut_lower2.intersect(cut_right2)
    tri_ll = cut_lower2.intersect(cut_top2)
    tri_t = cut_top2.intersect(cut_right2)

    plt.plot(tri_lr.x, tri_lr.y, "o", label="tri_lr")
    plt.plot(tri_ll.x, tri_ll.y, "o", label="tri_ll")
    plt.plot(tri_t.x, tri_t.y, "o", label="tri_t")

    pt1i_ = cut_lower.intersect(cut_right2)
    pt1i = pt1i_ - ((pt1i_ - pt1) / (pt1i_ - pt1).norm()) * s

    pt2i_ = cut_right.intersect(cut_top2)
    pt2i = pt2i_ - ((pt2i_ - pt2) / (pt2i_ - pt2).norm()) * s

    pt3i_ = cut_top.intersect(cut_lower2)
    pt3i = pt3i_ - ((pt3i_ - pt3) / (pt3i_ - pt3).norm()) * s

    plt.plot(pt1i.x, pt1i.y, "o", label="pt1i")
    plt.plot(pt2i.x, pt2i.y, "o", label="pt2i")
    plt.plot(pt3i.x, pt3i.y, "o", label="pt3i")

    plt.legend()
    plt.xlim((-0.05, 2.05))
    plt.ylim((-0.05, 1.8))
    plt.show()

    # flank_lower = Polygon.MakeFromPoints([pt1, pt2, ...])
    # flank_lower_l = flank_lower.mirror_across_line(line)
    #
    # doubleflank = flank_lower.connect_with_poly(flank_lower_l)

    flank_lower = [corner_left, pt1, pt1i, tri_lr, pt2s]
    # tmp
    flank_lower_l = [point.mirror_across_line(line_left) for point in flank_lower]

    flank_right = [corner_right, pt2, pt2i, tri_t, pt3s]
    flank_top = [corner_top, pt3, pt3i, tri_ll, pt1s]
    triangle = [tri_ll, tri_lr, tri_t]

    cell_bottom = [flank_lower, flank_right, flank_top, triangle]

    lower_half = []
    cell_l = [
        [point.mirror_across_line(line_left) for point in poly] for poly in cell_bottom
    ]
    lower_half.extend(cell_l)
    lower_half.extend(
        [
            [point.mirror_across_line(line_right) for point in poly]
            for poly in cell_bottom
        ]
    )
    lower_half.extend(cell_bottom)

    # TODO: This doesn't work atm, since Point.y is not a gradient carrier
    # line_top = Line(0, corner_top.y)
    line_top = Line(0, 0).translate(corner_top)

    upper_half = [
        [point.mirror_across_line(line_top) for point in poly] for poly in lower_half
    ]

    draw_polygons([*lower_half, *upper_half], draw_grads=["l"])

    # connect flank_lower and flank_lower_l
    abc = join_two_polygons(flank_lower, flank_lower_l)

    draw_polygons([abc])


def is_oriented_ccw(poly: ty.List[Point]) -> bool:
    edge_sum = 0

    for i, pt in enumerate(poly):
        if i == 0:
            continue

        pt1 = poly[i - 1]
        pt2 = poly[i]

        edge = (pt2.x - pt1.x) * (pt2.y + pt1.y)
        edge_sum += edge

    if not np.allclose(poly[0].as_numpy(), poly[-1].as_numpy()):
        edge_sum += (poly[0].x - poly[-1].x) * (poly[0].y + poly[-1].y)

    return edge_sum < 0


def same_orientation(poly1: ty.List[Point], poly2: ty.List[Point]) -> bool:
    orient1 = is_oriented_ccw(poly1)
    orient2 = is_oriented_ccw(poly2)

    return orient1 == orient2


def flip_orientation(poly: ty.List[Point]) -> ty.List[Point]:
    return list(reversed(poly))


def join_two_polygons(poly1, poly2):
    poly1 = copy.deepcopy(poly1)
    poly2 = copy.deepcopy(poly2)

    if same_orientation(poly1, poly2):
        poly2 = flip_orientation(poly2)

    in_polys = [poly1, poly2]

    def vert_exists_in(vert: Point, other_poly: ty.List[Point]):
        shared_points = [
            idx
            for idx, point in enumerate(other_poly)
            if np.allclose(vert.as_numpy(), point.as_numpy())
        ]

        if len(shared_points) == 0:
            return False

        return int(shared_points[0])

    def vert_exists_in_other(poly_idx, vert_idx):
        other_poly_idx = 1 if poly_idx == 0 else 0

        other_poly = in_polys[other_poly_idx]
        vert = in_polys[poly_idx][vert_idx]

        return vert_exists_in(vert, other_poly)

    start_vert = None

    for i, _ in enumerate(poly1):
        if vert_exists_in_other(0, i) is False:
            start_vert = i
            break

    curr_vert_idx: int = start_vert
    curr_poly_idx: int = 0

    out_poly: ty.List[Point] = [in_polys[curr_poly_idx][curr_vert_idx]]

    while True:
        if curr_vert_idx + 1 < len(in_polys[curr_poly_idx]):
            curr_vert_idx += 1
        else:
            curr_vert_idx = 0

        if (
            vert_exists_in(in_polys[curr_poly_idx][curr_vert_idx], out_poly)
            is not False
        ):
            return np.array(out_poly)

        out_poly.append(in_polys[curr_poly_idx][curr_vert_idx])

        idx_in_other = vert_exists_in_other(curr_poly_idx, curr_vert_idx)
        if idx_in_other is False:
            pass
        else:
            curr_poly_idx = 1 if curr_poly_idx == 0 else 0
            curr_vert_idx = idx_in_other


def draw_polygons(
    polygons, ax=None, title=None, debug=False, draw_grads: ty.List[str] = None
):
    if ax is None:
        fig, ax = plt.subplots()

    for poly in polygons:
        n = len(poly)
        points2D = [(point.x, point.y) for point in poly]

        if not np.allclose(points2D[-1], points2D[0]):
            points2D.append(points2D[0])
            n += 1

        codes = []
        codes.append(Path.MOVETO)
        for i in range(n - 2):
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)

        facecolor = "orange"

        if debug:
            red = np.random.uniform(0, 1)
            green = np.random.uniform(0, 1)
            blue = np.random.uniform(0, 1)

            facecolor = [red, green, blue, 0.5]
            textcolor = [red * 0.75, green * 0.75, blue * 0.75, 1.0]

            for i, pt in enumerate(points2D):
                ax.text(*pt, i, {"color": textcolor})
            ax.scatter(*zip(*points2D), c=[textcolor])

        path = Path(points2D, codes)
        patch = patches.PathPatch(path, facecolor=facecolor, lw=0.25)
        ax.add_patch(patch)

    if draw_grads is not None:
        for poly in polygons:
            for point in poly:
                assert isinstance(point, Point)

                for grad_name in draw_grads:
                    if grad_name not in point.grads:
                        continue

                    plt.plot(
                        [point.x, point.x + point.grads[grad_name][0][0]],
                        [point.y, point.y + point.grads[grad_name][1][0]],
                        c="k",
                    )

    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-1.5, 3.5)
    ax.axis("equal")

    if title is not None:
        ax.set_title(title)

    if fig:
        plt.show()


if __name__ == "__main__":
    main()
