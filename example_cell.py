import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore

from diff_polygons import (
    Line as Line,
    Vector,
    Point,
    Scalar,
    Param,
    Polygon,
    MultiPolygon,
)


def main():
    poly = create_unit_cell()
    poly.draw()


def create_unit_cell(
    s=0.05, t=0.15, l=2.0, angle_cut=10, angle_opening=0, c=0.05
) -> Polygon:

    theta = angle_cut
    phi = angle_opening

    c = Param("c", c)  # thickness of compliant hinge
    s = Param("s", s)  # width of cuts
    t = Param("t", t)
    l = Param("l", l)
    theta = Param("theta", np.radians(theta))
    opening_phi = Param("phi", np.radians(phi))
    lower_angles = Param("lower_angles", np.radians(60))

    corner_left = Point(0, 0)
    corner_right = Point(l, 0)

    line_horiz = Line.make_from_points(corner_left, corner_right)
    line_left = line_horiz.rotate_ccw(lower_angles)
    line_right = line_horiz.rotate_ccw(-lower_angles).translate(Vector(l, 0))

    corner_top = line_left.intersect(line_right)

    vec_left_up = (corner_top - corner_left) / (corner_top - corner_left).norm()
    vec_bottom_left = (corner_left - corner_right) / (corner_left - corner_right).norm()
    vec_right_down = (corner_right - corner_top) / (corner_right - corner_top).norm()

    pt1 = corner_left + vec_left_up * t
    pt1s = corner_left + vec_left_up * (t + s)
    pt2 = corner_right + vec_bottom_left * t
    pt2s = corner_right + vec_bottom_left * (t + s)
    pt3 = corner_top + vec_right_down * t
    pt3s = corner_top + vec_right_down * (t + s)

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

    tri_lr = cut_lower2.intersect(cut_right2)
    tri_ll = cut_lower2.intersect(cut_top2)
    tri_t = cut_top2.intersect(cut_right2)

    # <--
    vec_tri_bl = (tri_ll - tri_lr) / (tri_ll - tri_lr).norm()
    # /^
    vec_tri_lu = (tri_t - tri_ll) / (tri_t - tri_ll).norm()
    # \v
    vec_tri_rd = (tri_lr - tri_t) / (tri_lr - tri_t).norm()

    tri_lri = tri_lr + vec_tri_bl * c
    tri_lli = tri_ll + vec_tri_lu * c
    tri_ti = tri_t + vec_tri_rd * c

    pt1i_ = cut_lower.intersect(cut_right2)
    pt1i = pt1i_ - ((pt1i_ - pt1) / (pt1i_ - pt1).norm()) * c

    pt2i_ = cut_right.intersect(cut_top2)
    pt2i = pt2i_ - ((pt2i_ - pt2) / (pt2i_ - pt2).norm()) * c

    pt3i_ = cut_top.intersect(cut_lower2)
    pt3i = pt3i_ - ((pt3i_ - pt3) / (pt3i_ - pt3).norm()) * c

    flank_lower = Polygon([corner_left, pt1, pt1i, tri_lri, tri_lr, pt2s])
    flank_right = Polygon([corner_right, pt2, pt2i, tri_ti, tri_t, pt3s])
    flank_top = Polygon([corner_top, pt3, pt3i, tri_lli, tri_ll, pt1s])
    triangle_pre = Polygon([tri_lli, tri_ll, tri_lri, tri_lr, tri_ti, tri_t])
    triangle = triangle_pre.rotate(Point(0, 0), opening_phi)

    flank_lower = flank_lower.translate(triangle.points[1] - triangle_pre.points[1])
    flank_right = flank_right.translate(triangle.points[2] - triangle_pre.points[2])
    flank_top = flank_top.translate(triangle.points[0] - triangle_pre.points[0])

    line_left = Line.make_from_points(flank_top.points[0], flank_top.points[-1])
    line_right = Line.make_from_points(flank_right.points[0], flank_right.points[-1])
    line_horiz = Line.make_from_points(flank_lower.points[0], flank_lower.points[-1])
    corner_top = line_left.intersect(line_right)

    ## Infill
    line_left_inner = Line.make_from_points(flank_top.points[1], flank_top.points[2])
    flank_top._points[0] = corner_top
    flank_top._points[1] = line_left_inner.intersect(line_right)

    line_bottom_inner = Line.make_from_points(
        flank_lower.points[1], flank_lower.points[2]
    )
    flank_lower._points[0] = line_horiz.intersect(line_left)
    flank_lower._points[1] = line_bottom_inner.intersect(line_left)

    line_right_inner = Line.make_from_points(
        flank_right.points[1], flank_right.points[2]
    )
    flank_right._points[0] = line_right.intersect(line_horiz)
    flank_right._points[1] = line_right_inner.intersect(line_horiz)
    ## /Infill

    cell_bottom = MultiPolygon(
        [flank_lower, flank_right, flank_top, triangle]
    ).join_polygons()

    cell_left = cell_bottom.mirror_across_line(line_left)
    cell_right = cell_bottom.mirror_across_line(line_right)
    lower_half = MultiPolygon.FromMultipolygons([cell_bottom, cell_left, cell_right])

    line_horiz_top = Line(0, 0, 1, 0).translate(corner_top)
    upper_half = lower_half.mirror_across_line(line_horiz_top)

    hex_disconnected = MultiPolygon.FromMultipolygons([lower_half, upper_half])
    hex_connected = hex_disconnected.join_polygons()

    return hex_connected


if __name__ == "__main__":
    print("what")
    main()
