from __future__ import annotations

import copy
from typing import List, Union
import warnings

import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from matplotlib.path import Path  # type:ignore
import matplotlib.patches as patches  # type:ignore

from main import Line, Vector, Point, Scalar, Param, GradientCarrier


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

    flank_lower = Polygon([corner_left, pt1, pt1i, tri_lr, pt2s])
    flank_right = Polygon([corner_right, pt2, pt2i, tri_t, pt3s])
    flank_top = Polygon([corner_top, pt3, pt3i, tri_ll, pt1s])
    triangle = Polygon([tri_ll, tri_lr, tri_t])

    cell_bottom = MultiPolygon([flank_lower, flank_right, flank_top, triangle])
    cell_left = cell_bottom.mirror_across_line(line_left)
    cell_right = cell_bottom.mirror_across_line(line_right)
    lower_half = MultiPolygon.FromMultipolygons([cell_bottom, cell_left, cell_right])

    line_horiz_top = Line(0, 0).translate(corner_top)
    upper_half = lower_half.mirror_across_line(line_horiz_top)

    hex_disconnected = MultiPolygon.FromMultipolygons([lower_half, upper_half])
    hex_connected = hex_disconnected.join_polygons()

    hex_disconnected.draw(draw_grads=["l"], debug=True)
    hex_connected.draw(draw_grads=["l"])


# TODO: Turn into gradient carrier, once it becomes necessary
class Polygon:
    def __init__(self, points: List[Point] = None):
        super().__init__()

        if points is None:
            points = []

        self._points: List[Point] = points

    @property
    def points(poly: Polygon) -> List[Point]:
        return copy.copy(poly._points)

    def copy(self):
        return Polygon(copy.deepcopy(self._points))

    def draw(self: Polygon, **kwargs):
        mpoly = MultiPolygon([self])
        mpoly.draw(**kwargs)

    def add_point(poly: Polygon, point: Point) -> Polygon:
        if not isinstance(point, Point):
            raise TypeError("Can't append anything else than a 'Point' to 'Polygon'")

        points = poly.points
        points.append(point)

        return Polygon(points)

    def mirror_across_line(poly: Polygon, line: Line) -> Polygon:
        points = poly.points
        points_new = [point.mirror_across_line(line) for point in points]

        return Polygon(points_new)

    def is_oriented_ccw(poly: Polygon) -> bool:
        edge_sum = 0
        points = poly.points

        for i, pt in enumerate(points):
            if i == 0:
                continue

            pt1 = points[i - 1]
            pt2 = points[i]

            edge = (pt2.x - pt1.x) * (pt2.y + pt1.y)
            edge_sum += edge

        if not np.allclose(points[0].as_numpy(), points[-1].as_numpy()):
            edge_sum += (points[0].x - points[-1].x) * (points[0].y + points[-1].y)

        return edge_sum < 0

    def same_orientation_as(poly1: Polygon, poly2: Polygon) -> bool:
        orient1 = poly1.is_oriented_ccw()
        orient2 = poly2.is_oriented_ccw()

        return orient1 == orient2

    def flip_orientation(poly: Polygon) -> Polygon:
        new_poly = poly.copy()
        new_poly._points = list(reversed(poly._points))

        return new_poly

    def num_verts_shared_with(poly1: Polygon, poly2: Polygon) -> int:
        nshared = 0

        for point in poly1.points:
            if poly2.contains_vert(point) is not False:
                nshared += 1

        return nshared

    def contains_vert(poly: Polygon, vert: Point) -> Union[int, bool]:
        shared_points = [
            idx
            for idx, point in enumerate(poly._points)
            if np.allclose(vert.as_numpy(), point.as_numpy())
        ]

        if len(shared_points) == 0:
            return False

        return int(shared_points[0])

    def connect_to_poly(poly1: Polygon, poly2: Polygon):
        poly1 = poly1.copy()
        poly2 = poly2.copy()

        if not poly1.same_orientation_as(poly2):
            poly2 = poly2.flip_orientation()

        in_polys = [poly1, poly2]

        def vert_exists_in_other(poly_idx, vert_idx):
            other_poly_idx = 1 if poly_idx == 0 else 0

            other_poly = in_polys[other_poly_idx]
            vert = in_polys[poly_idx].points[vert_idx]

            return other_poly.contains_vert(vert)

        start_vert = -1

        for i, _ in enumerate(poly1.points):
            if vert_exists_in_other(0, i) is False:
                start_vert = i
                break

        curr_vert_idx: int = start_vert
        curr_poly_idx: int = 0

        out_poly: Polygon = Polygon([in_polys[curr_poly_idx].points[curr_vert_idx]])

        while True:
            if curr_vert_idx + 1 < len(in_polys[curr_poly_idx].points):
                curr_vert_idx += 1
            else:
                curr_vert_idx = 0

            if (
                out_poly.contains_vert(in_polys[curr_poly_idx].points[curr_vert_idx])
                is not False
            ):
                return out_poly

            out_poly = out_poly.add_point(in_polys[curr_poly_idx].points[curr_vert_idx])

            idx_in_other = vert_exists_in_other(curr_poly_idx, curr_vert_idx)
            if idx_in_other is False:
                pass
            else:
                curr_poly_idx = 1 if curr_poly_idx == 0 else 0
                curr_vert_idx = idx_in_other


# TODO: Turn into gradient carrier, once it becomes necessary
class MultiPolygon:
    def __init__(self, polygons: List[Polygon] = None):
        super().__init__()

        if polygons is None:
            polygons = []

        self._polygons = polygons

    @staticmethod
    def FromMultipolygons(mpolys: List[MultiPolygon]) -> MultiPolygon:
        all_polygons = []
        for mpoly in mpolys:
            all_polygons.extend(mpoly.polygons)

        return MultiPolygon(all_polygons)

    @property
    def polygons(self):
        return copy.copy(self._polygons)

    def draw(
        mpoly: MultiPolygon,
        ax=None,
        title=None,
        debug=False,
        draw_grads: List[str] = None,
    ):
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        polygons = mpoly._polygons

        for poly in polygons:
            n = len(poly._points)
            points2D = [(point.x, point.y) for point in poly._points]

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
                for point in poly.points:
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

    def add_polygons(mpoly: MultiPolygon, polys: List[Polygon]) -> MultiPolygon:
        new_polys = mpoly.polygons
        new_polys.extend(polys)

        return MultiPolygon(new_polys)

    def add_polygon(mpoly: MultiPolygon, poly: Polygon) -> MultiPolygon:
        if not isinstance(poly, Polygon):
            raise TypeError(
                "Can't append anything else than a 'Polygon' to 'MultiPolygon'"
            )

        polys = mpoly.polygons
        polys.append(poly)

        return MultiPolygon(polys)

    def mirror_across_line(mpoly: MultiPolygon, line: Line) -> MultiPolygon:
        polygons = mpoly.polygons
        polygons_new = [poly.mirror_across_line(line) for poly in polygons]

        return MultiPolygon(polygons_new)

    def join_polygons(multipoly: MultiPolygon) -> MultiPolygon:
        """
        Takes a list of polygons, some of which might be connectable. It then recursively
        tries to match pairs of two together. As such, it can also resolve connections
        of more than two polygons.
        """
        visited_pairs = []
        out_multipoly = MultiPolygon()
        lone_polys = multipoly.polygons
        for ip1, poly1 in enumerate(multipoly._polygons):
            for ip2, poly2 in enumerate(multipoly._polygons):
                pair = set([ip1, ip2])

                if ip1 == ip2 or pair in visited_pairs:
                    continue
                visited_pairs.append(pair)

                poly1_taken = lone_polys[ip1] is None
                poly2_taken = lone_polys[ip2] is None

                if poly1_taken or poly2_taken:
                    continue

                if poly1.num_verts_shared_with(poly2) >= 2:
                    joined = poly1.connect_to_poly(poly2)
                    out_multipoly = out_multipoly.add_polygon(joined)

                    lone_polys[ip1] = None
                    lone_polys[ip2] = None

        lone_polys = [poly for poly in lone_polys if poly is not None]
        out_multipoly = out_multipoly.add_polygons(lone_polys)

        # If we have multiple polygons connected together, one pass alone won't be
        # enough to connect all of them together --> recurse
        visited_pairs = []
        for ip1, poly1 in enumerate(out_multipoly.polygons):
            for ip2, poly2 in enumerate(out_multipoly.polygons):
                pair = set([ip1, ip2])

                if ip1 == ip2 or pair in visited_pairs:
                    continue
                visited_pairs.append(pair)

                # There is at least one edge leftover where two polygons could be
                # connected
                if poly1.num_verts_shared_with(poly2) >= 2:
                    return out_multipoly.join_polygons()

        return out_multipoly


if __name__ == "__main__":
    main()
