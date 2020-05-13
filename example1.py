from __future__ import annotations

import copy
from typing import List, Union, Dict, Any
import warnings

import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from matplotlib.path import Path  # type:ignore
import matplotlib.patches as patches  # type:ignore

from .main import Line, Vector, Point, Scalar, Param, GradientCarrier

# TODO: Opening fails, since triangle coords are not adapted

try:
    from martinez.polygon import Polygon as MPolygon  # type:ignore
    from martinez.contour import Contour as MContour  # type:ignore
    from martinez.point import Point as MPoint  # type:ignore
    from martinez.boolean import (  # type:ignore
        compute as Mcompute,
        OperationType as MOperationType,
    )
except ImportError:
    warnings.warn(
        "!IMPORTANT! Couldn't find module 'martinez' needed for joining"
        "polygons. Connecting polygons not possible"
    )


def what():
    for i in range(100):
        main()


def main():
    hexpoly, holepoly, holepts, _ = create_unit_cell()


def create_unit_cell(
    s=0.05, t=0.15, l=2.0, angle_cut=10, angle_opening=-10, c=0.05
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

    # fig, ax = plt.subplots()
    # ax.plot([corner_top.x], [corner_top.y], "o")

    # lims = (0, l.value, 10)
    # line_horiz.plot(ax=ax, lims=lims)
    # line_left.plot(ax=ax, lims=lims)
    # line_right.plot(ax=ax, lims=lims)

    # commented out because w float32 the 2e-8
    # assert np.allclose(corner_top.grads["lower_angles"][0], [0.0])
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

    # ax.plot(pt1.x, pt1.y, "o")

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

    # cut_lower.plot(ax=ax, lims=lims, label="lower")
    # cut_lower2.plot(ax=ax, lims=lims, label="lowe2")
    # cut_right.plot(ax=ax, lims=lims, label="right")
    # cut_right2.plot(ax=ax, lims=lims, label="right2")
    # cut_top.plot(ax=ax, lims=lims, label="top")
    # cut_top2.plot(ax=ax, lims=lims, label="top2")

    assert np.allclose(cut_lower.grads["t"], cut_lower2.grads["t"])

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

    # plt.plot(tri_lr.x, tri_lr.y, "o", label="tri_lr")
    # plt.plot(tri_ll.x, tri_ll.y, "o", label="tri_ll")
    # plt.plot(tri_t.x, tri_t.y, "o", label="tri_t")

    pt1i_ = cut_lower.intersect(cut_right2)
    pt1i = pt1i_ - ((pt1i_ - pt1) / (pt1i_ - pt1).norm()) * c

    pt2i_ = cut_right.intersect(cut_top2)
    pt2i = pt2i_ - ((pt2i_ - pt2) / (pt2i_ - pt2).norm()) * c

    pt3i_ = cut_top.intersect(cut_lower2)
    pt3i = pt3i_ - ((pt3i_ - pt3) / (pt3i_ - pt3).norm()) * c

    # plt.plot(pt1i.x, pt1i.y, "o", label="pt1i")
    # plt.plot(pt2i.x, pt2i.y, "o", label="pt2i")
    # plt.plot(pt3i.x, pt3i.y, "o", label="pt3i")

    # plt.legend()
    # plt.xlim((-0.05, 2.05))
    # plt.ylim((-0.05, 1.8))
    # plt.show()

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

    line_horiz_top = Line(0, 0).translate(corner_top)
    upper_half = lower_half.mirror_across_line(line_horiz_top)

    hex_disconnected = MultiPolygon.FromMultipolygons([lower_half, upper_half])
    hex_connected = hex_disconnected.join_polygons()

    return hex_connected.polygons[0]


class MartinezPointWithGrad(MPoint):
    def __init__(self, pt_grad_carrier):
        self.gradients = pt_grad_carrier.grads
        super().__init__(pt_grad_carrier.x, pt_grad_carrier.y)

    def to_differentiable(self):
        pt = Point(self.x, self.y)
        pt.gradients = self.gradients

        return pt


# TODO: Turn into gradient carrier, once it becomes necessary
class Polygon:
    def __init__(self, points: List[Point] = None, holes: List[List[Point]] = None):
        super().__init__()

        if points is None:
            points = []

        if holes is None:
            holes = []

        self._points: List[Point] = points
        self._holes: List[List[Point]] = holes
        self._bounding_box: Union[Dict[str, float], None] = None

    @property
    def points(poly: Polygon) -> List[Point]:
        return copy.copy(poly._points)

    @property
    def holes(poly: Polygon):
        return copy.copy(poly._holes)

    def as_numpy(poly: Polygon, close=False) -> np.ndarray:
        points = poly.points
        holes = poly.holes
        holes_out = []

        if close:
            if not points[-1].same_as(points[0]):
                points.append(points[0])

            for hole in holes:
                if not hole[-1].same_as(hole[0]):
                    hole.append(hole[0])

        for hole in holes:
            holes_out.append([point.as_numpy().flatten() for point in hole])

        return (
            np.array([point.as_numpy().flatten() for point in points]),
            np.array(holes_out),
        )

    @property
    def bounding_box(self) -> Dict[str, float]:
        if self._bounding_box is None:
            xs = [point.x for point in self._points]
            ys = [point.y for point in self._points]

            self._bounding_box = {
                "minx": np.min(xs),
                "maxx": np.max(xs),
                "miny": np.min(ys),
                "maxy": np.max(ys),
            }

        return copy.copy(self._bounding_box)

    def copy(self):
        return Polygon(copy.copy(self._points))

    def bounding_box_intersects(poly1: Polygon, poly2: Polygon, grow=0.00) -> bool:
        bb1 = poly1.bounding_box
        bb2 = poly2.bounding_box

        scale_fac = grow + 1

        if (
            False
            or bb1["minx"] > bb2["maxx"] * scale_fac
            or bb1["miny"] > bb2["maxy"] * scale_fac
            or bb1["maxx"] * scale_fac < bb2["minx"]
            or bb1["maxy"] * scale_fac < bb2["miny"]
        ):
            return False

        return True

    def draw(self: Polygon, **kwargs):
        mpoly = MultiPolygon([self])
        mpoly.draw(**kwargs)

    # TODO: Fix this class to respect hole points everywhere!
    def snap_to_poly(poly_slave: Polygon, poly_master: Polygon) -> Polygon:
        """
        For neighbouring polygons: Sets "approximately same" points to "exactly same"
        Needed for union to work properly
        """

        patched_slave_pts = []

        for ipt, pt in enumerate(poly_slave.points):
            close = poly_master.has_vertex(pt)

            if close is not False:
                patched_slave_pts.append(poly_master.points[close])
            else:
                patched_slave_pts.append(poly_slave.points[ipt])

        return Polygon(patched_slave_pts, poly_slave._holes)

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

    def translate(poly: Polygon, vec: Vector) -> Polygon:
        points = poly.points
        points_new = [point.translate(vec) for point in points]

        holes = poly._holes
        holes_new = [[point.translate(vec) for point in hole] for hole in holes]

        return Polygon(points_new, holes_new)

    def rotate(poly: Polygon, origin: Point, angle: Scalar) -> Polygon:
        points = poly.points
        points_new = [point.rotate(origin, angle) for point in points]

        return Polygon(points_new)

    def is_oriented_ccw(poly: Polygon) -> bool:
        edge_sum = 0
        points = poly.points

        for i, pt in enumerate(points[1:]):
            pt1 = points[i]
            pt2 = points[i + 1]

            edge = (pt2.x - pt1.x) * (pt2.y + pt1.y)
            edge_sum += edge

        if not points[0].same_as(points[-1]):
            edge_sum += (points[0].x - points[-1].x) * (points[0].y + points[-1].y)

        return edge_sum < 0

    def same_orientation_as(poly1: Polygon, poly2: Polygon) -> bool:
        orient1 = poly1.is_oriented_ccw()
        orient2 = poly2.is_oriented_ccw()

        return orient1 == orient2

    def flip_orientation(poly: Polygon) -> Polygon:
        return Polygon(poly._points[::-1])

    def num_verts_shared_with(poly1: Polygon, poly2: Polygon) -> int:
        if not poly1.bounding_box_intersects(poly2, grow=0.05):
            return 0

        nshared = 0

        for point in poly1.points:
            if poly2.has_vertex(point) is not False:
                nshared += 1

        return nshared

    def has_vertex(poly: Polygon, vert: Point) -> Union[int, bool]:
        shared_points = [
            idx for idx, point in enumerate(poly._points) if vert.same_as(point)
        ]

        if len(shared_points) == 0:
            return False

        return int(shared_points[0])

    def connect_to_poly(poly1: Polygon, poly2: Polygon) -> Polygon:
        # if not poly1.same_orientation_as(poly2):
        #    poly2 = poly2.flip_orientation()

        poly2 = poly2.snap_to_poly(poly1)

        ####
        mpoints = [MartinezPointWithGrad(pt) for pt in poly1.points]
        mholes = [
            MContour([MartinezPointWithGrad(pt) for pt in hole], [], False)
            for hole in poly1._holes
        ]
        mpoly1 = MPolygon([MContour(mpoints, [], True), *mholes])

        mpoints = [MartinezPointWithGrad(pt) for pt in poly2.points]
        mholes = [
            MContour([MartinezPointWithGrad(pt) for pt in hole], [], False)
            for hole in poly2._holes
        ]
        mpoly2 = MPolygon([MContour(mpoints, [], True), *mholes])
        ####

        unioned_poly = Mcompute(mpoly1, mpoly2, MOperationType.UNION)

        exterior: List[Point] = None
        interiors: List[List[Point]] = []

        for ic, contour in enumerate(unioned_poly.contours):
            pts = [pt.to_differentiable() for ipt, pt in enumerate(contour.points)]

            if contour.is_external:
                exterior = pts
            else:
                interiors.append(pts)

        return Polygon(exterior, interiors)


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

    def as_numpy(mpoly: MultiPolygon, close=False):
        return np.array([poly.as_numpy(close=close) for poly in mpoly.polygons])

    def draw(
        mpoly: MultiPolygon,
        ax=None,
        title=None,
        debug=False,
        draw_grads: List[str] = None,
    ):
        def get_poly_path(points2D):
            if not np.allclose(points2D[-1], points2D[0]):
                points2D.append(points2D[0])

            codes = []
            codes.append(Path.MOVETO)
            for i in range(len(points2D) - 2):
                codes.append(Path.LINETO)
            codes.append(Path.CLOSEPOLY)

            path = Path(points2D, codes)
            return path

        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        polygons = mpoly._polygons

        for ipoly, poly in enumerate(polygons):
            points2D = [(point.x, point.y) for point in poly._points]

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
                ax.text(
                    *np.mean(points2D, axis=0),
                    "poly {}\n({} verts)".format(ipoly, len(points2D)),
                    {"color": textcolor}
                )

            path_poly = get_poly_path(points2D)
            patch_poly = patches.PathPatch(path_poly, facecolor=facecolor, lw=0.25)
            ax.add_patch(patch_poly)

            for holepts in poly._holes:
                points2D = [(point.x, point.y) for point in holepts]

                path_hole = get_poly_path(points2D)
                patch_hole = patches.PathPatch(path_hole, facecolor="white", lw=0.25)
                ax.add_patch(patch_hole)

        if draw_grads is not None:
            for poly in polygons:
                for point in poly.points:
                    assert isinstance(point, Point)

                    for grad_name in draw_grads:
                        if grad_name not in point.grads:
                            continue

                        ax.plot(
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

    # TODO: Add to original repo
    def translate(mpoly: MultiPolygon, vec: Vector) -> MultiPolygon:
        polygons = mpoly.polygons
        polygons_new = [poly.translate(vec) for poly in polygons]

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
    # what()
    main()
