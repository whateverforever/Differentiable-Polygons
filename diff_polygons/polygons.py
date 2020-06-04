from __future__ import annotations

import copy
from typing import List, Union, Dict, Any
import warnings

import numpy as np  # type:ignore
import matplotlib.pyplot as plt  # type:ignore
from matplotlib.path import Path  # type:ignore
import matplotlib.patches as patches  # type:ignore

from .primitives import Line, Vector, Point, Scalar, Param, GradientCarrier

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


class MartinezPointWithGrad(MPoint):
    def __init__(self, pt_grad_carrier):
        self.gradients = pt_grad_carrier.grads
        super().__init__(pt_grad_carrier.x.value, pt_grad_carrier.y.value)

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

    # TODO: also mirror holes
    def mirror_across_line(poly: Polygon, line:Line) -> Polygon:
        points = poly.points
        points_new = [line.mirror_pt(point) for point in points]

        return Polygon(points_new)

    def translate(poly: Polygon, vec: Vector) -> Polygon:
        points = poly.points
        points_new = [point.translate(vec) for point in points]

        holes = poly._holes
        holes_new = [[point.translate(vec) for point in hole] for hole in holes]

        return Polygon(points_new, holes_new)

    # TODO: also rotate holes
    def rotate(poly: Polygon, origin: Point, angle: Scalar) -> Polygon:
        points = poly.points
        points_new = [point.rotate(origin, angle) for point in points]

        return Polygon(points_new)

    def is_oriented_ccw(poly: Polygon) -> bool:
        """
        Checks if exterior points are oriented counter-clockwise
        """
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

    # TODO: Profile, check when KDTree becomes better
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

    # TODO: Make shorter, make simpler
    def draw(
        mpoly: MultiPolygon,
        ax=None,
        title=None,
        debug=False,
        debug_holes=False,
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

        def highlight_and_number_points(points, ax, ipoly):
            red = np.random.uniform(0, 1)
            green = np.random.uniform(0, 1)
            blue = np.random.uniform(0, 1)

            facecolor = [red, green, blue, 0.5]
            textcolor = [red * 0.75, green * 0.75, blue * 0.75, 1.0]

            for i, pt in enumerate(points):
                ax.text(*pt, i, {"color": textcolor})
            ax.scatter(*zip(*points), c=[textcolor])
            ax.text(
                *np.mean(points, axis=0),
                "poly {}\n({} verts)".format(ipoly, len(points)),
                {"color": textcolor}
            )
            return facecolor

        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        polygons = mpoly._polygons

        for ipoly, poly in enumerate(polygons):
            points2D = [(point.x.value, point.y.value) for point in poly._points]

            facecolor = "orange"
            if debug:
                facecolor = highlight_and_number_points(points2D, ax, ipoly)

            path_poly = get_poly_path(points2D)
            patch_poly = patches.PathPatch(path_poly, facecolor=facecolor, lw=0.25)
            ax.add_patch(patch_poly)

            for ihole, holepts in enumerate(poly._holes):
                points2D = [(point.x.value, point.y.value) for point in holepts]

                facecolor = "white"
                if debug_holes:
                    facecolor = highlight_and_number_points(points2D, ax, ihole)

                path_hole = get_poly_path(points2D)
                patch_hole = patches.PathPatch(path_hole, facecolor=facecolor, lw=0.25)
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

    def translate(mpoly: MultiPolygon, vec: Vector) -> MultiPolygon:
        polygons = mpoly.polygons
        polygons_new = [poly.translate(vec) for poly in polygons]

        return MultiPolygon(polygons_new)

    # TODO: Make simpler, maker shorter
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
