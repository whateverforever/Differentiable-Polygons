from __future__ import annotations

import copy
import typing as ty
import warnings
import numpy as np  # type: ignore

from numbers import Number


class GradientCarrier:
    def __init__(self):
        self.gradients = {}

    @property
    def grads(self):
        return copy.copy(self.gradients)

    # TODO: Needs to be other way round. Ground truth being an array of values,
    # and accessors like .x, .y, .m, .b etc should be @propertys
    # TODO: Also supply gradient accessor that respects the ordering of the
    # parameters. Maybe .jac() ?
    @property
    def properties(self):
        raise NotImplementedError(
            "{} doesn't have `properties` implemented yet".format(self.__class__)
        )

    def with_grads(self, grads):
        self_copy = copy.copy(self)
        self_copy.gradients = grads
        return self_copy

    def with_grads_from_previous(self, inputs, local_grads):
        """
        Takes the inputs into the function and the local gradients that matter
        inside of the function, calculates the new gradients wrt. to the original parameters
        (by utilizing the local gradients like dself_dprevpt or dself_dparam).

        inputs: GradientCarriers that that get transformed in the operation. Their
                gradients will be used for chaining new ones onto.

        local_grads: Gradients of this particular operation wrt the inputs. Must use
                     same names.

        returns: an instance of this particular GradientCarrier with the gradient set
        """
        self_copy = copy.copy(self)
        assert self_copy.gradients == {}

        self_copy.gradients = update_grads(inputs, local_grads)
        return self_copy


class Scalar(GradientCarrier):
    def __init__(self, value):
        super().__init__()

        # TODO: Change, inelegant. Lookup how coercion is usually done
        if isinstance(value, Scalar):
            self.value = value.value
            self.gradients = value.gradients
            return

        self.value = value
        self.name = None

    @property
    def properties(self):
        return [self.value]

    def __repr__(self):
        return "Scalar({:.4f})".format(self.value)

    def __add__(scal1: Scalar, scal2: Scalar) -> Scalar:
        scal1 = Scalar(scal1)
        scal2 = Scalar(scal2)

        val_new = scal1.value + scal2.value

        inputs = {"scal1": scal1, "scal2": scal2}
        grads = {"scal1": [[1]], "scal2": [[1]]}

        return Scalar(val_new).with_grads_from_previous(inputs, grads)

    def __neg__(scalar_old: Scalar):
        val_new = -scalar_old.value

        inputs = {"scalar_old": scalar_old}
        grads = {"scalar_old": [[-1]]}

        return Scalar(val_new).with_grads_from_previous(inputs, grads)

    # TODO: Add __mul__
    def __rmul__(self, scalar2):
        scalar1 = copy.deepcopy(self)

        scalar2 = Scalar(scalar2)
        inputs = {"scalar1": scalar1, "scalar2": scalar2}
        local_grads = {"scalar2": [[scalar1.value]], "scalar1": [[scalar2.value]]}

        return Scalar(scalar1.value * scalar2.value).with_grads_from_previous(
            inputs, local_grads
        )

    @staticmethod
    def Param(name, value):
        grads = {name: [[1.0]]}
        # TODO: This is a bit too much all over the place. How to harmonize?
        if isinstance(value, Scalar):
            grads.update(value.grads)

        param = Scalar(value).with_grads(grads)
        param.name = name

        return param


class Point(GradientCarrier):
    def __init__(self, x, y):
        super().__init__()

        x = Scalar(x)
        y = Scalar(y)

        inputs = {"_x": x, "_y": y}
        local_grads = {"_x": [[1], [0]], "_y": [[0], [1]]}

        self.x = x.value
        self.y = y.value
        self.gradients = update_grads(inputs, local_grads)

    @property
    def properties(self):
        return [self.x, self.y]

    def as_numpy(self):
        return np.array([[self.x], [self.y]])

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x, self.y)

    def __truediv__(pt: Point, s: ty.Union[Scalar, Number]) -> Point:
        if isinstance(s, Scalar):
            new_x = pt.x / s.value
            new_y = pt.y / s.value

            inputs = {"pt": pt, "s": s}
            grads = {
                "pt": [[1 / s.value, 0], [0, 1 / s.value]],
                "s": [[-pt.x / (s.value ** 2)], [-pt.y / (s.value ** 2)]],
            }

            return Point(new_x, new_y).with_grads_from_previous(inputs, grads)

        raise NotImplementedError("__truediv__ not yet impl for normal numbers")

    def __mul__(pt: Point, other: ty.Union[Point, Scalar, Number]) -> Point:
        if isinstance(other, Scalar):
            new_x = pt.x * other.value
            new_y = pt.y * other.value

            inputs = {"pt": pt, "scalar": other}
            grads = {
                "pt": [[other.value, 0], [0, other.value]],
                "scalar": [[pt.x], [pt.y]],
            }

            return Point(new_x, new_y).with_grads_from_previous(inputs, grads)

        raise NotImplementedError(
            "__mul__ not yet implemented for {}".format(type(other))
        )

    def __sub__(pt1: Point, pt2: Point) -> Point:
        x3 = pt1.x - pt2.x
        y3 = pt1.y - pt2.y

        inputs = {"pt1": pt1, "pt2": pt2}
        local_grads = {}
        local_grads["pt1"] = [[1, 0], [0, 1]]
        local_grads["pt2"] = [[-1, 0], [0, -1]]

        return Point(x3, y3).with_grads_from_previous(inputs, local_grads)

    def __add__(pt1: Point, pt2: Point) -> Point:
        x3 = pt1.x + pt2.x
        y3 = pt1.y + pt2.y

        inputs = {"pt1": pt1, "pt2": pt2}
        local_grads = {}
        local_grads["pt1"] = [[1, 0], [0, 1]]
        local_grads["pt2"] = [[1, 0], [0, 1]]

        return Point(x3, y3).with_grads_from_previous(inputs, local_grads)

    def __neg__(old_pt):
        x2 = -old_pt.x
        y2 = -old_pt.y

        inputs = {"old_pt": old_pt}
        grads = {"old_pt": [[-1, 0], [0, -1]]}

        return Point(x2, y2).with_grads_from_previous(inputs, grads)

    def same_as(pt1: Point, pt2: Point, eps=1e-4) -> bool:
        a = pt1.x - pt2.x
        b = pt1.y - pt2.y

        return a * a + b * b <= eps * eps

    def mirror_across_line(pt: Point, line: Line) -> Point:
        x = pt.x
        y = pt.y

        m = line.m
        b = line.b

        u = ((1 - m ** 2) * x + 2 * m * y - 2 * m * b) / (m ** 2 + 1)
        v = ((m ** 2 - 1) * y + 2 * m * x + 2 * b) / (m ** 2 + 1)

        du_dpt = [[-1 + 2 / (1 + m ** 2), (2 * m) / (1 + m ** 2)]]
        dv_dpt = [[(2 * m) / (1 + m ** 2), 1 - 2 / (1 + m ** 2)]]
        dself_dpt = [*du_dpt, *dv_dpt]

        du_dline = [
            [
                (2 * (b * (-1 + m ** 2) + y - m * (2 * x + m * y))) / (1 + m ** 2) ** 2,
                -((2 * m) / (1 + m ** 2)),
            ]
        ]
        dv_dline = [
            [
                (2 * x - 2 * m * (2 * b + m * x - 2 * y)) / (1 + m ** 2) ** 2,
                2 / (1 + m ** 2),
            ]
        ]
        dself_dline = [*du_dline, *dv_dline]
        inputs = {"pt": pt, "line": line}
        grads = {"pt": dself_dpt, "line": dself_dline}

        return Point(u, v).with_grads_from_previous(inputs, grads)

    def translate(pt: Point, vec: Point) -> Point:
        x2 = pt.x + vec.x
        y2 = pt.y + vec.y

        inputs = {"pt": pt, "vec": vec}
        _grads = {}
        _grads["pt"] = np.array([[1, 0], [0, 1]])
        _grads["vec"] = np.array([[1, 0], [0, 1]])

        pt2 = Point(x2, y2).with_grads_from_previous(inputs, _grads)
        return pt2

    def norm(pt: Point):
        eps = 1e-13

        l2_norm = np.sqrt(eps + pt.x ** 2 + pt.y ** 2)

        grad_pt = [[pt.x / l2_norm, pt.y / l2_norm]]

        inputs = {"pt": pt}
        _grads = {}
        _grads["pt"] = grad_pt

        return Scalar(l2_norm).with_grads_from_previous(inputs, _grads)

    def rotate(pt: Point, origin: Point, angle_rad: Scalar) -> Point:
        # TODO: Same for points, coercion
        angle_rad = Scalar(angle_rad)

        x1 = pt.x
        y1 = pt.y

        ox = origin.x
        oy = origin.y

        angle = angle_rad.value

        x2 = (x1 - ox) * np.cos(angle) - (y1 - oy) * np.sin(angle) + ox
        y2 = (x1 - ox) * np.sin(angle) + (y1 - oy) * np.cos(angle) + oy

        d_dpt = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        d_dorigin = [
            [-np.cos(angle) + 1, np.sin(angle)],
            [-np.sin(angle), -np.cos(angle) + 1],
        ]
        d_dangle = [
            [(oy - y1) * np.cos(angle) + (ox - x1) * np.sin(angle)],
            [(-ox + x1) * np.cos(angle) + (oy - y1) * np.sin(angle)],
        ]

        inputs = {"pt": pt, "origin": origin, "angle": angle_rad}

        _grads = {}
        _grads["pt"] = d_dpt
        _grads["origin"] = d_dorigin
        _grads["angle"] = d_dangle

        print("local grads", _grads["angle"])

        pt2 = Point(x2, y2).with_grads_from_previous(inputs, _grads)

        return pt2


Vector = Point
Param = Scalar.Param


def update_grads(
    inputs: ty.Dict[str, GradientCarrier],
    local_grads: ty.Dict[str, ty.Union[ty.List[Number], np.ndarray]],
):
    incoming_parameters = []
    for input_name, input_obj in inputs.items():
        if not isinstance(input_obj, GradientCarrier):
            warnings.warn(
                "Got non-GradientCarrier obj of type {} as input. Won't have gradient"
                " information, so please remove or replace by Scalar or Vector: {}".format(
                    type(input_obj), input_obj
                )
            )
            continue

        incoming_parameters.extend(
            [
                grad_name
                for grad_name in input_obj.gradients.keys()
                if grad_name not in incoming_parameters
            ]
        )

    # Parameters that previous operations don't know anything about
    # I.e. maybe we did translations on `l` before, and now a rotation
    # on new parameter `theta`
    local_params = list(local_grads.keys())
    input_params = inputs.keys()
    own_parameters = [
        param
        for param in local_params
        if param not in input_params and param not in incoming_parameters
    ]

    out_grads = {}
    inputs_items = inputs.items()
    shapeA = len(local_grads[local_params[0]])
    for param in incoming_parameters + own_parameters:
        grads = np.zeros((shapeA, 1))

        # If we have inputs that depended on parameters
        for input_name, input_obj in inputs_items:
            # TODO: Same as above
            if not isinstance(input_obj, GradientCarrier):
                continue
            # If one of the inputs doesn't depend on the parameter, we simply
            # ignore it. No gradient information in there!
            if param in input_obj.gradients:
                dself_dinput = local_grads[input_name]
                dinput_dparam = input_obj.gradients[param]

                grads += np.matmul(dself_dinput, dinput_dparam)

        out_grads[param] = grads
    return out_grads


class Line(GradientCarrier):
    """ Simple class representing a line, used to construct the unit cell """

    # TODO: Replace by better representation with no singularities
    def __init__(self, m, b):
        super().__init__()

        m = Scalar(m)
        b = Scalar(b)

        inputs = {"m": m, "b": b}
        local_grads = {"m": [[1], [0]], "b": [[0], [1]]}

        self.m = m.value
        self.b = b.value
        self.gradients = update_grads(inputs, local_grads)

    @property
    def properties(self):
        return [self.m, self.b]

    def __repr__(self):
        return f"Line(m={self.m:.4f}, b={self.b:.4f})"

    # TODO: Remove, make into normal method
    @staticmethod
    def make_from_points(p1: Point, p2: Point):
        return Line.make_from_points_({"p1": p1, "p2": p2})

    @staticmethod
    def make_from_points_(inputs: ty.Dict[str, Point]):
        """ Returns line that goes through p1 and p2 """
        p1 = inputs["p1"]
        p2 = inputs["p2"]

        x1 = p1.x
        y1 = p1.y

        x2 = p2.x
        y2 = p2.y

        b = (y1 * x2 - y2 * x1) / (x2 - x1)
        m = (y2 - b) / x2

        dm_dp1 = [[(-y1 + y2) / (x1 - x2) ** 2, 1 / (x1 - x2)]]
        dm_dp2 = [[(y1 - y2) / (x1 - x2) ** 2, 1 / (-x1 + x2)]]

        db_dp1 = [[(x2 * (y1 - y2)) / (x1 - x2) ** 2, x2 / (-x1 + x2)]]
        db_dp2 = [[(x1 * (-y1 + y2)) / (x1 - x2) ** 2, x1 / (x1 - x2)]]

        local_grads = {}
        local_grads["p1"] = d_dpt1 = np.vstack([dm_dp1, db_dp1])
        local_grads["p2"] = d_dpt2 = np.vstack([dm_dp2, db_dp2])

        new_line = Line(m, b).with_grads_from_previous(inputs, local_grads)
        return new_line

    def translate(a_line: Line, vec: Vector) -> Line:
        line_old = copy.deepcopy(a_line)

        m = line_old.m
        b = line_old.b + vec.y - line_old.m * vec.x

        inputs = {"vec": vec, "line": line_old}
        grads = {}
        grads["vec"] = [[0, 0], [-line_old.m, 1]]
        grads["line"] = [[1, 0], [-vec.x, 1]]

        new_line = Line(m, b).with_grads_from_previous(inputs, grads)

        return new_line

    def rotate_ccw(line1: Line, angle_rad: Scalar, pivot: Point = None) -> Line:
        if pivot is None:
            pivot = Point(0, 0)

        if not isinstance(angle_rad, Scalar):
            angle_rad = Scalar(angle_rad)

        line_centered = line1.translate(-pivot)

        m = line_centered.m
        b = line_centered.b

        m2 = np.tan(np.arctan(m) + angle_rad.value)
        b2 = b

        sec = lambda x: 1 / np.cos(x)
        d_dangle = np.radians(sec(angle_rad.value + np.arctan(m)) ** 2)

        local_grads = {}
        local_grads["angle_rad"] = [
            [d_dangle],
            [0],
        ]
        local_grads["line_centered"] = [
            [d_dangle / (1 + m ** 2), 0],
            [0, 1],
        ]

        inputs = {"angle_rad": angle_rad, "line_centered": line_centered}

        rotated_line = Line(m2, b2).with_grads_from_previous(inputs, local_grads)

        new_line = rotated_line.translate(pivot)

        return new_line

    def intersect(line_1: Line, line_2: Line) -> Point:
        m1 = line_1.m
        b1 = line_1.b

        m2 = line_2.m
        b2 = line_2.b

        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

        # dx_dline meaning [dx_dm, dx_db]
        dx_dline1 = [[(b1 - b2) / (m1 - m2) ** 2, 1 / (-m1 + m2)]]
        dx_dline2 = [[(-b1 + b2) / (m1 - m2) ** 2, 1 / (m1 - m2)]]

        dy_dline1 = [[((b1 - b2) * m2) / (m1 - m2) ** 2, -(m2 / (m1 - m2))]]
        dy_dline2 = [[((-b1 + b2) * m1) / (m1 - m2) ** 2, m1 / (m1 - m2)]]

        inputs = {"line_1": line_1, "line_2": line_2}
        local_grads = {
            "line_1": np.vstack([dx_dline1, dy_dline1]),
            "line_2": np.vstack([dx_dline2, dy_dline2]),
        }

        return Point(x, y).with_grads_from_previous(inputs, local_grads)

    def plot(self, ax=None, lims=(-20, 20, 10), label=None):
        import matplotlib.pyplot as plt  # type: ignore

        if ax is None:
            ax = plt

        x = np.linspace(*lims)
        y = self.m * x + self.b

        ax.plot(x, y, label=label)
        ax.axis("equal")
