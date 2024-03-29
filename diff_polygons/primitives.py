from __future__ import annotations

import copy
import typing as ty
import warnings
from math import isclose, sqrt
from numbers import Number

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from math import sin as _sin, cos as _cos, atan as _arctan, tan as _tan

GRADS_ENABLED = True

# TODO: Split this? Make only scalar GradientCarrier and Point, Line, etc.
# are somehow groups of "gradiented" scalars?
class GradientCarrier:
    @property
    def gradients(self):
        if not GRADS_ENABLED:
            return {}

        if isinstance(self, Scalar):
            return self._basegradients

        return combine_gradients(self._params)

    @gradients.setter
    def gradients(self, new_grads):
        if not GRADS_ENABLED:
            return

        if isinstance(self, Scalar):
            self._basegradients = new_grads
            return

        # For combined Carriers like Line or Point a gradient update
        # means updating its component Scalar Parameters
        for iparam, param in enumerate(self._params):
            scalar_grads = {}
            for grad_name, grad_values in new_grads.items():
                scalar_grads[grad_name] = [grad_values[iparam]]

            self._params[iparam].gradients = scalar_grads

    @property
    def grads(self):
        return {**self.gradients}

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
        if not GRADS_ENABLED:
            return self

        # assert self.gradients == {}

        self.gradients = update_grads(inputs, local_grads)
        return self


class Scalar(GradientCarrier):
    def __init__(self, value):
        basegrads = {}

        if isinstance(value, Scalar):
            basegrads = value.grads
            value = value.value

        self._basegradients = basegrads
        self._params = [value]
        self.name = None

    @property
    def value(self):
        return self._params[0]

    def with_grads(self, grads):
        self_copy = Scalar(self)
        self_copy._basegradients = grads
        return self_copy
    
    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return "Scalar({:.4f})".format(self.value)

    def __eq__(self: Scalar, other: Union[Scalar, float, int]) -> bool:
        if not isinstance(other, Scalar):
            return np.isclose(self.value, other)

        return self.value == other.value

    def __lt__(scal1: Scalar, other: Union[Scalar, float, int]) -> bool:
        if not isinstance(other, Scalar):
            return scal1.value < other

        return scal1.value < other.value
    
    def __gt__(scal1: Scalar, other: Union[Scalar, float, int]) -> bool:
        return other < scal1

    def __le__(scal1: Scalar, other: Union[Scalar, float, int]) -> bool:
        return (scal1 < other) or (scal1 == other)
    
    def __ge__(scal1: Scalar, other: Union[Scalar, Number]) -> bool:
        return other <= scal1

    def __rpow__(power: Scalar, base: Any) -> bool:
        return Scalar(base) ** power

    def __pow__(base: Scalar, power) -> bool:
        power = Scalar(power)

        val_new = base.value ** power.value

        # The log is very brittle. But it's evaluation is actually only needed when
        # the exponent contains gradients, hence we don't evaluate it if no need be
        d_dpower = -1
        if power.grads != {}:
            d_dpower = val_new * np.log(base.value)

        inputs = {"base": base, "power": power}
        grads = {
            "base": [[power.value * base.value ** (power.value - 1)]],
            "power": [[d_dpower]],
        }

        return Scalar(val_new).with_grads_from_previous(inputs, grads)

    def __radd__(scal1: Scalar, scal2: Any) -> Scalar:
        return scal1 + scal2

    def __add__(scal1: Scalar, scal2: Scalar) -> Scalar:
        scal1 = Scalar(scal1)
        scal2 = Scalar(scal2)

        val_new = scal1.value + scal2.value

        inputs = {"scal1": scal1, "scal2": scal2}
        grads = {"scal1": [[1]], "scal2": [[1]]}

        return Scalar(val_new).with_grads_from_previous(inputs, grads)

    def __rsub__(scal1: Scalar, scal2: Any) -> Scalar:
        return -scal1 + scal2

    def __sub__(scal1: Scalar, scal2: Scalar) -> Scalar:
        scal1 = Scalar(scal1)
        scal2 = -Scalar(scal2)

        return scal1 + scal2

    def __neg__(scalar_old: Scalar):
        val_new = -scalar_old.value

        inputs = {"scalar_old": scalar_old}
        grads = {"scalar_old": [[-1]]}

        return Scalar(val_new).with_grads_from_previous(inputs, grads)

    def __truediv__(scal1: Scalar, scal2: Scalar) -> Scalar:
        scal1 = Scalar(scal1)
        scal2 = Scalar(scal2)

        inputs = {"scal1": scal1, "scal2": scal2}
        local_grads = {
            "scal1": [[1 / scal2.value]],
            "scal2": [[-scal1.value / (scal2.value ** 2)]],
        }

        return Scalar(scal1.value / scal2.value).with_grads_from_previous(
            inputs, local_grads
        )

    def __rmul__(scal1: Scalar, scal2: Any) -> Scalar:
        return scal1 * scal2

    def __mul__(scal1: Scalar, scal2: Scalar) -> Scalar:
        scalar1 = Scalar(scal1)
        scalar2 = Scalar(scal2)

        inputs = {"scalar1": scalar1, "scalar2": scalar2}
        local_grads = {"scalar1": [[scalar2.value]], "scalar2": [[scalar1.value]]}

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
        self._params = [Scalar(x), Scalar(y)]

    @property
    def x(self):
        return self._params[0]

    @property
    def y(self):
        return self._params[1]

    def as_numpy(self):
        return np.reshape([prop.value for prop in self._params], (-1, 1))

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x.value, self.y.value)

    def __truediv__(pt: Point, s: ty.Union[Scalar, Number]) -> Point:
        s = Scalar(s)
        new_x = pt.x / s
        new_y = pt.y / s

        return Point(new_x, new_y)

    def __eq__(pt1: Point, pt2: ty.Any) -> bool:
        if not isinstance(pt2, Point):
            raise TypeError(
                "Can only compare Point to Point, not {}".format(pt2.__class__)
            )

        return pt1.same_as(pt2)

    def __mul__(pt: Point, other: ty.Union[Point, Scalar, Number]) -> Point:
        if isinstance(other, Scalar) or isinstance(other, Number):
            other = Scalar(other)

            new_x = pt.x * other
            new_y = pt.y * other

            return Point(new_x, new_y)

        raise NotImplementedError("__mul__ not implemented for {}".format(type(other)))

    def __sub__(pt1: Point, pt2: Point) -> Point:
        # reuse __add__ and __neg__
        return pt1 + -pt2

    def __add__(pt1: Point, pt2: Point) -> Point:
        x3 = pt1.x.value + pt2.x.value
        y3 = pt1.y.value + pt2.y.value

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
        a = pt1.x.value - pt2.x.value
        b = pt1.y.value - pt2.y.value

        return a * a + b * b <= eps * eps

    def mirror_across_line(pt: Point, line: Line) -> Point:
        return line.mirror_pt(pt)

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
        l2_norm = (eps + pt.x ** 2 + pt.y ** 2) ** 0.5

        return Scalar(l2_norm)

    def rotate(pt: Point, origin: Point, angle_rad: Scalar) -> Point:
        angle_rad = Scalar(angle_rad)

        x1 = pt.x
        y1 = pt.y

        ox = origin.x
        oy = origin.y

        angle = angle_rad.value

        x2 = (x1 - ox) * cos(angle_rad) - (y1 - oy) * sin(angle_rad) + ox
        y2 = (x1 - ox) * sin(angle_rad) + (y1 - oy) * cos(angle_rad) + oy

        return Point(x2, y2)
    
    def project_onto(pt: Point, vec: Vector) -> Scalar:
        vec_len = vec.norm().value
        if abs(vec_len - 1) > 0.01:
            warnings.warn("Vector you're trying to project onto doesn't have unit length: {:.3f}".format(vec_len))
        
        return pt.x * vec.x + pt.y * vec.y

Vector = Point
Param = Scalar.Param


class Line(GradientCarrier):
    """
    An infinite line. Parametrized by origin and direction vector. (4 parameters)
    """

    def __init__(self, ox, oy, dx, dy):
        super().__init__()

        ox = Scalar(ox)
        oy = Scalar(oy)

        dx = Scalar(dx)
        dy = Scalar(dy)

        length = dx.value ** 2 + dy.value ** 2
        if not isclose(length, 1.0, rel_tol=1e-3):
            warnings.warn(
                "Direction vector {:.3f},{:.3f} doesn't have unit length."
                "Fixed its size for you. Please do yourself next time.".format(
                    dx.value, dy.value
                )
            )
            dx /= length
            dy /= length

        self._params = [ox, oy, dx, dy]

    @staticmethod
    def make_from_points(pt1: Point, pt2: Point) -> Line:
        direction = (pt2 - pt1) / Scalar(np.linalg.norm((pt2 - pt1).as_numpy().ravel()))
        return Line(pt1.x, pt1.y, direction.x, direction.y)

    @property
    def ox(self):
        return self._params[0]

    @property
    def oy(self):
        return self._params[1]

    @property
    def dx(self):
        return self._params[2]

    @property
    def dy(self):
        return self._params[3]
    
    @property
    def origin(self):
        return Point(self.ox, self.oy)
    
    @property
    def direction(self):
        return Vector(self.dx, self.dy)

    def eval_at(line: Line, s) -> Point:
        s = Scalar(s)

        x = line.ox + s * line.dx
        y = line.oy + s * line.dy

        return Point(x, y)

    def intersect(line1: Line, Line: Line) -> Point:
        o2x = line1.ox
        o1x = Line.ox

        o2y = line1.oy
        o1y = Line.oy

        d2x = line1.dx
        d1x = Line.dx

        d2y = line1.dy
        d1y = Line.dy

        s1 = ((o2y - o1y) * d1x - (o2x - o1x) * d1y) / (d2x * d1y - d2y * d1x)

        return line1.eval_at(s1)
    
    def translate(a_line: Line, vec: Vector) -> Line:
        origin = Point(a_line.ox, a_line.oy)
        origin = origin.translate(vec)

        return Line(origin.x, origin.y, a_line.dx, a_line.dy)

    def rotate_ccw(line1: Line, angle_rad: Scalar, pivot: Point = None) -> Line:
        if pivot is None:
            pivot = Point(0,0)

        direction = Vector(line1.dx, line1.dy)
        direction = direction.rotate(Point(0,0), angle_rad)

        orig = Point(line1.ox, line1.oy)
        orig = orig.rotate(pivot, angle_rad)

        return Line(orig.x, orig.y, direction.x, direction.y)
    
    def mirror_pt(line: Line, pt: Point) -> Point:
        pt_relative = pt - line.origin
        dist = pt_relative.project_onto(line.direction)

        pt_on_ray = line.direction * dist
        vec_perp = pt_on_ray - pt_relative
        
        pt_mirrored = pt_on_ray + vec_perp + line.origin

        return pt_mirrored

    def plot(self, ax=plt):
        s = np.arange(0, 10)
        pts = [self.eval_at(si) for si in s]
        xys = [(pt.x, pt.y) for pt in pts]

        ax.plot(*zip(*xys))

# Utilities

def update_grads(
    inputs: ty.Dict[str, GradientCarrier],
    local_grads: ty.Dict[str, ty.Union[ty.List[Number], np.ndarray]],
):
    inputs_items = inputs.items()
    incoming_parameters = []
    for input_name, input_obj in inputs_items:
        incoming_parameters.extend(
            [
                grad_name
                for grad_name in input_obj.gradients.keys()
                if grad_name not in incoming_parameters
            ]
        )

    some_grad_name = list(local_grads.keys())[0]
    some_grad = local_grads[some_grad_name]
    grad_shape = [len(some_grad), 1]

    out_grads = {}
    for param in incoming_parameters:
        grads = np.zeros(grad_shape)
        for input_name, input_obj in inputs_items:
            # If one of the inputs doesn't depend on the parameter, we simply
            # ignore it. No gradient information in there!
            if param in input_obj.gradients:
                dself_dinput = local_grads[input_name]
                dinput_dparam = input_obj.gradients[param]

                grads += np.matmul(dself_dinput, dinput_dparam)

        out_grads[param] = grads
    return out_grads

def tan(angle: Scalar) -> Scalar:
    angle = Scalar(angle)

    inputs = {"angle": angle}
    grads = {"angle": [[1 / (_cos(angle.value) ** 2)]]}

    val_out = _tan(angle.value)

    return Scalar(val_out).with_grads_from_previous(inputs, grads)


def arctan(angle: Scalar) -> Scalar:
    angle = Scalar(angle)

    inputs = {"angle": angle}
    grads = {"angle": [[1 / (angle.value ** 2 + 1)]]}

    val_out = _arctan(angle.value)

    return Scalar(val_out).with_grads_from_previous(inputs, grads)


def sin(scal: Scalar) -> Scalar:
    scal = Scalar(scal)

    inputs = {"s": scal}

    grads = {}
    grads["s"] = [[_cos(scal.value)]]

    val_out = _sin(scal.value)

    return Scalar(val_out).with_grads_from_previous(inputs, grads)


def cos(scal: Scalar) -> Scalar:
    scal = Scalar(scal)

    inputs = {"s": scal}

    grads = {}
    grads["s"] = [[-_sin(scal.value)]]

    val_out = _cos(scal.value)

    return Scalar(val_out).with_grads_from_previous(inputs, grads)


def combine_gradients(carriers: ty.List[Scalar]):
    inputs: ty.List[str] = []
    for carrier in carriers:
        inputs.extend([key for key in carrier.grads.keys() if key not in inputs])
    # inputs = ["l", "sx", "sy",...]

    grads = {}
    for input_name in inputs:
        grads[input_name] = [0] * len(carriers)
    # grads = {'x1': [0, 0], 'x2': [0, 0], 'y1': [0, 0], 'y2': [0, 0]}

    for iout, carrier in enumerate(carriers):
        for input_name in inputs:
            # grads {'x': array([[9.23958299e-10]]), 's': array([[-0.05108443]])}
            if not input_name in carrier.grads:
                continue

            dout_dinput = carrier.grads[input_name][0][0]
            grads[input_name][iout] = dout_dinput

    # turn into column vector
    for gradname, gradvals in grads.items():
        grads[gradname] = np.reshape(gradvals, [-1, 1])

    return grads