from __future__ import annotations

import copy
import typing as ty
import warnings
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

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
        # assert self_copy.gradients == {}

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

    def __eq__(self: Scalar, other: Union[Scalar, float, int]) -> bool:
        if not isinstance(other, Scalar):
            return np.isclose(self.value, other)

        coords_equal = self.value == other.value
        grads_equal = True

        for key, val in self.grads.items():
            if key not in other.grads or not np.allclose(val, other.grads[key]):
                grads_equal = False
                break

        return coords_equal and grads_equal

    def __lt__(scal1: Scalar, other: Union[Scalar, float, int]) -> bool:
        if not isinstance(other, Scalar):
            return scal1.value < other

        # TODO: Is there a meaningful less than comparison amongst gradients?
        coords_equal = scal1.value < other.value
        return coords_equal

    def __le__(scal1: Scalar, other: Union[Scalar, float, int]) -> bool:
        return (scal1 < other) or (scal1 == other)

    # TODO: Add gt, ge

    def __rpow__(power: Scalar, base:Any) -> bool:
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


## UTILS
# TODO: Move to own file

def tan(angle: Scalar) -> Scalar:
    angle = Scalar(angle)

    inputs = {"angle": angle}
    grads = {"angle": [[1/(np.cos(angle.value)**2)]]}

    val_out = np.tan(angle.value)

    return Scalar(val_out).with_grads_from_previous(inputs, grads)

def arctan(angle: Scalar) -> Scalar:
    angle = Scalar(angle)

    inputs = {"angle": angle}
    grads = {"angle": [[1/(angle.value**2 + 1)]]}

    val_out = np.arctan(angle.value)

    return Scalar(val_out).with_grads_from_previous(inputs, grads)

def sin(scal: Scalar) -> Scalar:
    scal = Scalar(scal)

    inputs = {"s": scal}

    grads = {}
    grads["s"] = [[np.cos(scal.value)]]

    val_out = np.sin(scal.value)

    return Scalar(val_out).with_grads_from_previous(inputs, grads)


def cos(scal: Scalar) -> Scalar:
    scal = Scalar(scal)

    inputs = {"s": scal}

    grads = {}
    grads["s"] = [[-np.sin(scal.value)]]

    val_out = np.cos(scal.value)

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


class Point(GradientCarrier):
    def __init__(self, x, y):
        super().__init__()

        x = Scalar(x)
        y = Scalar(y)

        inputs = {"_x": x, "_y": y}
        local_grads = {"_x": [[1], [0]], "_y": [[0], [1]]}

        self.x = x
        self.y = y
        self.gradients = combine_gradients(self.properties)

    @property
    def properties(self):
        return [self.x, self.y]

    def as_numpy(self):
        return np.reshape([prop.value for prop in self.properties], (-1, 1))

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x, self.y)

    def __truediv__(pt: Point, s: ty.Union[Scalar, Number]) -> Point:
        if isinstance(s, Scalar):
            new_x = pt.x / s
            new_y = pt.y / s

            return Point(new_x, new_y)

        raise NotImplementedError("__truediv__ not yet impl for normal numbers")

    def __eq__(pt1: Point, pt2: Point) -> bool:
        coords_equal = pt1.same_as(pt2)
        grads_equal = True

        for key, val in pt1.grads.items():
            if key not in pt2.grads or not np.allclose(val, pt2.grads[key]):
                grads_equal = False
                break

        return coords_equal and grads_equal

    def __mul__(pt: Point, other: ty.Union[Point, Scalar, Number]) -> Point:
        if isinstance(other, Scalar):
            new_x = pt.x * other
            new_y = pt.y * other

            return Point(new_x, new_y)

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

        return Point(u,v)

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
        # TODO: Same for points, coercion
        angle_rad = Scalar(angle_rad)

        x1 = pt.x
        y1 = pt.y

        ox = origin.x
        oy = origin.y

        angle = angle_rad.value

        x2 = (x1 - ox) * cos(angle_rad) - (y1 - oy) * sin(angle_rad) + ox
        y2 = (x1 - ox) * sin(angle_rad) + (y1 - oy) * cos(angle_rad) + oy

        return Point(x2, y2)


Vector = Point
Param = Scalar.Param


def update_grads(
    inputs: ty.Dict[str, GradientCarrier],
    local_grads: ty.Dict[str, ty.Union[ty.List[Number], np.ndarray]],
):
    incoming_parameters = []
    for input_name, input_obj in inputs.items():
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
            # If one of the inputs doesn't depend on the parameter, we simply
            # ignore it. No gradient information in there!
            if param in input_obj.gradients:
                dself_dinput = local_grads[input_name]
                dinput_dparam = input_obj.gradients[param]

                grads += np.matmul(dself_dinput, dinput_dparam)

        out_grads[param] = grads
    return out_grads


class Line2(GradientCarrier):
    """
    An infinite line. Parametrized by origin and direction vector. (4 parameters)
    """

    def __init__(self, ox, oy, dx, dy):
        super().__init__()

        ox = Scalar(ox)
        oy = Scalar(oy)

        dx = Scalar(dx)
        dy = Scalar(dy)

        self._params = [ox, oy, dx, dy]

    @staticmethod
    def make_from_points(pt1: Point, pt2: Point) -> Line2:
        direction = (pt2 - pt1) / Scalar(np.linalg.norm((pt2 - pt1).as_numpy().ravel()))
        return Line2(pt1.x, pt1.y, direction.x, direction.y)

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

    def eval_at(line: Line2, s) -> Point:
        s = Scalar(s)

        x = line.ox + s * line.dx
        y = line.oy + s * line.dy

        return Point(x, y)

    def intersect(line1: Line2, line2: Line2) -> Point:
        o2x = line1.ox
        o1x = line2.ox

        o2y = line1.oy
        o1y = line2.oy

        d2x = line1.dx
        d1x = line2.dx

        d2y = line1.dy
        d1y = line2.dy

        s1 = ((o2y - o1y) * d1x - (o2x - o1x) * d1y) / (d2x * d1y - d2y * d1x)

        return line1.eval_at(s1)

    def plot(self, ax=plt):
        s = np.arange(0, 10)
        pts = [self.eval_at(si) for si in s]
        xys = [(pt.x, pt.y) for pt in pts]

        ax.plot(*zip(*xys))


class Line(GradientCarrier):
    """ Simple class representing a line, used to construct the unit cell """

    # TODO: Replace by better representation with no singularities
    def __init__(self, m, b):
        super().__init__()

        m = Scalar(m)
        b = Scalar(b)

        self.m = m
        self.b = b
        self.gradients = combine_gradients(self.properties)

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

        return Line(m, b)

    def translate(a_line: Line, vec: Vector) -> Line:
        line_old = copy.deepcopy(a_line)

        m = line_old.m
        b = line_old.b + vec.y - line_old.m * vec.x

        return Line(m, b)

    def rotate_ccw(line1: Line, angle_rad: Scalar, pivot: Point = None) -> Line:
        angle_rad = Scalar(angle_rad)
        
        if pivot is None:
            pivot = Point(0, 0)

        line_centered = line1.translate(-pivot)

        m = line_centered.m
        b = line_centered.b

        m2 = tan(arctan(m) + angle_rad)
        b2 = b

        return Line(m2, b2).translate(pivot)

    def intersect(line_1: Line, line_2: Line) -> Point:
        m1 = line_1.m
        b1 = line_1.b

        m2 = line_2.m
        b2 = line_2.b

        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

        return Point(x,y)

    def plot(self, ax=None, lims=(-20, 20, 10), label=None):
        import matplotlib.pyplot as plt  # type: ignore

        if ax is None:
            ax = plt

        x = np.linspace(*lims)
        y = self.m * x + self.b

        ax.plot(x, y, label=label)
        ax.axis("equal")
