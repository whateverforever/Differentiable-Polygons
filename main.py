import copy
import numpy as np
import matplotlib.pyplot as plt

from numbers import Number

from scipy import optimize

class Point:
    def __init__(self, x, y, params):
        self.x = x
        self.y = y
        self.gradients = {}
        self.params = params

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x, self.y)

    @property
    def grads(self):
        return copy.deepcopy(self.gradients)

def translate2(pt, vec):
    if isinstance(vec[0], Number):
        return translate_const(pt, vec)
    elif isinstance(vec[0], str):
        return translate_param1(pt, vec)

def translate_const(pt, vec):
    pass

def translate_param1(pt, vec):
    pname = vec[0]

    deltax = pt.params[pname]
    deltay = vec[1]

    x2 = pt.x + deltax
    y2 = pt.y + deltay

    _grads = {}
    _grads[pname] = [
        [1],
        [0]
    ]

    pt2 = copy.deepcopy(pt)
    pt2.x = x2
    pt2.y = y2
    pt2grads = pt2.gradients

    for param, grad in _grads.items():
        if param in pt2grads:
            pt2grads[param] *= np.array(_grads[param])
        else:
            pt2grads[param] = _grads[param]

    return pt2

def rotate_param(pt, origin, angle_param):
    x1 = pt.x
    y1 = pt.y

    ox = origin.x
    oy = origin.y

    angle = pt.params[angle_param]

    x2 = (x1 - ox) * np.cos(angle) - (y1 - oy) * np.sin(angle) + ox
    y2 = (x1 - ox) * np.sin(angle) + (y1 - oy) * np.cos(angle) + oy

    dpt = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    dorigin = [
        [-np.cos(angle) + 1, np.sin(angle)],
        [-np.sin(angle), -np.cos(angle) + 1],
    ]
    dangle = [
        [(x1 - ox) * (-np.sin(angle) - (y1 - oy) * np.cos(angle))],
        [(x1 - ox) * np.cos(angle) + (y1 - oy) * (-np.sin(angle))],
    ]

    _grads = {}
    _grads[angle_param] = dangle

    pt2 = copy.deepcopy(pt)
    pt2.x = x2
    pt2.y = y2

    for param, grad in _grads.items():
        if param in pt2.gradients:
            pt2.gradients[param] *= np.array(_grads[param])
        else:
            pt2.gradients[param] = _grads[param]

    return pt2

def translate(pt, vec):
    x2 = pt.x + vec.x
    y2 = pt.y + vec.y

    dpt = [1, 1]
    dvec = [1, 1]

    d = [dpt, dvec]

    return Point(x2, y2), d


def rotate(pt, origin, angle):
    x1 = pt.x
    y1 = pt.y

    ox = origin.x
    oy = origin.y

    x2 = (x1 - ox) * np.cos(angle) - (y1 - oy) * np.sin(angle) + ox
    y2 = (x1 - ox) * np.sin(angle) + (y1 - oy) * np.cos(angle) + oy

    dpt = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    dorigin = [
        [-np.cos(angle) + 1, np.sin(angle)],
        [-np.sin(angle), -np.cos(angle) + 1],
    ]
    dangle = [
        [(x1 - ox) * (-np.sin(angle) - (y1 - oy) * np.cos(angle))],
        [(x1 - ox) * np.cos(angle) + (y1 - oy) * (-np.sin(angle))],
    ]

    return Point(x2, y2), dpt, dorigin, dangle

def l2_distance(p1, p2):

    dist = np.linalg.norm(p1 - p2)

    dp1 = 2 * (p1 - p2)
    dp2 = 2 * (p1 - p2) * -1

    return dist, dp1.reshape(-1,2), dp2

class MultiPoly:
    def __init__(self):
        pass

    def compute(self, params):
        Point = MakePointInParameterSpace(params)

        a = Point(0,0)
        vec = Point("l",0)

        grad = vec.gradient()
        """
        grad = {
          "l": [1, 0]
        }
        
        d = rotate(vec, a, "angle")

        d.gradient() = {
            "l": rotate.grad(vec) @ vec.grad("l")
            "angle": rotate.grad(angle)
        }


        """

        # can the object, in this case the point, carry its own gradients?
        # wrt. parameters

        b, dtrans = translate(a, vec)
        c, _,_,_ = rotate(b, a, params["angle"])


def main():
    def parametric_pt(l=2.0, theta=np.radians(60)):
        """
        
        ideal:

        l = Param()
        theta = Param()

        translate2(pt, [2 * l, 0])

        """

        pt = Point(0, 0, {"l":l, "theta":theta})
        
        pt2 = translate2(pt, ["l", 0])
        pt3 = rotate_param(pt2, pt, "theta")
        pt4 = translate2(pt3, ["l", 0])

        return pt4, pt4.grads

    print("Go ####################\n\n")

    target = np.array([5.5, 1.1])

    def jac(x):
        l, theta = x
        pt, pt_grads = parametric_pt(l, theta)
        dist, dp1,_ = l2_distance([pt.x, pt.y], target)

        grads = []
        for param, grad in pt_grads.items():
            grads.append(float(dp1 @ grad))

        return np.array(grads)

    def f(x):
        l, theta = x
        pt, grads = parametric_pt(l, theta)
        dist,_,_ = l2_distance(np.array([pt.x, pt.y]), target)

        return dist

    res = optimize.minimize(f, [2.0, np.radians(60)], jac=jac)

    print(res)

    pt_reached, _  = parametric_pt(*res.x)
    print("Target was {}, arrived at {}".format(target, pt_reached))


def main2():
    a = Point(0, 0)
    vec = Point(3, 0)

    b, dtrans = translate(a, vec)
    c, dpt, dorigin, dangle = rotate(b, a, np.radians(45))

    d = copy.deepcopy(c)
    alpha = 0.1

    dangle = np.array(dangle)

    x = []
    y = []
    y2 = []

    i = 0

    angle = 30
    offset = np.array([0.0, 0.0])

    target = np.array([2.5, 2.598])

    dist, _, dd = l2_distance(target, np.array([d.x, d.y]) + offset)

    while dist > 0.001 and i < 500:
        x.append(i)
        y.append(angle)
        y2.append(offset[0])

        d, _, dorigin, dangle = rotate(b, a, np.radians(angle))

        dobj = np.dot(dd, dangle)
        angle -= alpha * dobj

        dobj = np.dot(dd, dorigin)
        offset -= alpha * dobj

        i += 1
        dist, _, dd = l2_distance(target, np.array([d.x, d.y]) + offset)

    print("Final")
    print("Angle", angle)
    print("Offset", offset)
    print("Final point", np.array([d.x, d.y]) + offset)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
