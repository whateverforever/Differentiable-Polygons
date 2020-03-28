import copy
import numpy as np
import matplotlib.pyplot as plt

from numbers import Number

from scipy import optimize

class Param:
    """
    Class used so that geometrical operations can access named parameters, and
    can change their gradients (from only const). Allows for a factor in front
    and a power in the back. But not more atm.
    """
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.factor = 1
        self.power = 1

    def __rmul__(self, other):
        self_copy = copy.deepcopy(self)

        if isinstance(other, Number):
            self_copy.factor = other

        return self_copy

    def __repr__(self):
        return "Param({}*{}^{}={})".format(self.factor, self.name, self.power, self.value)

    def compute(self):
        return self.factor * self.value ** self.power

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gradients = {}

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x, self.y)

    @property
    def grads(self):
        return copy.deepcopy(self.gradients)

    def update_grads(self, new_grads):
        pt2 = copy.deepcopy(self)
        pt2grads = pt2.gradients

        for param, grad in new_grads.items():
            if param in pt2grads:
                pt2grads[param] *= np.array(new_grads[param])
            else:
                pt2grads[param] = new_grads[param]

        return pt2

def translate2(pt, vec):
    if isinstance(vec[0], Number):
        return translate_const(pt, vec)
    elif isinstance(vec[0], Param):
        return translate_param1(pt, vec)

def translate_const(pt, vec):
    pass

def translate_param1(pt, vec):
    param = vec[0]

    deltax = param.compute()
    deltay = vec[1]

    x2 = pt.x + deltax
    y2 = pt.y + deltay

    _grads = {}
    _grads[param.name] = [
        [1],
        [0]
    ]

    pt2 = pt.update_grads(_grads)
    pt2.x = x2
    pt2.y = y2

    return pt2

def rotate_param(pt, origin, angle_param):
    x1 = pt.x
    y1 = pt.y

    ox = origin.x
    oy = origin.y

    angle = angle_param.compute()

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
    _grads[angle_param.name] = dangle

    pt2 = pt.update_grads(_grads)
    pt2.x = x2
    pt2.y = y2

    return pt2

def l2_distance(p1, p2):

    dist = np.linalg.norm(p1 - p2)

    dp1 = 2 * (p1 - p2)
    dp2 = 2 * (p1 - p2) * -1

    return dist, dp1.reshape(-1,2), dp2

def main():
    def parametric_pt(l=2.0, theta=np.radians(60)):
        l = Param("l", l)
        theta = Param("theta", theta)

        pt = Point(0, 0)
        
        pt2 = translate2(pt, [l, 0])
        pt3 = rotate_param(pt2, pt, theta)
        pt4 = translate2(pt3, [2*l, 0])

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

if __name__ == "__main__":
    main()
