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

    def grad(self):
        return self.factor * self.power * self.value ** (self.power-1)

class Differentiable:
    def __init__(self):
        pass


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.gradients = {}

        """
        each point has a
        - deriv wrt. to the point it was "computed" from (through translation, rotation, etc.)
        - deriv wrt. additional original parameters if they were used in the construction of this point
        """

    def __repr__(self):
        return "Pt({:.4f},{:.4f})".format(self.x, self.y)

    @property
    def grads(self):
        return copy.deepcopy(self.gradients)

    def update_grads(self, new_grads):
        """
        Chains new gradients onto the current ones
        """
        pt2 = copy.deepcopy(self)
        old_grads = pt2.gradients
        updated_grads = {}

        params = set(old_grads.keys()).union(set(new_grads.keys()))

        for param in params:
            if param == "d_prevpt":
                continue

            # we already have a trace to this parameter
            if param in old_grads:
                print("Trying to chain ", param)
                # we take our gradient towards last point
                # and the gradient of old point towards parameter
                # d_dl = d_dprev @ dprev_dl
                updated_grads[param] = np.array(new_grads["d_prevpt"]) @ old_grads[param]

            # we don't have a trace yet, meaning we start one
            else:
                # d_dl = XXX
                updated_grads[param] = new_grads[param]

        pt2.gradients = updated_grads
        return pt2

Vector = Point

def translate(pt, vec):
    deltax = vec[0].compute() if isinstance(vec[0], Param) else vec[0]
    deltay = vec[1].compute() if isinstance(vec[1], Param) else vec[1]

    x2 = pt.x + deltax
    y2 = pt.y + deltay

    _grads = {}
    if isinstance(vec[0], Param):
        _grads[vec[0].name] = [
            [vec[0].grad()],
            [0]
        ]
    
    if isinstance(vec[1], Param):
        _grads[vec[1].name] = [
            [0],
            [vec[1].grad()]
        ]

    d_prevpt = np.array([
        [1, 0],
        [0, 1]
    ])

    _grads["d_prevpt"] = d_prevpt

    pt2 = pt.update_grads(_grads)
    pt2.x = x2
    pt2.y = y2

    return pt2

def norm(pt):
    params = list(pt.grads.keys())

    l2_norm = np.sqrt(pt.x**2 + pt.y ** 2)

    grad = [[pt.x/np.sqrt(pt.x**2 + pt.y**2), pt.y/np.sqrt(pt.x**2 + pt.y**2)]]

    grads = {}
    grads["d_prevpt"] = grad

    print("new grads:", grads)

    # Point isnt the right class. Should Differentiable or Scalar or Thing or sth
    length = pt.update_grads(grads)
    length.x = l2_norm
    length.y = 0

    return length

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
    _grads["d_prevpt"] = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

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
        
        pt2 = translate(pt, [l, 0])
        pt3 = rotate_param(pt2, pt, theta)
        pt4 = translate(pt3, [2*l, 0])

        #length = norm(pt4)
        #print(length)

        """
        A = diff_vec = pt4 - pt2
        dA/dparams = pt4.grads() - pt2.grads()

        B = diff = diff_vec.length()
        dB/dparams = dlength/dA * dA/dparams
        
        translate(pt4, [diff * l, 0])  # diff * l is a Scalar/Point * param
            diff.compute() * l.compute()

            if other is param:
                per parameter:
                    self.grad[parameter] *= other.grad() * self.compute() + self.grad() * other.compute()
        """

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
