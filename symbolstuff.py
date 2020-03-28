import copy
from numbers import Number

class Param:
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
        return "Param({}*{}={})".format(self.factor, self.name, self.value)

    def compute(self):
        return self.factor * self.value ** self.power

def main():
    l = Param("l", 2.0)

    print(2*l)
    print(l)

    print("d =", l.factor)
    print((2*l).compute())



if __name__ == '__main__':
    main()