![Pytest](https://github.com/whateverforever/Differentiable-Polygons/workflows/Do%20some%20testing/badge.svg?event=push)

# Definition of Done

- [ ] Point contains all important methods
- [ ] Line has been completely replaced by Line2
- [ ] Line contains all important methods
- [ ] Add tests for all methods of polygons
- [ ] All GradientCarriers use _params notation (harmonize!)
- [ ] Polygons apply all their operations to their holes as well
- [ ] All tests use only public interfaces (no .m or .b is tested)
- [ ] Every method has its dedicated test that tests value and gradients
- [ ] Implemented fast version of Point and Line in cython
- [ ] Replace all if clauses in primitives_test with expected throws/warnings
- [ ] The polygon joining has been profiled and made as fast as possible
- [ ] No more TODO comments left over in code base
- [ ] All methods have docstrings
- [ ] All functions have complete type annotations
- [ ] The util functions with gradients are in a file of their own, with tests of their own
- [ ] Both main code files have no more than 500 sloc