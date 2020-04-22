from example1 import main, create_unit_cell


def test_performance(benchmark):
    result = benchmark(main)
