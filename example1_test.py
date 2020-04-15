from example1 import main


def test_performance(benchmark):
    result = benchmark(main)
