from pytest import fixture

@fixture
def check(check_gpu):
    return check_gpu

def test_gpu_iobatch_worker_partitioning_even(check):
    pass