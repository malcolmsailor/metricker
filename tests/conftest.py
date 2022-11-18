import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--quick", action="store_true", help="run 'quick' version of tests"
    )
    parser.addoption(
        "--slow", action="store_true", help="run 'slow', comprehensive tests"
    )


@pytest.fixture(scope="session")
def quick(request):
    return request.config.option.quick


@pytest.fixture(scope="session")
def slow(request):
    return request.config.option.slow


def pytest_sessionstart(session):
    pass


def pytest_sessionfinish(session, exitstatus):
    pass
