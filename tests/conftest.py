import pytest

from agentfund.data import generate_market


@pytest.fixture(scope="session")
def store():
    return generate_market(n_tickers=30, start="2016-01-01", end="2020-12-31", seed=11)
