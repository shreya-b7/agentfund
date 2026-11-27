PY ?= .venv/bin/python

.PHONY: setup test compare leakage
setup:
	uv venv .venv && uv pip install --python $(PY) -e ".[dev]"
test:
	$(PY) -m pytest -q
compare:
	$(PY) -m agentfund compare
leakage:
	for s in equal_weight momentum_12_1 inverse_vol; do $(PY) -m agentfund leakage --strategy $$s || exit 1; done
