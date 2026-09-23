# make        -> .venv with the four dependencies (uv if installed, else python -m venv)
# make run    -> browser UI on http://127.0.0.1:8744, armed: it can drive heaters
# make test   -> every self-test, no hardware
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
SYS ?= python
else
PY := .venv/bin/python
SYS ?= python3
endif
UV := $(shell command -v uv 2>/dev/null)

.PHONY: venv run test clean
venv: $(PY)
$(PY): requirements.txt
ifdef UV
	uv venv -q --allow-existing .venv
	uv pip install -q --python $(PY) -r requirements.txt
else
	$(SYS) -m venv .venv
	$(PY) -m pip install -q -r requirements.txt
endif
	@touch $(PY)

run: $(PY)
	$(PY) ui.py --arm $(ARGS)

test: $(PY)
	$(PY) -m pic selftest
	$(PY) ui.py --selftest

clean:
	rm -rf .venv
