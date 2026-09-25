# macOS / Linux shorthand for bootstrap.py, which is what Windows uses directly:
#   make        -> .venv with the requirements
#   make run    -> the UI on http://127.0.0.1:8744, armed
#   make test   -> every self-test, no hardware
PY3 ?= python3

.PHONY: venv run test clean
venv:
	$(PY3) bootstrap.py
run:
	$(PY3) bootstrap.py run $(ARGS)
test:
	$(PY3) bootstrap.py test
clean:
	$(PY3) -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
