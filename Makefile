.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint
lint:
	flake8 compose-ml && isort --check-only --recursive compose-ml

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --exclude="*/migrations/*" --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221" compose-ml
	isort --recursive compose-ml

.PHONY: test
test: lint
	pytest compose-ml/tests

.PHONY: testcoverage
testcoverage: lint
	pytest compose-ml/tests --cov=compose-ml

.PHONY: installdeps
installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -r dev-requirements.txt