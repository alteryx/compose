.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint
lint:
	flake8 composeml && isort --check-only --recursive composeml

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221" composeml
	isort --recursive composeml

.PHONY: test
test:
	pytest composeml/tests --cov=composeml

.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install -e . -q
	pip install -r dev-requirements.txt -q