.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	isort --check-only composeml/
	black composeml/ -t py311 --check
	flake8 composeml/

.PHONY: lint-fix
lint-fix:
	black -t py311 composeml/
	isort composeml/

.PHONY: test
test:
	pytest composeml/

.PHONY: testcoverage
testcoverage:
	pytest composeml/ --cov=composeml

.PHONY: installdeps
installdeps: upgradepip
	pip install -e ".[dev]"

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='matplotlib|pandas|seaborn|woodwork|featuretools|evalml|tqdm')
	pip freeze | grep -v "alteryx/compose.git" | grep -E $(allow_list) > $(OUTPUT_FILEPATH)

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: upgradesetuptools
upgradesetuptools:
	python -m pip install --upgrade setuptools

.PHONY: package
package: upgradepip upgradebuild upgradesetuptools
	python -m build
	$(eval COMPOSE_VERSION := $(shell grep '__version__\s=' composeml/version.py | grep -o '[^ ]*$$'))
	tar -zxvf "dist/composeml-${COMPOSE_VERSION}.tar.gz"
	mv "composeml-${COMPOSE_VERSION}" unpacked_sdist
