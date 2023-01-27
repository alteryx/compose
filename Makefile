.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	black . --check --config=./pyproject.toml
	ruff . --config=./pyproject.toml

.PHONY: lint-fix
lint-fix:
	black . --config=./pyproject.toml
	ruff . --fix --config=./pyproject.toml

.PHONY: test
test:
	python -m pytest composeml/ -n auto

.PHONY: testcoverage
testcoverage:
	python -m pytest composeml/ --cov=composeml -n auto

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
	$(eval PACKAGE=$(shell python -c 'import setuptools; setuptools.setup()' --version))
	tar -zxvf "dist/composeml-${PACKAGE}.tar.gz"
	mv "composeml-${PACKAGE}" unpacked_sdist
