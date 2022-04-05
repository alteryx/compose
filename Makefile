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
	black composeml/ -t py310 --check
	flake8 composeml/

.PHONY: lint-fix
lint-fix:
	black -t py310 composeml/
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
	pip freeze | grep -v "alteryx/composeml.git" | grep -E $(allow_list) > $(OUTPUT_PATH)

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: package_compose
package_compose: upgradepip upgradebuild
	python -m build
	$(eval FT_VERSION := $(shell grep '__version__\s=' composeml/version.py | grep -o '[^ ]*$$'))
	tar -zxvf "dist/composeml-${FT_VERSION}.tar.gz"
	mv "composeml-${FT_VERSION}" unpacked_sdist