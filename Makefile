docs_build:
	make -C docs clean html

doc_tests:
	make -C docs -e "SPHINXOPTS=-W" clean html

example_run:
	jupyter nbconvert --inplace --execute docs/source/examples/*.ipynb

.PHONY: lint
lint:
	isort --check-only composeml
	black composeml -t py39 --check
	flake8 composeml

.PHONY: lint-fix
lint-fix:
	black -t py39 composeml
	isort composeml

package_build:
	rm -rf dist/package
	python setup.py sdist
	$(eval package=$(shell python setup.py --fullname))
	tar -zxvf "dist/${package}.tar.gz" 
	mv ${package} dist/package

test:
	pytest composeml --cache-clear --show-capture=stderr -vv ${addopts}

checkdeps:
	$(eval allow_list='matplotlib|pandas|seaborn|woodwork|featuretools|evalml|tqdm')
	pip freeze | grep -v "compose.git" | grep -E $(allow_list) > $(OUTPUT_FILEPATH)

.PHONY: installdeps
installdeps: upgradepip
	pip install -e .
	pip install -r dev-requirements.txt

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip
