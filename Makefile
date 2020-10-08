lint-fix:
	select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221"
	autopep8 --in-place --recursive --max-line-length=100 --select=${select} composeml
	isort --recursive composeml

lint-tests:
	flake8 composeml
	isort --check-only --recursive composeml

unit-tests:
	pytest composeml --cache-clear --show-capture=stderr -vv ${ADDOPTS}

docs-build:
	make -C docs clean html

docs-build-test:
	make -C docs -e "SPHINXOPTS=-W" clean html

doc-tests: docs-build-test

example-run:
	jupyter nbconvert --inplace --execute docs/source/examples/*.ipynb

package-build:
	rm -rf dist/package
	python setup.py sdist
	$(eval package=$(shell python setup.py --fullname))
	tar -zxvf "dist/${package}.tar.gz" 
	mv ${package} dist/package
