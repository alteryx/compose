lint-fix:
	select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221"
	autopep8 --in-place --recursive --max-line-length=100 --select=${select} composeml
	isort --recursive composeml

lint-tests:
	flake8 composeml
	isort --check-only --recursive composeml

unit-tests:
	pytest --cache-clear --show-capture=stderr -vv ${ADDOPTS}
