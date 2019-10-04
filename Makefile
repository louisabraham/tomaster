pypi: test dist
	twine upload dist/*
	
dist: flake8
	-rm dist/*
	python setup.py sdist bdist_wheel

flake8:
	flake8 . --count --select=E901,E999,F821,F822,F823 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

clean:
	-rm -rf __pycache__ *.egg-info build dist .pytest_cache

test:
	pytest

.PHONY: pypi dist flake8 clean test