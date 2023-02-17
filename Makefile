install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black detect_who_is_talking tests
	isort detect_who_is_talking tests

lint:
	flake8 detect_who_is_talking tests
	mypy detect_who_is_talking tests

test:
	pytest -vv --cov-report term-missing --cov=detect_who_is_talking tests

all: format lint test
