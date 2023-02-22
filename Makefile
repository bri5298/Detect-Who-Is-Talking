install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black detect_who_is_talking
	isort detect_who_is_talking

lint:
	flake8 detect_who_is_talking
	mypy detect_who_is_talking

#test:
#	pytest -vv --cov-report term-missing --cov=detect_who_is_talking

all: format lint #test
