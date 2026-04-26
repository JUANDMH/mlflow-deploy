install:
	pip install --upgrade pip
	pip install -r requirements.txt

format:
	black train.py validate.py

train:
	python train.py

validate:
	python validate.py

ci: install format train validate
