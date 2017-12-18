MODELS = logreg

all: data

INPUT= input/test.csv input/train.csv
DATA= data/train_x.csv data/train_y.csv data/test_x.csv data/test_id.csv

data: $(DATA)

data/%.csv: data.py clean-data.py $(INPUT)
	python clean-data.py

clean:
	rm -f $(DATA)