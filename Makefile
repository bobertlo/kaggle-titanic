MODELS = logreg

all: data bb-models

INPUT= input/test.csv input/train.csv
DATA= data/train.csv data/test.csv
BBMODS = output/bb-logreg.csv output/bb-rf.csv output/bb-ada.csv output/bb-gb.csv output/bb-et.csv output/bb-svc.csv output/bb-voting.csv

data: $(DATA)

bb-models: $(BBMODS)

data/%.csv: data.py clean-data.py $(INPUT)
	python clean-data.py

output/bb-%.csv: data bb-models.py
	python bb-models.py

clean:
	rm -f $(DATA) $(BBMODS)