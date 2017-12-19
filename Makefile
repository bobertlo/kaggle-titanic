MODELS = logreg

all: data bb-models grid-models

INPUT= input/test.csv input/train.csv
DATA= data/train.csv data/test.csv
BBMODS = output/bb-logreg.csv output/bb-rf.csv output/bb-ada.csv output/bb-gb.csv output/bb-et.csv output/bb-svc.csv output/bb-voting.csv
GRIDMODS = output/grid-rf.csv output/grid-voting.csv

data: $(DATA)


data/%.csv: data.py clean-data.py $(INPUT)
	python clean-data.py

bb-models: $(BBMODS)

output/bb-%.csv: data bb-models.py
	python bb-models.py

grid-models: $(GRIDMODS)

output/grid-%.csv: data grid-models.py
	python grid-models.py 

clean:
	rm -f $(DATA) $(BBMODS) $(GRIDMODS)
