# atsc3lib build/test entry points.
#
# sdrbindings (the SDRplay capture extension) is a separate package and a
# declared dependency, installed with the project rather than built here.
#
#   make            install atsc3lib (editable) and its dependencies
#   make test       run the test suite
#   make clean      remove build artefacts

PYTHON ?= python3

.PHONY: all install test clean

all: install

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf build dist *.egg-info atsc3lib/__pycache__ tests/__pycache__
	find atsc3lib tests -name '*.pyc' -delete
