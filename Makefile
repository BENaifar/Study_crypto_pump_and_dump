# Brief deliverable: one command regenerates every table, figure and PDF of the notebooks (seeds fixed).
#   make all            QUICK sizes (default)
#   make all FULL=1     sizes required by the brief (PND_QUICK=0)
PND_QUICK := $(if $(FULL),0,1)

.PHONY: all notebooks report test lint

all: notebooks

notebooks:
	PND_QUICK=$(PND_QUICK) uv run python notebooks/run_all.py

report:
	uv run python notebooks/run_all.py --report-only

test:
	uv run pytest

lint:
	uv run ruff check src tests notebooks/run_all.py notebooks/build_complete_report.py notebooks/_src
