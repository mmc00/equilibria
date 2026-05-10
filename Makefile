.PHONY: benchmarks benchmarks-9x10 benchmarks-nus333 docs

PYTHON ?= .venv/bin/python
BENCH_DIR := docs/site/_data/benchmarks

benchmarks: benchmarks-9x10 benchmarks-nus333

benchmarks-9x10:
	@mkdir -p $(BENCH_DIR)
	$(PYTHON) scripts/gtap/diff_9x10_full.py --csv $(BENCH_DIR)/9x10.csv

benchmarks-nus333:
	@mkdir -p $(BENCH_DIR)
	$(PYTHON) scripts/gtap/diff_nus333_full.py --csv $(BENCH_DIR)/nus333.csv

docs:
	$(MAKE) -C docs/site html
