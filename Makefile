.PHONY: benchmarks benchmarks-9x10 benchmarks-nus333 benchmarks-nus333-neos-only benchmarks-nlp docs

PYTHON ?= .venv/bin/python
BENCH_DIR := docs/site/_data/benchmarks
BENCH_RUNS ?= 5

# Full benchmark refresh. NUS333 gets the dual-reference treatment
# (Python vs NEOS + Python vs GAMS local + wall-time N runs).
# 9x10 stays NEOS-only: the model exceeds the GAMS community-license
# nonlinear limit of 2500 rows/cols, so it cannot be run locally.
benchmarks: benchmarks-9x10 benchmarks-nus333

benchmarks-9x10:
	@mkdir -p $(BENCH_DIR)
	$(PYTHON) scripts/gtap/diff_9x10_full.py --csv $(BENCH_DIR)/9x10.csv

# Dual-reference NUS333 bench: writes
#   $(BENCH_DIR)/nus333.csv         Python vs NEOS
#   $(BENCH_DIR)/nus333_local.csv   Python vs GAMS local COMP.gdx
#   $(BENCH_DIR)/nus333_timing.csv  per-run wall-time samples
# Tune the number of timing runs with `make benchmarks-nus333 BENCH_RUNS=10`.
benchmarks-nus333:
	@mkdir -p $(BENCH_DIR)
	$(PYTHON) scripts/gtap/bench_nus333_dual.py --runs $(BENCH_RUNS) \
	    --neos-csv $(BENCH_DIR)/nus333.csv \
	    --local-csv $(BENCH_DIR)/nus333_local.csv \
	    --timing-csv $(BENCH_DIR)/nus333_timing.csv

# NLP wall-time bench: Python (IPOPT) vs GAMS ifMCP=0 (IPOPT) local, same host.
# Unlike MCP/PATH (community-license capped at 1000 rows -> NEOS-only above
# ~NUS333), the NLP path uses open-source IPOPT which GAMS does NOT cap, so both
# sides run locally up to gtap7_15x10. Writes:
#   $(BENCH_DIR)/nlp_timing.csv   per-(dataset,mode,ifsub) per-run wall-times
# 10x7 and 15x10 are slow; tune with `make benchmarks-nlp BENCH_RUNS=3`.
benchmarks-nlp:
	@mkdir -p $(BENCH_DIR)
	$(PYTHON) scripts/gtap/bench_nlp_timing.py --runs $(BENCH_RUNS) \
	    --timing-csv $(BENCH_DIR)/nlp_timing.csv

# Old NEOS-only NUS333 path, kept for environments without local GAMS.
benchmarks-nus333-neos-only:
	@mkdir -p $(BENCH_DIR)
	$(PYTHON) scripts/gtap/diff_nus333_full.py --csv $(BENCH_DIR)/nus333.csv

docs:
	$(MAKE) -C docs/site html
