# ═══════════════════════════════════════════════════════════════════════════════
# TRADING TOOLKIT - Makefile
# ═══════════════════════════════════════════════════════════════════════════════

PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

.PHONY: setup install test clean run-backtest run-optimize run-ui help

# Default target
help:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "                    TRADING TOOLKIT - Commands                      "
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  make setup          - Create virtual environment and install deps"
	@echo "  make run-ui         - 🚀 Launch the Streamlit UI"
	@echo "  make install        - Install dependencies only"
	@echo "  make test           - Run unit tests"
	@echo "  make run-backtest   - Run single backtest with default config"
	@echo "  make run-optimize   - Run Bayesian optimization"
	@echo "  make run-quick      - Quick optimization (fewer trials)"
	@echo "  make clean          - Remove venv and cache files"
	@echo ""
	@echo "Examples:"
	@echo "  make setup && make run-backtest"
	@echo "  make run-optimize SYMBOL=TSLA INTERVAL=15m"
	@echo ""

# Setup virtual environment
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "✅ Setup complete! Activate with: source $(VENV)/bin/activate"

# Install dependencies only
install:
	$(PIP) install -r requirements.txt

# Run unit tests
test: setup
	@echo "Running tests..."
	$(PYTEST) tests/ -v

# Run Streamlit UI
run-ui: setup
	@echo "Starting Trading Toolkit UI..."
	$(VENV)/bin/streamlit run app.py

# Run single backtest
run-backtest: setup
	@echo "Running backtest..."
	$(VENV)/bin/python scripts/run_backtest.py \
		--symbol $(or $(SYMBOL),SPY) \
		--interval $(or $(INTERVAL),1d) \
		--start $(or $(START),2020-01-01) \
		--end $(or $(END),2024-01-01)

# Run full Bayesian optimization
run-optimize: setup
	@echo "Running Bayesian optimization..."
	$(VENV)/bin/python scripts/run_optimization.py \
		--symbol $(or $(SYMBOL),SPY) \
		--interval $(or $(INTERVAL),1d) \
		--start $(or $(START),2020-01-01) \
		--end $(or $(END),2024-01-01) \
		--trials $(or $(TRIALS),200)

# Quick optimization (fewer trials)
run-quick: setup
	@echo "Running quick optimization..."
	$(VENV)/bin/python scripts/run_optimization.py \
		--symbol $(or $(SYMBOL),SPY) \
		--interval $(or $(INTERVAL),1d) \
		--start $(or $(START),2022-01-01) \
		--end $(or $(END),2024-01-01) \
		--trials 50

# Generate HTML report from last run
report: setup
	@echo "Generating report..."
	$(VENV)/bin/python scripts/generate_report.py

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/*/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf reports/*.html
	@echo "✅ Cleaned"
