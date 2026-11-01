UV = uv
RUN = $(UV) run

all: install

install:
	$(UV) sync --extra dev

run:
	$(RUN) python -m src $(ARGS)

debug:
	$(RUN) python -m pdb -m src

lint:
	$(RUN) flake8 src
	$(RUN) mypy src \
			--warn-return-any \
			--warn-unused-ignores \
			--ignore-missing-imports \
			--disallow-untyped-defs \
			--check-untyped-defs

test:
	$(RUN) pytest

lint-strict:
	$(RUN) flake8 src
	$(RUN) mypy src --strict

clean:
	@echo "Cleaning up temporary files, indexes, and caches..."
	@find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" -o -name "*.egg-info" -o -name "dist" -o -name "build" \) -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .venv .uv-cache .hf-cache .cache .vscode .locks
	@rm -rf models--* index_store
	@rm -f uv.lock

.PHONY: all install run debug lint lint-strict clean
