install:
    uv sync --all-extras --dev

ruff:
    uv run ruff format .
    uv run ruff check . --fix --unsafe-fixes

check:
    just ruff
