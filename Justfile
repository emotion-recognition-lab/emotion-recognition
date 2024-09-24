install:
    uv sync --all-extras --dev

ruff:
    uv run ruff check . --fix --unsafe-fixes
    uv run ruff format .

check:
    just ruff
