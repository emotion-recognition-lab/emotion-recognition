install:
    uv sync --all-extras --dev

install-tuna:
    uv sync --all-extras --dev --index-url https://pypi.tuna.tsinghua.edu.cn/simple

ruff:
    uvx ruff format .
    uvx ruff check . --fix --unsafe-fixes

check:
    just ruff
