# od-matrix-completion

Установка [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)-окружения:
```bash
uv python install 3.13
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Установка проекта:
```bash
uv sync
```

Добавление новых зависимостей:
```bash
uv add package_name
```

Если добавил не то:
```bash
uv remove package_name
```