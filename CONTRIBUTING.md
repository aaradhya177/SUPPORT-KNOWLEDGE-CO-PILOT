# Contributing

Thanks for improving Support Knowledge Copilot.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quality Checks

Run tests before opening a pull request:

```bash
pytest
```

Run formatting and linting:

```bash
black .
ruff check .
```

## Contribution Guidelines

- Keep changes scoped and covered by tests.
- Do not commit real API keys, `.env`, generated indexes, or private customer data.
- For retrieval, generation, verification, or scoring changes, update or add evaluation examples.
- If a change affects measured claims, rerun `python eval/run_eval.py` and update the reports.
