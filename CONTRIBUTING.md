# Contributing

Thank you for your interest in contributing!

## How to contribute
- Open an issue to discuss bugs, features, or questions.
- Fork the repo and create a feature branch.
- Add tests for any changes; keep style consistent.
- Run `pytest -q` and ensure all tests pass.
- Open a pull request referencing the issue.

## Development setup
```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt  # if present
pytest -q
```

## Code style
- Prefer small, focused changes.
- Keep public APIs stable.
- No inline comments unless requested; use clear names.

## Releases
- Version via `pyproject.toml`.
- Tag releases and update the changelog.

## License
Contributions are licensed under the MIT License.
