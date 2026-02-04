# Claude Code Guidelines for pseudopeople

## Project Overview

pseudopeople adds realistic noise to simulated population data. The public API is dataset generation functions like `generate_decennial_census()`, `generate_american_community_survey()`, etc.

## Code Structure

- `src/pseudopeople/interface.py` - Public API functions for generating datasets
- `src/pseudopeople/noise.py` - Core noising logic
- `src/pseudopeople/configuration/` - Noise configuration handling
- `src/pseudopeople/schema_entities.py` - Dataset and column definitions
- `tests/unit/` - Unit tests (run with `pytest tests/unit/`)
- `tests/integration/` - Integration tests (require additional data)

## Development Practices

- **Avoid breaking changes** - Don't rename public function parameters
- **Keep it simple** - Use existing libraries rather than writing custom solutions (e.g., `tqdm.auto` handles Jupyter detection automatically)
- **Progress bars** - Use `from tqdm.auto import tqdm` for automatic Jupyter notebook support

## Testing

```bash
pip install -e .
pytest tests/unit/
```

Integration tests are skipped by default (require external data).
