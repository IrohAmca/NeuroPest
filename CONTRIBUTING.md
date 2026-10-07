# Contributing to NeuroPest

Thank you for your interest in contributing to **NeuroPest**! We welcome bug reports, documentation improvements, biophysical model validations, and feature contributions.

---

## Development Setup

NeuroPest uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible Python environment management.

### 1. Clone & Install

```bash
git clone https://github.com/IrohAmca/NeuroPest.git
cd NeuroPest
uv sync
```


### 2. Connectome Data Setup

You can run NeuroPest immediately with the lightweight synthetic toy circuit (146 neurons), or download and build the full FlyWire v783 biological connectome (~135 MB download):

```bash
uv run neuropest-download
```

---

## Running Tests & Code Quality

All unit and integration tests are self-contained and run on synthetic mock datasets:

```bash
# Run test suite
uv run pytest

# Run fast code linting (Ruff)
uv run ruff check .
```

Ensure all tests pass and no linter warnings exist before submitting a Pull Request.

---

## Code Quality & Architecture

- **Biophysical Consistency:** Synaptic weights, delays, and neuronal dynamics should adhere to the established parameters in [`REFERENCES.md`](REFERENCES.md).
- **Process Isolation:** Heavy computations (LIF simulation loop, screen capture) must remain isolated from the main Qt GUI thread.
- **Type Annotations & Python Support:** We target Python 3.11+. Please maintain clean type hints.

---

## Submitting Pull Requests

1. Fork the repository and create a feature branch (`git checkout -b feat/your-feature`).
2. Commit your changes with clear, descriptive commit messages.
3. Verify that the automated test suite passes (`uv run pytest`).
4. Push your branch to GitHub and open a Pull Request against `main`.

---

## License Notice

By contributing to NeuroPest, you agree that your contributions will be licensed under the project's [PolyForm Noncommercial License 1.0.0](LICENSE).
