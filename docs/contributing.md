# Contributing

SIAI welcomes focused changes that preserve the small, trainer-independent core.

## Local setup

```bash
git clone https://github.com/penguinkang/siai.git
cd siai
python -m venv .venv
source .venv/bin/activate
pip install -e ".[docs]"
```

Run the tests and documentation build before opening a pull request:

```bash
python -m unittest discover -s tests -v
mkdocs build --strict
```

## Branches

`master` contains the release library and documentation. Development-only sample
projects, measured outputs, and larger training environments live on
`dev/micro-projects` and must not be merged into release artifacts.

## Contribution criteria

A change should have a clear contract, relevant tests, and documentation of its
failure behavior. Trainer integrations should keep provider-specific dependencies
optional. Performance claims require equal-budget comparisons and reproducible
evidence; a successful smoke test is an integration result, not an effectiveness
result.

