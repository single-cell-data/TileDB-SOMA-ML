# TileDB-SOMA-ML docs build

From this directory:
```bash
# Build static site under _build/html
make html

# 1. Build docs/  # `make html`, outputs site to _build/html/
# 2. Build www/   # Outputs shuffle viz at dist/shuffle/
# 3. cp -r _build/html/* www/dist/  # Combine 1. and 2. under www/dist/
make dist
```

The [`ghp.yml`] GHA also performs these steps, and deploys www/dist to [single-cell-data.github.io/TileDB-SOMA-ML].

[`ghp.yml`]: ../.github/workflows/ghp.yml
[single-cell-data.github.io/TileDB-SOMA-ML]: https://single-cell-data.github.io/TileDB-SOMA-ML/
