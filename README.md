# makesense.ai inference
This repository is meant to be used as a template for serving models as part of integration with
[makesense.ai](https://github.com/SkalskiP/make-sense)


## Repository structure
#TODO

## Contribution guide

### :rotating_light: Repository setup
To initialize conda environment use
```bash
conda create -n MakeSenseServing python=3.9
conda activate MakeSenseServing
```

To install dependencies use
```bash
(MakeSenseServing) repository_root$ pip install -r requirements[-gpu].txt
(MakeSenseServing) repository_root$ pip install -r requirements-dev.txt
```

To enable `pre-commit` use
```bash
(MakeSenseServing) repository_root$ pre-commit install
```

To run `pre-commit` check
```bash
(MakeSenseServing) repository_root$ pre-commit
```

To run tests, linter
```bash
(MakeSenseServing) repository_root$ pytest
(MakeSenseServing) repository_root$ black .
```
