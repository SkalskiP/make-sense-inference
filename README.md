# makesense.ai inference
This repository is meant to be used as a template for serving models as part of integration with
[makesense.ai](https://github.com/SkalskiP/make-sense)


## Repository structure
#TODO

## :rotating_light: Repository setup
To initialize conda environment use
```bash
conda create -n MakeSenseServing python=3.9
conda activate MakeSenseServing
```

To install dependencies use
```bash
(TestingInMLEWorld) repository_root$ pip install -r requirements[-gpu].txt
(TestingInMLEWorld) repository_root$ pip install -r requirements-dev.txt
```

To enable `pre-commit` use
```bash
(TestingInMLEWorld) repository_root$ pre-commit install
```

To run `pre-commit` check
```bash
(TestingInMLEWorld) repository_root$ pre-commit
```

To run tests, linter and type-checker
```bash
(TestingInMLEWorld) repository_root$ pytest
(TestingInMLEWorld) repository_root$ black .
(TestingInMLEWorld) repository_root$ mypy .
```
