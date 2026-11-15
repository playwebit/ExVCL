# ExVCL

**Evidence-Governed Admission for Secure Continual Learning**

ExVCL is a Python research framework for evidence-governed admission in continual learning. It places execution-derived behavioural evidence in front of a continual learner and uses an admission gate to govern which labelled observations are allowed to update the learner.

The project supports research into constraining the influence of incorrect or corrupted training observations in continual-learning workflows.

## Repository structure

```text
ExVCL/
├── exvcl/       # Public Python package and framework API
├── evcl/        # Earlier research-engine implementation
├── README.md
├── LICENSE
└── CITATION.cff
```

- **`exvcl/`** is the public package intended for distribution through PyPI.
- **`evcl/`** is retained as research-source code associated with earlier experimental phases. It is not intended to be part of the public PyPI distribution.

## Main capabilities

The public package includes components for:

- Evidence modelling and calibration.
- Admission control with `none`, `selective`, and `strict` modes.
- Integration with continual-learning models.
- Evidence attribution configurations, including real, shuffled, global-random, and mixed evidence.
- Telemetry processing and collector-related utilities.

The package API also includes a bundled learner interface supporting Softmax, MLP, and Random Forest learners.

## Installation

The PyPI release is being prepared. After publication, install ExVCL with:

```bash
pip install exvcl
```

## Requirements

The current `exvcl` 0.2.0 package metadata declares:

- Python 3.9 or later
- NumPy
- pandas
- scikit-learn
- SciPy

## Research and evaluation

ExVCL has been developed and evaluated across multiple experimental phases, including simulated studies, real-host telemetry, multi-collector evaluation, and a live operational run.

The repository retains the `evcl/` research-engine source for continuity with earlier experiments. Experimental scripts, datasets, logs, and evaluation artifacts should be documented and distributed separately from the installable `exvcl` package.

Reported results apply to the tested configurations and environments. They are not a guarantee of effectiveness on every dataset, host environment, attack, or continual-learning system.

## Basic package imports

```python
from exvcl import (
    EvidenceModel,
    AdmissionGate,
    ExVCL,
    ChunkVerifier,
    make_learner,
)
```

Refer to the source modules and forthcoming examples/documentation for component-specific usage and configuration.

## License

This project is distributed under the Apache License 2.0. See [LICENSE](LICENSE).

## Citation

If you use ExVCL in academic work, please cite the software using the metadata in [CITATION.cff](CITATION.cff).

## Project

GitHub: https://github.com/playwebit/ExVCL
