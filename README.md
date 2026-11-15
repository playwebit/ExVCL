# ExVCL

**Evidence-Governed Admission for Secure Continual Learning**

ExVCL is a Python framework for evidence-governed admission in continual learning. It places execution-derived behavioural evidence in front of a continual learner and controls which labelled observations are admitted to update that learner.

The framework is designed for settings in which an upstream model supplies labels or pseudo-labels, while a separate execution/telemetry path supplies behavioural evidence. ExVCL's gate uses that evidence, together with the observed label and (in selective mode) the learner's attack score, to decide whether an observation may update the model.

- **GitHub:** https://github.com/playwebit/ExVCL
- **PyPI distribution:** `exvcl`
- **Python import:** `exvcl`
- **License:** Apache License 2.0

> **Research scope.** ExVCL is a research framework, not a turnkey host-security product. Its behaviour depends on the telemetry source, event coverage, calibration data, evidence quality, learner, admission mode, and verification budget. Evaluate it in the target environment before relying on it operationally.

## 1. How ExVCL works

The main processing path is:

1. **Trusted initialization:** initialize the continual learner using trusted labelled data.
2. **Evidence calibration:** fit an `EvidenceModel` using a separate trusted, labelled calibration set of executions.
3. **Streaming:** receive a chunk of feature vectors and observed labels/pseudo-labels.
4. **Evidence verification:** for observations selected by the gate, obtain execution-derived evidence counts and a telemetry-success mask.
5. **Admission decision:** the gate decides which observations may be used for learning.
6. **Continual update:** the bundled `ExVCL` object updates its learner only with admitted observations and the labels returned by the gate.

The gate does not receive ground-truth labels for the incoming stream. `y_obs` is the label supplied to the learning pipeline. Ground truth, where available, should be kept separately for retrospective evaluation.

### Architecture

```text
Trusted labelled calibration executions
                 |
                 v
           EvidenceModel
                 |
                 v
Stream: X + observed labels ---> Admission gate <--- execution evidence
                                      |
                                admit / defer /
                                  reject decision
                                      |
                                      v
                              Continual learner
                                      |
                                updated model
```

## 2. Installation

After the package is published on PyPI:

```bash
python -m pip install exvcl
```

The current 0.2.0 package metadata declares Python 3.9 or later and these dependencies:

- NumPy
- pandas
- scikit-learn
- SciPy

For development from a source checkout:

```bash
git clone https://github.com/playwebit/ExVCL.git
cd ExVCL
python -m venv .venv
```

Activate the environment, then install the project:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

## 3. Quick start: bundled continual learner

The following example demonstrates the library API using already-prepared feature arrays and evidence-count arrays. It does not collect host telemetry itself: the caller must supply evidence produced by an appropriate isolated execution/telemetry pipeline.

```python
import numpy as np
from exvcl import EvidenceModel, ExVCL

# F_cal: trusted calibration evidence, shape (n_calibration, 7)
# y_cal_malicious: 0 for benign and 1 for malicious calibration executions
#
# X_initial: trusted initial feature matrix, shape (n_initial, n_features)
# y_initial: trusted initial labels; 0 is benign and positive values are attack classes
#
# X_chunk: incoming feature matrix, shape (n_chunk, n_features)
# y_observed: incoming labels/pseudo-labels, one per row of X_chunk
# F_chunk: evidence counts for the corresponding observations,
#          shape (n_chunk, 7)

evidence_model = EvidenceModel(
    target_fpr=0.02,
    target_fnr=0.02,
    seed=0,
).fit(F_cal, y_cal_malicious)

model = ExVCL(
    n_features=X_initial.shape[1],
    mode="strict",
    evidence=evidence_model,
    learner="softmax",
    budget=1.0,
    seed=0,
)

# Bootstrap is performed on trusted labelled initial data.
model.bootstrap(X_initial, y_initial)

# verify(idx) must return evidence rows for exactly the requested
# positions, plus a boolean mask indicating successful telemetry.
def verify(idx):
    return F_chunk[idx], np.ones(len(idx), dtype=bool)

result = model.process_chunk(X_chunk, y_observed, verify=verify)

print(result.summary())
print("Admitted:", int(result.admit.sum()), "/", len(result.admit))
```

**Important:** `F_cal`, `X_initial`, `y_initial`, `X_chunk`, `y_observed`, and `F_chunk` above are placeholders for your own data. The example assumes the evidence rows are already aligned with the observations in the chunk. Do not use synthetic or untrusted data as the trusted bootstrap/calibration set in a real evaluation.

## 4. Evidence model

`EvidenceModel` maps seven execution-derived telemetry counts to an evidence verdict. Fit it on a trusted, labelled calibration set that is separate from the stream being protected.

```python
from exvcl import EvidenceModel, Verdict

evidence_model = EvidenceModel(
    target_fpr=0.02,
    target_fnr=0.02,
    support_check=False,
    seed=0,
).fit(F_cal, y_cal_malicious)

batch = evidence_model.verdicts(F_test, ok=telemetry_ok)
print(batch.verdict)
print(batch.p_malicious)
```

### Calibration input contract

- `F_cal` must be a finite numeric array with shape `(n, 7)`.
- `y_cal_malicious` must have one entry per row: `0` for benign and `1` for malicious.
- The default minimum is 10 benign and 10 malicious calibration executions. Smaller or one-sided calibration sets raise `CalibrationError`.
- The seven evidence columns must be in the exact order below.
- `ok` is an optional boolean vector indicating whether telemetry was successfully obtained for each execution. Failed telemetry is marked `UNAVAILABLE`.

### Seven evidence counts

The order is fixed and is also exposed as `exvcl.EVENT_KINDS`:

1. `process_spawn`
2. `shell_exec`
3. `file_write`
4. `sensitive_file_access`
5. `net_connect_out`
6. `priv_change`
7. `syscall_total`

Use the library's parsers/helpers where applicable rather than changing the feature order. The evidence model's `drop` option can exclude selected named counts from fusion and the support check.

### Verdicts

`Verdict` defines the evidence outcomes:

- `BENIGN`
- `MALICIOUS`
- `UNCERTAIN`
- `UNAVAILABLE`

When telemetry is missing or marked unsuccessful, the evidence is `UNAVAILABLE`. If `support_check=True`, evidence sufficiently unfamiliar relative to the calibration executions can also be marked unavailable. The gate's policy determines how verdicts affect admission.

### Save and reload calibration

```python
evidence_model.save("evidence_model.json")

loaded_model = EvidenceModel.load("evidence_model.json")
```

Keep the calibration dataset, configuration, and model artifact versioned with your experiment so that the calibration process can be reproduced.

## 5. Admission modes

The public API supports three modes.

| Mode | Behaviour | Main consideration |
|---|---|---|
| `none` | Accepts all observations; no evidence verification is required | Useful as an accept-all baseline |
| `selective` | Verifies observations selected by model/label triage and may audit a portion of the fast path | Requires attack scores and triage calibration; unverified fast-path observations can be admitted |
| `strict` | Verifies observations subject to the verification budget; observations beyond the budget are deferred | No fast path; lower budget can reduce the amount of data available for learning |

Example:

```python
from exvcl import AdmissionGate

gate = AdmissionGate(
    mode="strict",
    evidence=evidence_model,
    budget=0.30,
    seed=0,
)
```

`budget` can be a fraction of the current chunk (for example, `0.30` for approximately 30%) or an integer number of executions. In strict mode, observations beyond the budget are deferred rather than learned from.

### Selective mode

Selective mode uses the learner's current attack scores to decide which observations need verification. It requires `p_attack` values when using `AdmissionGate` directly. With the bundled `ExVCL` class, the learner supplies these scores automatically.

For a standalone `AdmissionGate`, calibrate the triage threshold using benign calibration scores:

```python
threshold = gate.calibrate_triage(p_attack_benign_calibration)
```

The `ExVCL` wrapper provides the corresponding method:

```python
threshold = model.calibrate_triage(X_benign_calibration)
```

The `audit_rate` reserves part of the verification budget to audit observations otherwise routed to the selective fast path.

## 6. Use your own continual learner

`AdmissionGate` is separate from the bundled learner and can be used with an existing model. The caller remains responsible for updating its learner only with admitted rows.

```python
import numpy as np
from exvcl import AdmissionGate

gate = AdmissionGate(
    mode="strict",
    evidence=evidence_model,
    budget=0.5,
    seed=0,
)

def verify(idx):
    # Return (evidence_counts, telemetry_success_mask) for idx.
    return F_chunk[idx], telemetry_ok[idx]

decision = gate.decide(
    y_obs=y_observed,
    verify=verify,
)

# Apply the admission decision to your own learner.
your_learner.partial_fit(
    X_chunk[decision.admit],
    decision.label[decision.admit],
)
```

For `selective` mode, pass `p_attack=...` to `gate.decide(...)`. The score vector must contain the current model's attack probability/score for each observation in the chunk.

## 7. Evidence supplier / verification callback contract

For `strict` and `selective` modes, provide a callable:

```python
def verify(idx):
    return F, ok
```

- `idx` is an integer array of positions selected for verification in the current chunk.
- `F` must have shape `(len(idx), 7)` and contain evidence counts in `EVENT_KINDS` order.
- `ok` must be a boolean array of shape `(len(idx),)`. `True` means telemetry was successfully obtained for that execution.
- The returned rows must correspond to `idx` in the same order.
- The callback should obtain evidence independently of the observed label and should not use ground truth to decide what evidence to return.
- If collection fails, mark the corresponding `ok` entry `False`; do not fabricate a benign evidence vector.

The library does not itself guarantee that a telemetry collector is isolated, complete, tamper-resistant, or correctly attributed to the observation. Those properties depend on the surrounding collection and execution environment and must be evaluated for the intended deployment.

## 8. Evidence attribution controls

`ChunkVerifier` provides evidence-control variants for controlled experiments:

```python
from exvcl import ChunkVerifier

real = ChunkVerifier(F_chunk, ok=telemetry_ok, control="real")
shuffled = ChunkVerifier(
    F_chunk, ok=telemetry_ok, control="shuffled",
    rng=np.random.default_rng(1),
)
global_random = ChunkVerifier(
    F_chunk, ok=telemetry_ok, control="global_random",
    pool=F_cal,
    rng=np.random.default_rng(1),
)
mixed = ChunkVerifier(
    F_chunk, ok=telemetry_ok, control="mix",
    pool=F_cal,
    q=0.5,
    rng=np.random.default_rng(1),
)
```

Controls:

- `real`: each observation receives its own evidence.
- `shuffled`: evidence is permuted within the requested verification batch.
- `global_random`: evidence is sampled from a calibration evidence pool independently of the observation.
- `mix`: each observation retains its real evidence with probability `q`; otherwise evidence is drawn from the calibration pool (`q=1` is real evidence and `q=0` is the global-random control).

These are experimental controls, not production telemetry collection modes. Use matched streams, budgets, and random seeds when comparing them.

The package also exposes:

- `admitted_poison_share(admit, poisoned)`
- `clean_retention(admit, poisoned)`
- `paired_report(real, control, label="control")`

The `paired_report` helper expects one metric value per independent seed for each condition. It reports a paired comparison; apply the provided `holm` correction across the family of comparisons you report.

## 9. Telemetry helpers and command-line interface

The package exposes:

```python
from exvcl import EVENT_KINDS, counts_from_strace_text

counts = counts_from_strace_text(strace_text)
print(dict(zip(EVENT_KINDS, counts)))
```

There is also a file parser:

```python
from exvcl.telemetry import counts_from_strace_file

counts = counts_from_strace_file("session.strace")
```

The file helper supports plain text and `.gz` input.

After installation, inspect the available CLI commands:

```bash
exvcl --help
```

Common checks and commands:

```bash
exvcl version
exvcl selftest
exvcl selftest --no-engine
exvcl demo --seed 0 --sessions 80 --json demo_report.json
exvcl parse-strace session.strace
```

- `selftest` runs the library tests and, by default, the bundled research engine's tests.
- `selftest --no-engine` skips the engine's own tests.
- `demo` uses synthetic evidence and simulated features. It is a software/pipeline check, **not a research result**.
- `vm-check` records real `strace` telemetry on the local machine and requires `strace` to be installed. Review its output and environment before running it.
- `eval-recording` evaluates a supplied lab recording and session labels.
- `weblab` provides a separate web-attack lab workflow; consult `exvcl weblab --help` and the relevant lab documentation before use.

Some CLI workflows require optional system tools or optional Python dependencies. For example, real VM checks and web-lab collection require `strace`; the web-lab server requires Flask. Install and configure these explicitly in a suitable test environment.

## 10. Experimental practice and reproducibility

For a defensible evaluation:

1. Keep trusted initialization data, evidence-calibration data, and the evaluated stream logically separate.
2. Document how each telemetry row is collected and attributed to its corresponding observation.
3. Record the evidence feature order, collector/version, operating system, calibration settings, admission mode, verification budget, learner configuration, and random seeds.
4. Include accept-all (`none`) and evidence-control conditions where relevant.
5. Report both protection and data-retention effects. A lower admitted-poison share can come with more deferred or rejected clean observations.
6. Keep ground-truth labels out of the live admission path when evaluating pseudo-label admission; use them separately for retrospective scoring.
7. Report missing telemetry, unsupported event families, and evidence blind spots.
8. Do not describe the synthetic `demo` output as empirical validation.

The repository retains `evcl/`, the earlier research-engine namespace used by the framework's implementation and research experiments. In this release, both `exvcl/` and `evcl/` are included in the same `exvcl` distribution because the public package imports the engine modules. Users install one distribution:

```bash
python -m pip install exvcl
```

## 11. API reference

Main public exports from `exvcl`:

| Symbol | Purpose |
|---|---|
| `EvidenceModel` | Fit, apply, save, and load the calibrated evidence model |
| `Verdict` | Evidence verdict constants |
| `CalibrationError` | Error for invalid/insufficient evidence calibration |
| `AdmissionGate` | Admission control independent of the bundled learner |
| `GateResult` | Admission mask, learning labels, routes, verdicts, and summary |
| `ExVCL` | Admission gate combined with a bundled continual learner |
| `make_learner` | Construct a bundled learner (`softmax`, `mlp`, or `forest`) |
| `ChunkVerifier` | Evidence supplier with attribution controls |
| `EVENT_KINDS` | Required seven-feature evidence order |
| `counts_from_strace_text` | Parse a strace text record into evidence counts |
| `admitted_poison_share` | Calculate the share of poisoned observations admitted |
| `clean_retention` | Calculate the share of non-poisoned observations admitted |
| `paired_report` | Paired comparison helper |

## 12. Limitations and responsible use

- ExVCL is not a replacement for a secure execution sandbox, operating-system isolation, endpoint detection, or a production intrusion-detection system.
- Evidence quality depends on the collector, operating system, permissions, event coverage, attribution correctness, and calibration representativeness.
- The seven-count evidence representation may not distinguish every benign and malicious behaviour. An evidence model can be uncertain or unavailable.
- A verification budget constrains how many observations are checked. In strict mode, the remainder are deferred; in selective mode, some observations can take the fast path.
- Calibration and bootstrap data are trusted inputs. Contamination of these sets can undermine the resulting model and admission policy.
- Performance and security claims should be limited to the tested datasets, collectors, configurations, and threat models.

## 13. Citation

If you use ExVCL in academic work, cite the software using [`CITATION.cff`](CITATION.cff). Please also cite the associated research publication when one is available.

## 14. Contributing

Bug reports, reproducibility feedback, documentation improvements, and research extensions are welcome through GitHub Issues and Pull Requests. When proposing changes to admission logic, evidence processing, or collectors, include tests and describe the experimental conditions under which the change was evaluated.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
