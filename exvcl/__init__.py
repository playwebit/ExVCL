"""exvcl -- execution-verified admission control for continual network intrusion detection.

ExVCL is a gate placed in front of an existing continual learner. It decides which labelled observations may update the learner,
using behavioural evidence from an isolated execution (host telemetry) that is collected without access to the label.

Public API
    EvidenceModel     calibrated fusion of the seven evidence counts into a verdict (+ optional support check, modification M1)
    AdmissionGate     the learning admission gate (modes: none, selective, strict) -- works with ANY learner
    ExVCL             gate + a bundled continual learner (softmax, MLP or random forest) in one object
    ChunkVerifier     evidence supplier with the attribution controls (real, shuffled, global_random, mix)
    attribution       admitted-poison share, clean retention, paired statistics

The validated research engine that produced the results of the paper is bundled unchanged as the package `evcl`.
"""
from __future__ import annotations

__version__ = "0.2.0"

from .evidence import CalibrationError, EvidenceModel, Verdict
from .gate import AdmissionGate, GateResult
from .api import ExVCL, make_learner
from .attribution import ChunkVerifier, admitted_poison_share, clean_retention, paired_report
from .telemetry import EVENT_KINDS, counts_from_strace_text

__all__ = ["EvidenceModel", "Verdict", "CalibrationError", "AdmissionGate", "GateResult", "ExVCL", "make_learner",
           "ChunkVerifier", "admitted_poison_share", "clean_retention", "paired_report", "EVENT_KINDS",
           "counts_from_strace_text", "__version__"]
