"""evcl.checkpoint -- crash-safe checkpointing (works on Google Drive paths)."""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
from typing import Any, Dict, List, Optional


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


class CheckpointManager:
    """Layout: <root>/<experiment>/<config_hash>/{config.json, env.json, trials/*.json, states/*.pkl}.
    Trial results are written once finished; mid-trial states are written atomically with a .prev backup.
    NOTE: state files are pickles -- only load checkpoints you created."""

    def __init__(self, root: str, experiment: str, cfg_hash: str):
        self.run_dir = os.path.join(root, experiment, cfg_hash)
        self.trials_dir = os.path.join(self.run_dir, "trials")
        self.states_dir = os.path.join(self.run_dir, "states")
        for d in (self.trials_dir, self.states_dir):
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _atomic(path: str, data: bytes):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def write_json(self, name: str, obj: Any):
        self._atomic(os.path.join(self.run_dir, name), json.dumps(obj, indent=2, default=str).encode())

    def read_json(self, name: str) -> Optional[Any]:
        p = os.path.join(self.run_dir, name)
        return json.load(open(p)) if os.path.exists(p) else None

    # trials
    def _tp(self, key): return os.path.join(self.trials_dir, _safe(key) + ".json")
    def has_trial(self, key): return os.path.exists(self._tp(key))
    def save_trial(self, key, res): self._atomic(self._tp(key), json.dumps(res, default=str).encode())
    def load_trial(self, key): return json.load(open(self._tp(key)))

    def load_all_trials(self) -> List[Dict]:
        return [json.load(open(os.path.join(self.trials_dir, f)))
                for f in sorted(os.listdir(self.trials_dir)) if f.endswith(".json")]

    # mid-trial state
    def _sp(self, key): return os.path.join(self.states_dir, _safe(key) + ".pkl")

    def save_state(self, key, state):
        p = self._sp(key)
        if os.path.exists(p):
            shutil.copyfile(p, p + ".prev")
        self._atomic(p, pickle.dumps(state, protocol=4))

    def load_state(self, key):
        for p in (self._sp(key), self._sp(key) + ".prev"):
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        return pickle.load(f)
                except Exception:  # corrupt/partial -> try backup
                    continue
        return None

    def drop_state(self, key):
        for p in (self._sp(key), self._sp(key) + ".prev"):
            if os.path.exists(p):
                os.remove(p)


class SeedRegistry:
    """Persistent record of seed usage. (1) A seed used in development may never be used in a final run (and vice
    versa). (2) A FINAL seed may only ever be evaluated under ONE configuration hash: re-running final seeds after
    changing the configuration is refused, which prevents tuning on the final evaluation seeds."""

    def __init__(self, root: str, experiment: str):
        os.makedirs(os.path.join(root, experiment), exist_ok=True)
        self.path = os.path.join(root, experiment, "seed_registry.json")

    def load(self):
        reg = json.load(open(self.path)) if os.path.exists(self.path) else {}
        reg.setdefault("development", [])
        reg.setdefault("final", [])
        reg.setdefault("final_config_hash", {})
        return reg

    def register(self, run_mode: str, seeds, cfg_hash: str = ""):
        reg = self.load()
        other = "final" if run_mode == "development" else "development"
        clash = sorted(set(map(int, seeds)) & set(reg[other]))
        if clash:
            raise RuntimeError(f"seeds {clash} were already used in {other} runs; refusing to reuse them in "
                               f"{run_mode} runs (development/final seed separation)")
        if run_mode == "final" and cfg_hash:
            for sd in map(int, seeds):
                prev = reg["final_config_hash"].get(str(sd))
                if prev and prev != cfg_hash:
                    raise RuntimeError(f"final seed {sd} was already evaluated with configuration {prev}; refusing to "
                                       f"evaluate it with {cfg_hash} (no re-tuning on final seeds)")
                reg["final_config_hash"][str(sd)] = cfg_hash
        reg[run_mode] = sorted(set(reg[run_mode]) | set(map(int, seeds)))
        tmp = self.path + ".tmp"
        json.dump(reg, open(tmp, "w"))
        os.replace(tmp, self.path)
        return reg