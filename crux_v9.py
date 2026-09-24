"""
CRUX v9 -- Contrastive, Reference-anchored, Uncertainty-calibrated eXamination
for open-ended video-game glitch detection and temporal grounding
(VideoGlitchBench; baseline and evaluation parity with the official GliDe code,
https://github.com/SandyyyZheng/GliDe, arXiv:2604.07818).

Task (GliDe problem statement): given a gameplay video, output every glitch as a
free-text description plus its time span(s); scored by an LLM judge (0-5) with
Hungarian matching, and by the judge rating times temporal IoU.

Central hypothesis of v9: SELF-REFERENTIAL NORMALITY
-----------------------------------------------------
A game's physics, art style and camera are unknown to a game-disjoint detector,
but the video itself shows what is normal for that game. v9 treats a glitch as a
statistically significant, temporally coherent, and temporally LOCALISABLE
deviation from the video's own norm, and measures each of those three properties
with the VLM's answer-token likelihood rather than with verbalised text.

What v8 got wrong, and what v9 changes (each change has an ablation switch)
----------------------------------------------------------------------------
  1. C1 contextual calibration (grey image) cannot affect rel_z: subtracting a
     per-video constant is removed exactly by the median in C2. The grey image
     is also out of distribution for a VLM, so it estimates the prompt prior but
     not the footage prior. v9 adds a same-content counterfactual:
        C1b  lo_dyn(w) = lo(w) - lo(static(w)),
     where static(w) replaces every frame of w by the per-pixel temporal median
     of w (breakdown point 50%: a transient glitch covering fewer than half of
     the frames is removed while scene, style and HUD are kept). lo_dyn is the
     evidence attributable to the temporal dynamics of the window. This is
     domain-context calibration (Fei et al., ACL 2023) with an in-domain
     counterfactual instead of random tokens.
  2. C3 compared each candidate with the LOWEST-scoring windows of the video.
     Those are usually menus, black or loading frames, so the pair
     contrast was confounded with content. v9 uses MATCHED references: windows
     in the lower half of the detector score (median split, robust while glitches
     cover < 50% of the video) that are nearest in appearance
     (cosine similarity of a Lab thumbnail). This is a matched-pairs design:
     removing shared content lowers the variance of the paired contrast.
  3. C4 (neighbour mean as a logistic feature) double-counted correlated
     evidence and was ad hoc. v9 replaces it with an explicit two-state HMM
     over windows. Emissions are Bayes likelihood ratios obtained from the
     calibrated classifiers,
        log LR_i = logit p_i - logit pi,     pi = unconditional window base rate
     (p_i is the posterior given everything observed for window i, including
     whether it was screened, so the correct prior is the unconditional one).
     Transitions are maximum-likelihood estimates from calibration label
     sequences (add-one smoothing). The initial state is the stationary
     distribution. A single evidence temperature gamma corrects for
     emission dependence and is chosen by out-of-fold log-likelihood (a proper
     scoring rule). The decision uses posterior marginals from forward-backward.
  4. Sub-window grounding asked the model to READ frame labels ('first_frame'),
     which small VLMs do unreliably. v9 grounds frames with the same
     likelihood oracle used for windows: keep only frames 0..k of the onset
     window (the rest are replaced by grey cells) and bisect k. "The prefix
     0..k shows the glitch" is monotone in k, so ceil(log2 n) = 3 calls give
     the first glitch frame. The offset uses the mirror suffix search
     (frames k..n-1). A verified step repairs monotonicity violations. Frame
     times come from the decoded timestamps, not from id / fps.
  5. Candidate generation describes one PEAK window per HMM run, not every
     confirmed window. This lowers the number of generation calls, and the
     entity-anchored resolution then merges runs into events.
  6. Statistics: v8 reported the paired game-cluster BOOTSTRAP "p-value". With
     G clusters a bootstrap percentile p is not a test of H0 and reached 0.0000
     with G = 2 (it is impossible to reject at 0.05 with 2 clusters). v9 tests
     with an exact (2^G <= B) or Monte-Carlo cluster sign-flip permutation test
     (Fisher randomisation; minimum attainable p = 2^(1-G)). Holm uses those p.
     The bootstrap is kept only for confidence intervals of the effect size.

Method (one pass per video)
---------------------------
  C1   lo_abs(w) = logit P(Yes|w) - logit P(Yes|grey)          all windows
  C2   rel_z(w)  = (lo_abs - median_v) / (1.4826 MAD_v)         all windows
  S    screen the top-B windows of each video by lo_abs
  C1b  lo_dyn(w) = logit P(Yes|w) - logit P(Yes|static(w))     screened
  C3   pair_lo(w) = mean_r (lo(A|w,r) - lo(A|r,w)) / 2          screened, matched r
  F    stage-1 ridge-logistic on (lo_abs, rel_z) over all windows; stage-2 on
       all features over screened windows; both game-grouped cross-fitted
  H    HMM forward-backward on gamma * log LR; threshold on posterior by MCC;
       degeneracy check: game-cluster CI of out-of-fold posterior ROC-AUC > 0.5
  D    describe the peak window of each posterior run; resolve runs into events
  G    event-specific presence oracle: galloping window bisection, gap bridging,
       recurrence search, then masked-frame bisection inside boundary windows

Honest scope: nothing here is trained; every claim of benefit is a hypothesis
that the ablation command tests on held-out games. The mock self-test verifies
the MECHANICS (bias cancellation, HMM exactness, bisection correctness, test
validity), not the size of any real-data gain.

Commands
--------
  python crux_v9.py selftest [--glide_repo /path/to/GliDe]
  python crux_v9.py prepare     --split dev
  python crux_v9.py probe       --split dev --limit 60      # cheap go/no-go
  python crux_v9.py calibrate   --calib_splits dev
  python crux_v9.py run         --method crux    --split test
  python crux_v9.py run         --method vanilla --split test
  python crux_v9.py import_glide --batch_report /path/batch_report.json --split test
  python crux_v9.py export      --run_file <runs/..../crux_test.json>
  python crux_v9.py evaluate    --split test --run_files crux_test.json glide_official_test.json
  python crux_v9.py ablate      --split test

Fair-comparison recipe (same backbone, same server, same judge, same videos)
  1. vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --max-logprobs 20
  2. official GliDe, unmodified:
       cd GliDe && python run.py --video-dir <root>/raw --api-base http://localhost:8000/v1 \
                                 --model Qwen/Qwen2.5-VL-7B-Instruct
  3. python crux_v9.py import_glide --split test --batch_report GliDe/data/results/batch_report.json
  4. python crux_v9.py probe --split dev --limit 80      (stop here if the verdict is NO-GO)
  5. python crux_v9.py calibrate --calib_splits dev
  6. python crux_v9.py run --method crux --split test ; run --method vanilla --split test
  7. vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001
  8. python crux_v9.py evaluate --split test --gt_json <official groundtruth.json> \
       --run_files crux_test.json glide_official_test.json vanilla_test.json
     (first file = reference; evaluation uses the intersection of video sets)

The work directory defaults to crux_v8_work so that v8's windows and VLM caches
are reused (the preprocessing fingerprint is unchanged); v9 writes its
calibrations, runs and evaluations under a separate 'v9' sub-tree.

Portions (evaluation prompt, summariser prompt, preprocessing geometry,
evaluator logic) follow the GliDe repository, MIT licence, (c) its authors.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp
from scipy.stats import wilcoxon

_CV2_ERR: Optional[Exception] = None
try:
    import cv2
except Exception as _e:  # pragma: no cover - environment dependent
    cv2 = None
    _CV2_ERR = _e

LOG = logging.getLogger("crux")
EPS = 1e-9
VERSION = "9.0"

# Backbones for the comparison matrix. With backend=openai the same names are
# what the vLLM server must be serving (e.g. `vllm serve Qwen/Qwen2.5-VL-7B-Instruct`).
BACKBONES: Dict[str, Dict[str, str]] = {
    "qwen2.5-vl-3b": {"hf": "Qwen/Qwen2.5-VL-3B-Instruct", "family": "qwen"},
    "qwen2.5-vl-7b": {"hf": "Qwen/Qwen2.5-VL-7B-Instruct", "family": "qwen"},
    "internvl2.5-4b": {"hf": "OpenGVLab/InternVL2_5-4B", "family": "internvl"},
    "internvl2.5-8b": {"hf": "OpenGVLab/InternVL2_5-8B", "family": "internvl"},
}
STAGE1_FEATURES = ("lo_abs", "rel_z")                 # defined on every window
STAGE2_FEATURES = ("lo_abs", "rel_z", "lo_dyn", "pair_lo")  # screened windows only
SCREENED_ONLY = {"lo_dyn", "pair_lo"}
STOPWORDS = {"the", "a", "an", "is", "are", "of", "in", "on", "and", "or", "to", "with",
             "that", "this", "it", "its", "as", "by", "for", "be", "can", "may", "game",
             "player", "character", "when", "which", "from", "at", "while", "into"}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # ---- paths ---------------------------------------------------------------
    root: str = "/home/tahir/VLM-Project/VideoGlitchBench"
    raw_subdir: str = "raw"
    splits_subdir: str = "splits"
    work_subdir: str = "crux_v8_work"   # shared with v8: windows and VLM caches are reused
    metadata_name: str = "annotations.json"
    gt_json: str = ""            # official groundtruth.json; used verbatim by evaluate
    glide_repo: str = ""         # optional clone of the official GliDe repo

    # ---- preprocessing: official GliDe defaults (config.py) ------------------
    fps: float = 4.0
    window_size: int = 8
    jpeg_quality: int = 95       # official writes frames and stitches at q=95
    pair_max_side: int = 2688    # downscaled copies used ONLY by the 2-image C3 query

    # ---- detector backbone -----------------------------------------------------
    backbone: str = "qwen2.5-vl-7b"
    backend: str = "openai"      # 'openai' (vLLM, same stack as GliDe) | 'hf' | 'mock'
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    family: str = "qwen"
    hf_dtype: str = "bfloat16"
    hf_device_map: str = "auto"
    load_in_4bit: bool = False
    max_pixels: int = 0          # 0 = processor/server default (match your GliDe server)
    request_timeout: int = 300
    max_retries: int = 3

    # Generation is ALWAYS fully specified; nothing is inherited from the
    # checkpoint's generation_config (the v7 top_k=1 bug).
    temperature: float = 0.0     # descriptions/summaries: greedy => reproducible
    top_p: float = 1.0
    top_k: int = -1              # -1 / 0 disables top-k
    repetition_penalty: float = 1.0
    desc_max_tokens: int = 384
    summary_max_tokens: int = 512
    top_logprobs: int = 20

    # ---- evaluation judge: official protocol ------------------------------------
    judge_backend: str = "openai"    # 'openai' | 'hf' | 'mock'
    judge_api_base: str = "http://localhost:8001/v1"
    judge_api_key: str = "EMPTY"
    judge_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    judge_temperature: float = 0.3   # Evaluator default in the official repo
    judge_max_tokens: int = 512

    # ---- CRUX -------------------------------------------------------------------
    use_contextual_calibration: bool = True   # C1 prior removal
    use_rel_z: bool = True                    # C2
    use_static_contrast: bool = True          # C1b same-content counterfactual
    use_pairwise: bool = True                 # C3
    use_matched_refs: bool = True             # C3 matched-pairs reference choice
    use_hmm: bool = True                      # H temporal posterior
    use_bisection: bool = True                # G window-level
    use_subframe: bool = True                 # G frame-level (masked likelihood bisection)
    use_recurrence: bool = True
    use_gap_tolerance: bool = True
    use_verified_bisection: bool = True
    use_event_resolution: bool = True
    use_self_verification: bool = True        # drop events the presence oracle never confirms

    n_references: int = 3
    ref_min_gap: int = 2
    pair_budget: int = 8          # screened windows per video that receive C3
    screen_min_windows: int = 1
    presence_margin: float = 0.0  # present iff calibrated log-odds > margin
    gallop_max_steps: int = 6
    gap_tolerance: int = 1
    recurrence_stride: int = 2
    max_intervals_per_event: int = 4
    refine_mode: str = "frame"    # 'window' | 'frame' | 'shrink_guarded'
    shrink_guard: float = 0.5
    max_events_per_video: int = 6
    thumb_size: int = 32          # appearance embedding: thumb_size x thumb_size/2 Lab
    hmm_gammas: str = "0.25,0.5,0.75,1.0,1.25,1.5,2.0"
    hmm_smoothing: float = 1.0    # add-k smoothing of transition counts

    # ---- fusion / selection ------------------------------------------------------
    label_min_overlap: float = 0.25
    fusion_l2: float = 1.0
    cv_folds: int = 5
    decision_threshold: float = 0.5
    fusion: Dict[str, Any] = field(default_factory=dict)
    min_auc_ci_lo: float = 0.5
    min_mcc: float = 0.10
    fail_on_degenerate: bool = True
    select_refine_mode: bool = True
    quantize_seconds: bool = False   # emulate GliDe's int(frame // fps) output

    # ---- statistics / bookkeeping -------------------------------------------------
    seed: int = 20260924
    n_bootstrap: int = 2000
    n_permutations: int = 10000
    limit: Optional[int] = None

    def paths(self) -> Dict[str, str]:
        w = os.path.join(self.root, self.work_subdir)
        b = self.backbone
        return {"raw": os.path.join(self.root, self.raw_subdir),
                "splits": os.path.join(self.root, self.splits_subdir),
                "meta": os.path.join(self.root, self.metadata_name),
                "work": w,
                "windows": os.path.join(w, "windows"),
                "cache": os.path.join(w, "cache", b),
                "judge_cache": os.path.join(w, "cache", "judge",
                                            re.sub(r"[^A-Za-z0-9_.-]", "_", self.judge_model)),
                "runs": os.path.join(w, "v9", "runs", b),
                "params": os.path.join(w, "v9", "params", b),
                "eval": os.path.join(w, "v9", "eval", b)}

    def with_backbone(self, key: str) -> "Config":
        require(key in BACKBONES, f"unknown backbone {key}; have {sorted(BACKBONES)}")
        spec = BACKBONES[key]
        return Config(**{**asdict(self), "backbone": key, "model": spec["hf"],
                         "family": spec["family"]})

    def prep_fingerprint(self) -> str:
        blob = json.dumps({"fps": self.fps, "window_size": self.window_size,
                           "q": self.jpeg_quality, "pair": self.pair_max_side,
                           "v": "glide-official-geometry-1"}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# =============================================================================
# Utilities
# =============================================================================

def setup_logging(work_dir: str, tag: str) -> None:
    os.makedirs(work_dir, exist_ok=True)
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    fh = logging.FileHandler(os.path.join(work_dir, f"crux_v9_{tag}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _repair_json_str(s: str) -> str:
    """Port of GliDe LLMClient._repair_json_str (identical heuristics)."""
    s = re.sub(r"(?<=[{,\s])\s*'([^']+?)'\s*:", r' "\1":', s)
    s = re.sub(r":\s*'([^']*?)'\s*([,}\]])", r': "\1"\2', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


def _try_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair_json_str(s))
    except json.JSONDecodeError:
        return None


def parse_json_from_text(text: str) -> Optional[Dict]:
    """Port of GliDe LLMClient._parse_json_from_text: fences first, then the
    first balanced {...}, then the whole string. Used for every JSON reply so
    that parse failures behave identically for GliDe-style and CRUX calls."""
    if not text:
        return None
    if "```json" in text:
        st = text.find("```json") + 7
        en = text.find("```", st)
        if en != -1:
            r = _try_loads(text[st:en].strip())
            if isinstance(r, dict):
                return r
    if "```" in text:
        st = text.find("```") + 3
        en = text.find("```", st)
        if en != -1:
            r = _try_loads(text[st:en].strip())
            if isinstance(r, dict):
                return r
    m = re.search(r"\{", text)
    if m:
        depth = 0
        for i in range(m.start(), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    r = _try_loads(text[m.start():i + 1])
                    if isinstance(r, dict):
                        return r
                    break
    r = _try_loads(text.strip())
    return r if isinstance(r, dict) else None


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z]+", str(text).lower())
            if t not in STOPWORDS and len(t) > 2}


# =============================================================================
# Interval algebra (used by grounding; evaluation uses the official IoU port)
# =============================================================================

Interval = Tuple[float, float]


def normalize_intervals(ivs: Sequence[Sequence[float]]) -> List[Interval]:
    clean = []
    for iv in ivs or []:
        if iv is None or len(iv) < 2:
            continue
        a, b = float(iv[0]), float(iv[1])
        if b < a:
            a, b = b, a
        if b - a <= 0:
            continue
        clean.append((a, b))
    if not clean:
        return []
    clean.sort()
    out = [clean[0]]
    for a, b in clean[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def intersect_length(A: Sequence[Interval], B: Sequence[Interval]) -> float:
    i = j = 0
    acc = 0.0
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if hi > lo:
            acc += hi - lo
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return acc


def parse_time_nodes(raw: Any) -> List[List[Interval]]:
    """VideoGlitchBench time_nodes -> one list of intervals per bug."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list) or not raw:
        return []
    if len(raw) == 2 and all(isinstance(x, (int, float)) for x in raw):
        return [[(float(raw[0]), float(raw[1]))]]
    out: List[List[Interval]] = []
    for entry in raw:
        if isinstance(entry, list) and len(entry) == 2 and all(
                isinstance(x, (int, float)) for x in entry):
            out.append(normalize_intervals([entry]))
        elif isinstance(entry, list):
            out.append(normalize_intervals(entry))
    return [o for o in out if o]


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class Record:
    vid: str
    video_name: str
    game: str
    genre: str
    video_path: str
    no_bugs: bool
    bugs: List[str]
    time_nodes: List[List[Interval]]
    duration: float = 0.0

    def gt_reports(self) -> List[Dict]:
        return [{"description": d, "spans": self.time_nodes[i]}
                for i, d in enumerate(self.bugs)
                if i < len(self.time_nodes) and self.time_nodes[i]]


def load_records(cfg: Config) -> List[Record]:
    """Accepts both VideoGlitchBench schemas: (A) the official GliDe format
    {video_name, game_name, bugs, time_nodes, no_bugs} and (B) a report list
    [{description, time_nodes|time_nodes_validated}]."""
    p = cfg.paths()
    require(os.path.exists(p["meta"]), f"metadata not found at {p['meta']}")
    rows = read_json(p["meta"])
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("records", []))
    recs: List[Record] = []
    missing = 0
    for r in rows:
        name = (r.get("video_name") or r.get("file") or r.get("video") or r.get("path") or "")
        name = os.path.basename(name) if name else f"{r.get('id') or r.get('video_id')}.mp4"
        if not name.endswith(".mp4"):
            name = name + ".mp4"
        game = str(r.get("game_name") or r.get("game") or "unknown")
        cands = [os.path.join(p["raw"], name), os.path.join(p["raw"], game, name)]
        ap = r.get("path") or r.get("video_path")
        if isinstance(ap, str) and ap.startswith("/"):
            cands.append(ap)
        vp = next((c for c in cands if os.path.exists(c)), None)
        if vp is None:
            import glob as _glob
            hits = _glob.glob(os.path.join(p["raw"], "*", name))
            vp = hits[0] if hits else None
        if vp is None:
            missing += 1
            continue
        bugs: List[str] = []
        tns: List[List[Interval]] = []
        raw_bugs = r.get("bugs")
        if isinstance(raw_bugs, str):
            raw_bugs = _try_loads(raw_bugs) or [raw_bugs]
        if isinstance(raw_bugs, list) and raw_bugs:
            bugs = [str(b) for b in raw_bugs]
            tns = parse_time_nodes(r.get("time_nodes"))
        else:
            for rep in (r.get("reports") or []):
                if not isinstance(rep, dict):
                    continue
                desc = str(rep.get("description", "")).strip()
                ivs = parse_time_nodes(rep.get("time_nodes_validated")
                                       or rep.get("time_nodes")
                                       or rep.get("time_nodes_original") or [])
                if desc and ivs:
                    bugs.append(desc)
                    tns.append(ivs[0])
        explicit = r.get("no_bugs")
        recs.append(Record(
            vid=str(r.get("id") or r.get("video_id") or name.replace(".mp4", "")),
            video_name=name.replace(".mp4", ""), game=game,
            genre=str(r.get("genre", "unknown")), video_path=vp,
            no_bugs=bool(explicit) if explicit is not None else (len(bugs) == 0),
            bugs=bugs, time_nodes=tns))
    require(recs, "no usable records; check raw/ and metadata paths")
    if missing:
        LOG.warning("%d annotated videos have no file on disk and were dropped", missing)
    LOG.info("loaded %d records over %d games", len(recs), len({r.game for r in recs}))
    return recs


def load_splits(cfg: Config, recs: List[Record]) -> Dict[str, List[Record]]:
    p = cfg.paths()
    require(os.path.isdir(p["splits"]), f"splits dir missing: {p['splits']}")
    index = {r.vid: r for r in recs}
    by_name = {r.video_name: r for r in recs}
    out: Dict[str, List[Record]] = {}
    for name in ("train", "dev", "test"):
        hits = sorted(f for f in os.listdir(p["splits"]) if f.startswith(name))
        require(hits, f"no split file for '{name}' in {p['splits']}")
        with open(os.path.join(p["splits"], hits[0])) as f:
            raw = f.read().strip()
        ids = _try_loads(raw)
        if isinstance(ids, dict):
            ids = ids.get("ids", [])
        if not isinstance(ids, list):
            ids = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        sel = []
        for i in ids:
            k = str(i).replace(".mp4", "")
            rr = index.get(k) or by_name.get(k)
            if rr is not None:
                sel.append(rr)
        require(sel, f"split '{name}' resolved to 0 records -- id format mismatch")
        out[name] = sel
        LOG.info("split %-5s: %4d videos, %3d games", name, len(sel), len({r.game for r in sel}))
    tr, te = {r.game for r in out["train"]}, {r.game for r in out["test"]}
    LOG.info("train/test game overlap: %d of %d test games", len(tr & te), len(te))
    return out


def window_labels(rec: Record, wins: Sequence["Window"], min_overlap: float) -> np.ndarray:
    gt = normalize_intervals([iv for spans in rec.time_nodes for iv in spans])
    y = np.zeros(len(wins), np.int64)
    for i, w in enumerate(wins):
        if w.end > w.start:
            y[i] = int(intersect_length([(w.start, w.end)], gt) / (w.end - w.start)
                       >= min_overlap)
    return y


# =============================================================================
# Preprocessing -- geometry of the official GliDe VideoPreprocessor
#   * time-based sampling: t = 0, 1/fps, 2/fps, ... while t < duration, seek+read
#   * non-overlapping windows of `window_size` frames (last window may be short)
#   * stitch: native resolution, 2 rows x ceil(n/2) cols, GLOBAL frame index
#     label "#<id>" on a black box, JPEG quality 95
# =============================================================================

@dataclass
class Window:
    idx: int
    start: float
    end: float
    stitch_path: str
    frame_ids: List[int]
    frame_times: List[float]
    pair_path: str = ""


def extract_frames_official(video_path: str, fps: float, quality: int = 95
                            ) -> Tuple[List[np.ndarray], List[float], float]:
    require(cv2 is not None, f"opencv is required ({_CV2_ERR})")
    cap = cv2.VideoCapture(video_path)
    require(cap.isOpened(), f"cannot open video {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / native if native > 0 else 0.0
    interval = 1.0 / fps
    targets, t = [], 0.0
    while t < duration:                 # float accumulation, exactly as official
        targets.append(t)
        t += interval
    frames, times = [], []
    for tt in targets:
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        # Official GliDe writes each frame as JPEG (q=95) and stitches from the
        # decoded files; the round trip reproduces those pixels exactly.
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frames.append(cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else fr)
        times.append(tt)
    cap.release()
    require(frames, f"decoded 0 frames from {video_path}")
    return frames, times, float(duration)


def stitch_official(frames: Sequence[np.ndarray], ids: Sequence[int]) -> np.ndarray:
    n = len(frames)
    require(n > 0, "no frames to stitch")
    fh, fw = frames[0].shape[:2]
    cols, rows = (n + 1) // 2, 2
    canvas = np.zeros((fh * rows, fw * cols, 3), dtype=np.uint8)
    for k, (fr, fid) in enumerate(zip(frames, ids)):
        r, c = k // cols, k % cols
        y0, x0 = r * fh, c * fw
        if fr.shape[:2] != (fh, fw):
            fr = cv2.resize(fr, (fw, fh), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + fh, x0:x0 + fw] = fr
        _draw_label(canvas, x0, y0, fid)
    return canvas


def _draw_label(canvas: np.ndarray, x0: int, y0: int, fid: int) -> None:
    font, fs, th, pad = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2, 5
    label = f"#{fid}"
    (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
    cv2.rectangle(canvas, (x0, y0), (x0 + tw + pad * 2, y0 + tht + pad * 2), (0, 0, 0), -1)
    cv2.putText(canvas, label, (x0 + pad, y0 + tht + pad), font, fs, (255, 255, 255), th)


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def prepare_record(rec: Record, cfg: Config, force: bool = False) -> List[Window]:
    out_dir = os.path.join(cfg.paths()["windows"], cfg.prep_fingerprint(), rec.vid)
    manifest = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest) and not force:
        data = read_json(manifest)
        rec.duration = float(data["duration"])
        return [Window(**w) for w in data["windows"]]
    frames, times, duration = extract_frames_official(rec.video_path, cfg.fps,
                                                      cfg.jpeg_quality)
    rec.duration = duration
    os.makedirs(out_dir, exist_ok=True)
    wins: List[Window] = []
    for wi, s in enumerate(range(0, len(frames), cfg.window_size)):
        e = min(s + cfg.window_size, len(frames))
        ids = list(range(s, e))
        st = stitch_official(frames[s:e], ids)
        sp = os.path.join(out_dir, f"window_{wi:04d}_stitched.jpg")
        require(cv2.imwrite(sp, st, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]),
                f"failed to write {sp}")
        pp = os.path.join(out_dir, f"window_{wi:04d}_pair.jpg")
        cv2.imwrite(pp, _downscale(st, cfg.pair_max_side), [cv2.IMWRITE_JPEG_QUALITY, 92])
        t0 = times[s]
        t1 = min(duration, times[e - 1] + 1.0 / cfg.fps)
        wins.append(Window(wi, float(t0), float(max(t1, t0 + EPS)), sp, ids,
                           [float(x) for x in times[s:e]], pp))
    require(wins, f"created 0 windows for {rec.vid}")
    # Content-free probe: a uniform grey image with the stitched geometry.
    h, w = cv2.imread(wins[0].stitch_path).shape[:2]
    cf = os.path.join(out_dir, "content_free.jpg")
    cv2.imwrite(cf, np.full((h, w, 3), 128, np.uint8), [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    atomic_write_json(manifest, {"duration": duration, "fps": cfg.fps,
                                 "windows": [asdict(x) for x in wins]})
    return wins


def content_free_path(wins: Sequence[Window]) -> str:
    return os.path.join(os.path.dirname(wins[0].stitch_path), "content_free.jpg")


# =============================================================================
# Derived images. All are built lazily from the stitched window and inherit
# its exact geometry, so the preprocessing fingerprint (and v8 caches) is kept.
#   static(w):  every cell = per-pixel temporal median of the window's frames
#   keep(w,a,b): cells outside frame positions [a, b] are grey (labels kept)
# =============================================================================

def _split_cells(img: np.ndarray, n: int) -> Tuple[List[np.ndarray], int, int, int]:
    cols = (n + 1) // 2
    fh, fw = img.shape[0] // 2, img.shape[1] // cols
    cells = [img[(k // cols) * fh:(k // cols + 1) * fh, (k % cols) * fw:(k % cols + 1) * fw]
             for k in range(n)]
    return cells, fh, fw, cols


def _derived_path(win: Window, suffix: str) -> str:
    base = win.stitch_path
    stem = base[:-len("_stitched.jpg")] if base.endswith("_stitched.jpg") else base.rsplit(".", 1)[0]
    return f"{stem}_{suffix}.jpg"


def static_counterfactual_path(win: Window, quality: int = 95) -> str:
    """Same content, no dynamics. The per-pixel median over the window's frames
    removes any transient event that occupies fewer than half of the frames
    (breakdown point of the median) and keeps scene, style, HUD and camera."""
    out = _derived_path(win, "static")
    if os.path.exists(out):
        return out
    img = cv2.imread(win.stitch_path)
    require(img is not None, f"cannot read {win.stitch_path}")
    n = len(win.frame_ids)
    cells, fh, fw, cols = _split_cells(img, n)
    med = np.median(np.stack(cells).astype(np.float32), axis=0).round().astype(np.uint8)
    canvas = np.zeros_like(img)
    for k, fid in enumerate(win.frame_ids):
        y0, x0 = (k // cols) * fh, (k % cols) * fw
        canvas[y0:y0 + fh, x0:x0 + fw] = med
        _draw_label(canvas, x0, y0, fid)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return out


def masked_window_path(win: Window, a: int, b: int, quality: int = 95) -> str:
    """Keep frame POSITIONS a..b (inclusive) of the window, grey out the rest.
    The grid geometry and the frame labels are unchanged, so the model still
    sees where in time the kept frames sit."""
    n = len(win.frame_ids)
    require(0 <= a <= b < n, f"bad keep range {a}..{b} for {n} frames")
    out = _derived_path(win, f"keep_{win.frame_ids[a]}_{win.frame_ids[b]}")
    if os.path.exists(out):
        return out
    img = cv2.imread(win.stitch_path)
    require(img is not None, f"cannot read {win.stitch_path}")
    _, fh, fw, cols = _split_cells(img, n)
    canvas = img.copy()
    for k, fid in enumerate(win.frame_ids):
        if a <= k <= b:
            continue
        y0, x0 = (k // cols) * fh, (k % cols) * fw
        canvas[y0:y0 + fh, x0:x0 + fw] = 128
        _draw_label(canvas, x0, y0, fid)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return out


def window_embedding(path: str, size: int = 32) -> np.ndarray:
    """Zero-mean, unit-norm CIE-Lab thumbnail of a stitched window; the cosine
    similarity of two embeddings is a cheap content-similarity measure.
    Computed once per video from the downscaled pair image (no memo: a global
    cache would grow with the number of windows in the benchmark)."""
    h = max(1, size // 2)
    img = cv2.imread(path) if cv2 is not None and os.path.exists(path) else None
    if img is None:
        return np.zeros(size * h * 3, np.float32)
    th = cv2.resize(img, (size, h), interpolation=cv2.INTER_AREA)
    v = cv2.cvtColor(th, cv2.COLOR_BGR2LAB).astype(np.float32).ravel()
    v = v - v.mean()
    nv = float(np.linalg.norm(v))
    return v / nv if nv > 0 else v


_PREP_MEMO: Dict[Tuple[str, str], List[Window]] = {}


def prepare_split(recs: Sequence[Record], cfg: Config) -> Dict[str, List[Window]]:
    fp, out, t0 = cfg.prep_fingerprint(), {}, time.time()
    for i, r in enumerate(recs):
        key = (fp, r.vid)
        if key not in _PREP_MEMO:
            _PREP_MEMO[key] = prepare_record(r, cfg)
        out[r.vid] = _PREP_MEMO[key]
        if r.duration <= 0 and out[r.vid]:
            r.duration = out[r.vid][-1].end
        if (i + 1) % 100 == 0:
            LOG.info("prepared %d/%d videos (%.0f s)", i + 1, len(recs), time.time() - t0)
    return out


# =============================================================================
# VLM backends. Two primitives only:
#   generate(system, user, images) -> text          (fully specified sampling)
#   score(system, user, images, options) -> {option: normalised log-prob}
# =============================================================================

@dataclass
class Budget:
    calls: int = 0
    model_calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprob_truncations: int = 0
    by_stage: Dict[str, int] = field(default_factory=dict)


def _file_sig(path: str) -> str:
    ap = os.path.abspath(path)
    try:
        st = os.stat(ap)
        return sha1(f"{ap}:{st.st_size}:{int(st.st_mtime)}")
    except OSError:
        return sha1(ap)


def build_generation_kwargs(cfg: Config, max_tokens: int) -> Dict[str, Any]:
    """Every sampling knob is explicit. v7 omitted top_k and silently inherited
    top_k=1 from the Qwen2.5-VL generation_config, turning every 'sample' into
    the greedy decode."""
    do_sample = cfg.temperature > 0
    kw: Dict[str, Any] = {"max_new_tokens": int(max_tokens), "do_sample": do_sample,
                          "repetition_penalty": float(cfg.repetition_penalty)}
    if do_sample:
        kw.update({"temperature": float(cfg.temperature), "top_p": float(cfg.top_p),
                   "top_k": int(cfg.top_k) if cfg.top_k and cfg.top_k > 0 else 0})
    else:
        kw.update({"temperature": 1.0, "top_p": 1.0, "top_k": 0})
    return kw


def normalise_option_logprobs(raw: Dict[str, float]) -> Dict[str, float]:
    keys = list(raw)
    z = float(logsumexp([raw[k] for k in keys]))
    return {k: float(raw[k] - z) for k in keys}


def binary_log_odds(lp: Dict[str, float], pos: str, neg: str) -> float:
    return float(lp[pos] - lp[neg])


class VLMBase:
    name = "base"
    needs_pixels = True

    def __init__(self, cfg: Config, cache_dir: str):
        self.cfg = cfg
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.budget = Budget()

    # ---- caching ---------------------------------------------------------------
    def _key(self, kind: str, system: str, user: str, images: Sequence[str],
             extra: Dict) -> str:
        return sha1(json.dumps({"k": kind, "b": self.name, "m": self.cfg.model,
                                "s": system, "u": user, "x": extra,
                                "px": self.cfg.max_pixels,
                                "i": [_file_sig(p) for p in images]}, sort_keys=True))

    def _cached(self, key: str) -> Optional[Any]:
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        return read_json(path)["v"] if os.path.exists(path) else None

    def _store(self, key: str, value: Any) -> None:
        atomic_write_json(os.path.join(self.cache_dir, key[:2], key + ".json"), {"v": value})

    def _count(self, stage: str, cached: bool) -> None:
        self.budget.calls += 1
        self.budget.by_stage[stage] = self.budget.by_stage.get(stage, 0) + 1
        if cached:
            self.budget.cache_hits += 1
        else:
            self.budget.model_calls += 1

    # ---- public ---------------------------------------------------------------
    def generate(self, system: str, user: str, images: Sequence[str] = (),
                 max_tokens: int = 512, stage: str = "other") -> str:
        extra = build_generation_kwargs(self.cfg, max_tokens)
        key = self._key("gen", system, user, images, extra)
        hit = self._cached(key)
        if hit is not None:
            self._count(stage, True)
            return hit
        out = self._generate(system, user, images, max_tokens)
        self._count(stage, False)
        self._store(key, out)
        return out

    def generate_json(self, system: str, user: str, images: Sequence[str] = (),
                      max_tokens: int = 512, stage: str = "other",
                      retries: int = 1) -> Optional[Dict]:
        for a in range(retries + 1):
            hint = "" if a == 0 else "\nReturn ONLY a single valid JSON object."
            obj = parse_json_from_text(self.generate(system, user + hint, images,
                                                     max_tokens, stage))
            if obj is not None:
                return obj
        return None

    def score(self, system: str, user: str, images: Sequence[str],
              options: Dict[str, Sequence[str]], stage: str = "other") -> Dict[str, float]:
        extra = {"opts": {k: list(v) for k, v in sorted(options.items())}}
        key = self._key("score", system, user, images, extra)
        hit = self._cached(key)
        if hit is not None:
            self._count(stage, True)
            return hit
        raw = self._score(system, user, images, options)
        out = normalise_option_logprobs(raw)
        self._count(stage, False)
        self._store(key, out)
        return out

    def _generate(self, system, user, images, max_tokens) -> str:
        raise NotImplementedError

    def _score(self, system, user, images, options) -> Dict[str, float]:
        raise NotImplementedError


def _variants(word: str) -> List[str]:
    return list(dict.fromkeys([word, word.lower(), word.upper(), word.capitalize()]))


def match_top_logprobs(top: Sequence[Dict], options: Dict[str, Sequence[str]]
                       ) -> Tuple[Dict[str, float], bool]:
    """Collapse an OpenAI-style top_logprobs list onto answer options. An option
    absent from the list gets the smallest listed log-prob, which is an UPPER
    bound on its true value; the second return flags that truncation."""
    lps = [float(t["logprob"]) for t in top]
    floor = min(lps) if lps else -30.0
    out, truncated = {}, False
    for opt, words in options.items():
        wanted = {w.strip().lower() for word in words for w in _variants(word)}
        got = [float(t["logprob"]) for t in top
               if str(t.get("token", "")).strip().lower() in wanted]
        if got:
            out[opt] = float(logsumexp(got))
        else:
            out[opt] = floor
            truncated = True
    return out, truncated


class OpenAIVLM(VLMBase):
    """OpenAI-compatible client (vLLM). Same serving stack as official GliDe,
    so detector capacity and image handling are identical by construction."""
    name = "openai"

    def _post(self, payload: Dict) -> Dict:
        import requests
        hdr = {"Content-Type": "application/json"}
        if self.cfg.api_key and self.cfg.api_key != "EMPTY":
            hdr["Authorization"] = f"Bearer {self.cfg.api_key}"
        if self.cfg.max_pixels > 0:
            payload["mm_processor_kwargs"] = {"max_pixels": int(self.cfg.max_pixels)}
        last = None
        for a in range(self.cfg.max_retries):
            try:
                r = requests.post(f"{self.cfg.api_base.rstrip('/')}/chat/completions",
                                  headers=hdr, json=payload, timeout=self.cfg.request_timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # pragma: no cover - network
                last = e
                time.sleep(2 ** a)
        raise RuntimeError(f"VLM request failed after {self.cfg.max_retries} attempts: {last}")

    @staticmethod
    def _content(user: str, images: Sequence[str]) -> Any:
        if not images:
            return user
        parts: List[Dict] = [{"type": "text", "text": user}]
        for p in images:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return parts

    def _generate(self, system, user, images, max_tokens) -> str:
        g = build_generation_kwargs(self.cfg, max_tokens)
        payload = {"model": self.cfg.model, "max_tokens": int(max_tokens),
                   "temperature": float(self.cfg.temperature),
                   "top_p": g["top_p"], "top_k": g["top_k"] if g["top_k"] > 0 else -1,
                   "repetition_penalty": g["repetition_penalty"],
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": self._content(user, images)}]}
        d = self._post(payload)
        u = d.get("usage") or {}
        self.budget.prompt_tokens += int(u.get("prompt_tokens", 0) or 0)
        self.budget.completion_tokens += int(u.get("completion_tokens", 0) or 0)
        return str(d["choices"][0]["message"]["content"] or "").strip()

    def _score(self, system, user, images, options) -> Dict[str, float]:
        payload = {"model": self.cfg.model, "max_tokens": 1, "temperature": 0.0,
                   "logprobs": True, "top_logprobs": int(self.cfg.top_logprobs),
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": self._content(user, images)}]}
        d = self._post(payload)
        u = d.get("usage") or {}
        self.budget.prompt_tokens += int(u.get("prompt_tokens", 0) or 0)
        content = ((d["choices"][0].get("logprobs") or {}).get("content") or [])
        require(content, "server returned no logprobs; start vLLM with logprobs enabled "
                         "(--max-logprobs >= top_logprobs)")
        out, trunc = match_top_logprobs(content[0].get("top_logprobs") or [], options)
        self.budget.logprob_truncations += int(trunc)
        return out


class HFVLM(VLMBase):
    """Local transformers backend (Qwen2.5-VL family). InternVL's remote-code
    .chat() hides the logits, so InternVL is supported through vLLM only."""
    name = "hf"

    def __init__(self, cfg: Config, cache_dir: str):
        super().__init__(cfg, cache_dir)
        require(cfg.family == "qwen", "the hf backend supports the qwen family only; "
                                      "serve InternVL with vLLM and use --backend openai")
        self._model = None
        self._proc = None
        self._opt_ids: Dict[Tuple[str, ...], List[int]] = {}

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLM
        except ImportError:  # pragma: no cover
            from transformers import Qwen2VLForConditionalGeneration as VLM
        kw: Dict[str, Any] = {"device_map": self.cfg.hf_device_map}
        if self.cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = getattr(torch, self.cfg.hf_dtype)
        self._model = VLM.from_pretrained(self.cfg.model, **kw).eval()
        pk = {"max_pixels": int(self.cfg.max_pixels)} if self.cfg.max_pixels > 0 else {}
        self._proc = AutoProcessor.from_pretrained(self.cfg.model, **pk)
        LOG.info("loaded %s (hf); checkpoint generation_config is IGNORED: %s",
                 self.cfg.model, self._model.generation_config.to_diff_dict())

    def _inputs(self, system, user, images):
        from PIL import Image
        pil = [Image.open(p).convert("RGB") for p in images]
        content = [{"type": "text", "text": user}] + [{"type": "image"} for _ in pil]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self._proc(text=[text], images=pil or None, return_tensors="pt",
                          padding=True).to(self._model.device)

    def _generate(self, system, user, images, max_tokens) -> str:
        import torch
        from transformers import GenerationConfig
        self._lazy()
        inp = self._inputs(system, user, images)
        gc = GenerationConfig(**build_generation_kwargs(self.cfg, max_tokens),
                              eos_token_id=self._model.generation_config.eos_token_id,
                              pad_token_id=self._model.generation_config.pad_token_id)
        with torch.no_grad():
            out = self._model.generate(**inp, generation_config=gc)
        gen = out[0][inp["input_ids"].shape[1]:]
        self.budget.prompt_tokens += int(inp["input_ids"].shape[1])
        self.budget.completion_tokens += int(gen.shape[0])
        return self._proc.decode(gen, skip_special_tokens=True).strip()

    def _first_ids(self, words: Sequence[str]) -> List[int]:
        key = tuple(words)
        if key not in self._opt_ids:
            tok = self._proc.tokenizer
            ids = set()
            for w in words:
                for v in _variants(w):
                    for s in (v, " " + v):
                        t = tok.encode(s, add_special_tokens=False)
                        if t:
                            ids.add(int(t[0]))
            self._opt_ids[key] = sorted(ids)
        return self._opt_ids[key]

    def _score(self, system, user, images, options) -> Dict[str, float]:
        import torch
        self._lazy()
        inp = self._inputs(system, user, images)
        with torch.no_grad():
            # Only the last position is needed; materialising logits for ~10^4
            # visual tokens x 152k vocab would cost several GB.
            out = None
            for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
                try:
                    out = self._model(**inp, **kw)
                    break
                except TypeError:
                    continue
            logits = out.logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        self.budget.prompt_tokens += int(inp["input_ids"].shape[1])
        return {o: float(logsumexp(lp[self._first_ids(w)])) for o, w in options.items()}


class MockVLM(VLMBase):
    """Scripted backend for offline tests. oracle(kind, system, user, images, options)
    returns text for kind='gen' and a raw {option: logprob} dict for kind='score'."""
    name = "mock"
    needs_pixels = False

    def __init__(self, cfg: Config, cache_dir: str, oracle: Callable):
        super().__init__(cfg, cache_dir)
        self.oracle = oracle

    def _cached(self, key):      # never cache in tests
        return None

    def _store(self, key, value):
        return None

    def _generate(self, system, user, images, max_tokens) -> str:
        return str(self.oracle("gen", system, user, list(images), None))

    def _score(self, system, user, images, options) -> Dict[str, float]:
        return dict(self.oracle("score", system, user, list(images), options))


def make_vlm(cfg: Config, oracle: Optional[Callable] = None) -> VLMBase:
    cache = cfg.paths()["cache"]
    if cfg.backend == "openai":
        return OpenAIVLM(cfg, cache)
    if cfg.backend == "hf":
        return HFVLM(cfg, cache)
    if cfg.backend == "mock":
        require(oracle is not None, "mock backend needs an oracle")
        return MockVLM(cfg, cache, oracle)
    raise ValueError(f"unknown backend {cfg.backend}")


# =============================================================================
# Prompts
# =============================================================================

_GLITCH_DEF = (
    "A glitch is a visible defect the developers did not intend: objects or characters "
    "clipping through geometry, floating without support, being launched or flung, "
    "teleporting, stretching or deforming, T-posing or frozen/broken animation while "
    "moving, missing or corrupted textures, flickering, lighting or shadow errors, AI "
    "stuck in place, physically impossible motion. Stylised art, intended abilities, "
    "menus, HUD, cut-scenes and ordinary camera motion are NOT glitches.")

P_DETECT_SYS = (
    "You are a video game QA expert. The image is a stitched sequence of consecutive "
    "gameplay frames, each labelled '#<frame index>' in temporal order (left to right, "
    "top row then bottom row). " + _GLITCH_DEF)
P_PAIR_SYS = (
    "You are a video game QA expert comparing two stitched windows taken from the SAME "
    "gameplay video. Each window is a sequence of consecutive frames labelled "
    "'#<frame index>' in temporal order. Both windows share the game's art style, camera "
    "and mechanics, so judge only what differs between them. " + _GLITCH_DEF)
P_PRESENCE_SYS = (
    "You are a video game QA expert. You are given one stitched window of consecutive "
    "gameplay frames and the description of one specific glitch. Decide whether that "
    "same glitch, on that same entity, is visibly occurring in this window. Being "
    "plausible is not enough; it must be observable.")
P_DESC_SYS = (
    "You are a video game QA analyst writing one line of a bug report. The image is a "
    "stitched sequence of consecutive gameplay frames labelled '#<frame index>'. "
    + _GLITCH_DEF + " Name the affected object or character by its visual appearance, "
    "state exactly what is abnormal, and where in the scene it happens, in 1-3 sentences. "
    "No frame numbers or timestamps. Reply with one JSON object only, keys: description, "
    "entity, glitch_type.")
P_EAR_SYS = (
    "You consolidate per-window glitch observations from one gameplay video into distinct "
    "glitch EVENTS. Two observations belong to the same event if they concern the same "
    "entity (allowing for wording variance) and the same kind of anomaly, even if far "
    "apart in time. Different entities, or different anomaly kinds on the same entity, "
    "are different events. Reply with one JSON object only, key 'events': list of "
    "{member_ids (list of int), canonical_entity, glitch_type, description}.")

# Verbatim from GliDe summarizer/system_prompt.txt (MIT). {placeholders} filled by format().
P_SUMMARY_TEMPLATE = (
    "You are a video game glitch analyst. Your task is to summarize multiple fragmented "
    "glitch descriptions into a single, clear, and coherent description.\n\n"
    "## Input Descriptions:\n{descriptions}\n\n## Glitch Category: {category}\n"
    "## Glitch Subtype: {subtype}\n## Time Range: {time_range}\n\n## Instructions:\n"
    "1. Read all the fragmented descriptions carefully\n"
    "2. Identify the core glitch phenomenon being described\n"
    "3. Write a single, coherent paragraph that:\n"
    "   - Clearly describes what the glitch is\n"
    "   - Mentions the visual/behavioral anomaly observed\n"
    "   - Is concise but complete (2-4 sentences)\n"
    "   - Does NOT include frame numbers or technical details\n"
    "   - Does NOT include JSON formatting or code blocks\n"
    "   - Uses natural, descriptive language\n\n"
    "## Output:\nWrite ONLY the summarized description paragraph, nothing else.")

P_VANILLA_SYS = (
    "You are a video game QA expert. Watch the gameplay windows and report every glitch: "
    "for each, a natural-language description and its start/end time in seconds. Reply "
    "with one JSON object only, keys: no_bugs (bool), bugs (list of {description, "
    "time_nodes: [[start,end], ...]}).")

YES_NO = {"yes": ["Yes"], "no": ["No"]}
A_B = {"A": ["A"], "B": ["B"]}


def detect_user(game: str) -> str:
    return (f"Game: {game}.\nDoes this window show a gameplay glitch? "
            f"Answer with exactly one word: Yes or No.")


def pair_user(game: str) -> str:
    return (f"Game: {game}.\nThe first image is window A and the second image is window B. "
            f"Which window is more likely to contain a genuine gameplay glitch? "
            f"Answer with exactly one letter: A or B.")


def presence_user(game: str, desc: str, entity: str) -> str:
    return (f"Game: {game}.\nGlitch: {desc}\nAffected entity: {entity}\n"
            f"Is this same glitch visibly occurring in this window? "
            f"Answer with exactly one word: Yes or No.")


# =============================================================================
# C1-C4: likelihood evidence
# =============================================================================

def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 MAD). 1.4826 makes MAD a consistent estimator of
    sigma under normality; the floor keeps a flat video from exploding."""
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return (x - med) / max(mad, 0.25)


def debiased_pair_log_odds(lo_first: float, lo_swapped: float) -> float:
    """lo_first = logit P(A | c as A, r as B) = b + d; lo_swapped = logit P(A |
    r as A, c as B) = b - d. Returns d, the position-free preference for c."""
    return 0.5 * (float(lo_first) - float(lo_swapped))


def choose_references(lo_abs: np.ndarray, c: int, m: int, gap: int) -> List[int]:
    """v8 rule (ablation 'no_matched_refs'): the m lowest-scoring windows."""
    order = np.argsort(lo_abs, kind="mergesort")
    return [int(j) for j in order if abs(int(j) - c) >= gap][:m]


def choose_matched_references(lo_abs: np.ndarray, emb: np.ndarray, c: int, m: int,
                              gap: int) -> List[int]:
    """Matched-pairs reference selection. Eligible: temporal gap >= gap and a
    detector score at or below the video median (the 'normal half'; valid while
    glitches cover < 50% of the video, which calibration logs). Among eligible
    windows pick the m most similar in appearance to c, so the paired contrast
    cancels shared content. If fewer than m are eligible, fall back to the
    lowest-scoring windows (v8 rule)."""
    n = int(lo_abs.size)
    med = float(np.median(lo_abs))
    elig = [j for j in range(n) if abs(j - c) >= gap and lo_abs[j] <= med]
    sim = emb @ emb[c] if emb.size else np.zeros(n)
    elig.sort(key=lambda j: (-float(sim[j]), float(lo_abs[j]), j))
    out = elig[:m]
    if len(out) < m:
        out += [j for j in choose_references(lo_abs, c, n, gap) if j not in out][:m - len(out)]
    return [int(j) for j in out]


def screen_indices(lo_abs: np.ndarray, budget: int, min_windows: int) -> List[int]:
    n = int(lo_abs.size)
    k = int(clamp(budget, min(min_windows, n), n))
    return sorted(int(i) for i in np.argsort(-lo_abs, kind="mergesort")[:k])


@dataclass
class VideoEvidence:
    vid: str
    lo_raw: List[float]
    lo_cf: float
    lo_abs: List[float]
    rel_z: List[float]
    screened: List[int]
    lo_dyn: Dict[int, float]
    pair_lo: Dict[int, float]
    refs: Dict[int, List[int]]

    def features(self, i: int) -> Dict[str, float]:
        return {"lo_abs": self.lo_abs[i], "rel_z": self.rel_z[i],
                "lo_dyn": float(self.lo_dyn.get(i, 0.0)),
                "pair_lo": float(self.pair_lo.get(i, 0.0))}


def collect_video_evidence(vlm: VLMBase, rec: Record, wins: Sequence[Window],
                           cfg: Config) -> VideoEvidence:
    usr = detect_user(rec.game)

    def lo_of(path: str, stage: str) -> float:
        return binary_log_odds(vlm.score(P_DETECT_SYS, usr, [path], YES_NO, stage), "yes", "no")

    lo_raw = np.array([lo_of(w.stitch_path, "c1_detect") for w in wins])
    lo_cf = 0.0
    if cfg.use_contextual_calibration:
        cf = content_free_path(wins) if vlm.needs_pixels else "__content_free__"
        lo_cf = lo_of(cf, "c1_calib")
    lo_abs = lo_raw - lo_cf
    rel = robust_z(lo_abs)
    screened = screen_indices(lo_abs, cfg.pair_budget, cfg.screen_min_windows)
    lo_dyn: Dict[int, float] = {}
    if cfg.use_static_contrast:
        # lo_cf cancels in the difference, so the raw log-odds are used.
        for c in screened:
            lo_dyn[c] = float(lo_raw[c] - lo_of(static_counterfactual_path(wins[c]), "c1b_static"))
    pair_lo: Dict[int, float] = {}
    refs: Dict[int, List[int]] = {}
    if cfg.use_pairwise:
        pu = pair_user(rec.game)
        emb = (np.stack([window_embedding(w.pair_path or w.stitch_path, cfg.thumb_size)
                         for w in wins])
               if cfg.use_matched_refs else np.zeros(0))
        for c in screened:
            rs = (choose_matched_references(lo_abs, emb, c, cfg.n_references, cfg.ref_min_gap)
                  if cfg.use_matched_refs
                  else choose_references(lo_abs, c, cfg.n_references, cfg.ref_min_gap))
            refs[c] = rs
            ds = []
            for r in rs:
                a, b = wins[c].pair_path or wins[c].stitch_path, wins[r].pair_path or wins[r].stitch_path
                l1 = binary_log_odds(vlm.score(P_PAIR_SYS, pu, [a, b], A_B, "c3_pair"), "A", "B")
                l2 = binary_log_odds(vlm.score(P_PAIR_SYS, pu, [b, a], A_B, "c3_pair"), "A", "B")
                ds.append(debiased_pair_log_odds(l1, l2))
            pair_lo[c] = float(np.mean(ds)) if ds else 0.0
    return VideoEvidence(rec.vid, lo_raw.tolist(), float(lo_cf), lo_abs.tolist(),
                         rel.tolist(), screened, lo_dyn, pair_lo, refs)


def stage_features(cfg: Config) -> Tuple[List[str], List[str]]:
    on = {"lo_abs": True, "rel_z": cfg.use_rel_z, "lo_dyn": cfg.use_static_contrast,
          "pair_lo": cfg.use_pairwise}
    return ([f for f in STAGE1_FEATURES if on[f]], [f for f in STAGE2_FEATURES if on[f]])


# =============================================================================
# H: two-state HMM over windows (exact forward-backward in log space)
# =============================================================================

def hmm_transitions(label_seqs: Sequence[Sequence[int]], k: float = 1.0) -> Dict[str, float]:
    """MLE with add-k smoothing: a = P(1 | 0), b = P(0 | 1)."""
    c = np.full((2, 2), float(k))
    for seq in label_seqs:
        for s0, s1 in zip(seq[:-1], seq[1:]):
            c[int(s0), int(s1)] += 1.0
    return {"a": float(c[0, 1] / c[0].sum()), "b": float(c[1, 0] / c[1].sum())}


def forward_backward(log_lr: Sequence[float], a: float, b: float) -> np.ndarray:
    """Posterior P(s_t = 1 | all evidence) for a 2-state chain with emission
    log-likelihood ratios log_lr = log p(x_t|1) - log p(x_t|0). The initial
    distribution is the stationary one, pi_1 = a / (a + b)."""
    e = np.asarray(log_lr, float)
    T = e.size
    if T == 0:
        return np.zeros(0)
    a, b = clamp(a, 1e-6, 1 - 1e-6), clamp(b, 1e-6, 1 - 1e-6)
    logA = np.log(np.array([[1 - a, a], [b, 1 - b]]))
    em = np.stack([np.zeros(T), e], axis=1)
    p1 = a / (a + b)
    al = np.empty((T, 2))
    be = np.zeros((T, 2))
    al[0] = np.log([1 - p1, p1]) + em[0]
    for t in range(1, T):
        al[t] = logsumexp(al[t - 1][:, None] + logA, axis=0) + em[t]
    for t in range(T - 2, -1, -1):
        be[t] = logsumexp(logA + (em[t + 1] + be[t + 1])[None, :], axis=1)
    post = al + be
    return expit(post[:, 1] - post[:, 0])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def window_probs(model: Dict[str, Any], rows: Sequence[Dict[str, float]],
                 screened: Sequence[int]) -> np.ndarray:
    """Stage-1 posterior on every window, replaced by the stage-2 posterior on
    screened windows (whose observation includes the extra evidence)."""
    p = fusion_predict(model["s1"], rows) if rows else np.zeros(0)
    scr = [int(i) for i in screened]
    if scr and model.get("s2") is not None:
        p[scr] = fusion_predict(model["s2"], [rows[i] for i in scr])
    return p


def posterior_from_probs(p: np.ndarray, model: Dict[str, Any], use_hmm: bool,
                         gamma: Optional[float] = None) -> np.ndarray:
    if not use_hmm or p.size == 0:
        return np.asarray(p, float)
    g = float(model["gamma"] if gamma is None else gamma)
    log_lr = _logit(p) - float(_logit(np.array([model["prior"]]))[0])
    return forward_backward(g * log_lr, model["hmm"]["a"], model["hmm"]["b"])


def runs_above(q: Sequence[float], thr: float) -> List[List[int]]:
    out, cur = [], []
    for i, v in enumerate(q):
        if v >= thr:
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


# =============================================================================
# Metrics
# =============================================================================

def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, float)
    srt = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() == 0:
        return float("nan")
    y = y[np.argsort(-s, kind="mergesort")]
    tp = np.cumsum(y)
    return float(((tp / np.arange(1, y.size + 1)) * y).sum() / y.sum())


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def cluster_bootstrap_auc(scores: Sequence[float], labels: Sequence[int],
                          groups: Sequence[str], B: int, seed: int) -> Dict[str, float]:
    """Resample GAMES with replacement. Windows of one video/game are not
    independent, so an i.i.d. bootstrap over windows would understate the
    variance and overstate significance."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        take = np.concatenate([idx[u] for u in rng.choice(uniq, uniq.size, replace=True)])
        a = roc_auc(s[take], y[take])
        if not math.isnan(a):
            vals.append(a)
    point = roc_auc(s, y)
    G = int(uniq.size)
    if not vals or G < 2:
        return {"auc": point, "lo": float("nan"), "hi": float("nan"), "n_groups": G}
    # Percentile cluster-bootstrap intervals under-cover when the number of
    # clusters is small (null false-pass rate ~12% at G=8 in the selftest study).
    # The interval used for decisions is therefore a bootstrap-SE interval with a
    # Student-t(G-1) quantile and the G/(G-1) small-sample variance correction.
    from scipy.stats import t as _t
    se = float(np.std(vals, ddof=1)) * math.sqrt(G / (G - 1.0))
    q = float(_t.ppf(0.975, G - 1))
    return {"auc": point, "lo": point - q * se, "hi": point + q * se,
            "pct_lo": float(np.percentile(vals, 2.5)), "pct_hi": float(np.percentile(vals, 97.5)),
            "se": se, "n_groups": G, "p_le_half": float(np.mean(np.asarray(vals) <= 0.5))}


def select_threshold_mcc(p: Sequence[float], y: Sequence[int]) -> Tuple[float, Dict]:
    p, y = np.asarray(p, float), np.asarray(y, int)
    grid = np.unique(np.round(np.concatenate([np.quantile(p, np.linspace(0.01, 0.99, 99)),
                                              np.arange(0.05, 0.96, 0.05)]), 6))
    best_t, best, curve = 0.5, -2.0, []
    for t in grid:
        pr = (p >= t).astype(int)
        tp, fp = int(((pr == 1) & (y == 1)).sum()), int(((pr == 1) & (y == 0)).sum())
        fn, tn = int(((pr == 0) & (y == 1)).sum()), int(((pr == 0) & (y == 0)).sum())
        m = mcc(tp, fp, fn, tn)
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        curve.append({"t": float(t), "mcc": m, "precision": prec, "recall": rec,
                      "pos_rate": float(pr.mean())})
        if m > best:
            best_t, best = float(t), m
    at = min(curve, key=lambda r: abs(r["t"] - best_t))
    return best_t, {"mcc": best, **{k: at[k] for k in ("precision", "recall", "pos_rate")},
                    "base_rate": float(y.mean()) if y.size else 0.0, "curve": curve}


# =============================================================================
# C5: fusion with game-grouped cross-fitting
# =============================================================================

def fit_logistic(X: np.ndarray, y: Sequence[int], l2: float = 1.0,
                 iters: int = 100) -> Tuple[np.ndarray, float]:
    """Newton-Raphson / IRLS for ridge logistic regression (intercept unpenalised)."""
    yv = np.asarray(y, float)
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0
    for _ in range(iters):
        pr = expit(Xb @ w)
        g = Xb.T @ (yv - pr) - reg @ w
        H = Xb.T @ (Xb * np.clip(pr * (1 - pr), 1e-6, None)[:, None]) + reg
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if float(np.abs(step).max()) < 1e-10:
            break
    return w[:-1], float(w[-1])


def fusion_fit(rows: Sequence[Dict[str, float]], y: Sequence[int], feats: Sequence[str],
               l2: float) -> Dict[str, Any]:
    X = np.array([[r[f] for f in feats] for r in rows], float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    w, b = fit_logistic((X - mu) / sd, y, l2)
    return {"features": list(feats), "mu": mu.tolist(), "sd": sd.tolist(),
            "w": w.tolist(), "b": b, "l2": l2}


def fusion_predict(model: Dict[str, Any], rows: Sequence[Dict[str, float]]) -> np.ndarray:
    feats = model["features"]
    X = np.array([[r[f] for f in feats] for r in rows], float).reshape(len(rows), len(feats))
    Z = (X - np.asarray(model["mu"])) / np.asarray(model["sd"])
    return expit(Z @ np.asarray(model["w"]) + float(model["b"]))


def game_folds(games: Sequence[str], k: int, seed: int) -> Dict[str, int]:
    uniq = sorted(set(games))
    random.Random(seed).shuffle(uniq)
    k = max(2, min(k, len(uniq)))
    return {g: i % k for i, g in enumerate(uniq)}


def _fit_stage(rows, y, feats, l2) -> Optional[Dict[str, Any]]:
    y = np.asarray(y, int)
    if len(rows) < 5 or len(set(y.tolist())) < 2:
        return None
    return fusion_fit(rows, y, feats, l2)


def _const_model(feats: Sequence[str], p: float) -> Dict[str, Any]:
    p = clamp(p, 1e-3, 1 - 1e-3)
    return {"features": list(feats), "mu": [0.0] * len(feats), "sd": [1.0] * len(feats),
            "w": [0.0] * len(feats), "b": float(math.log(p / (1 - p))), "l2": 0.0}


def fit_system(videos: Sequence[Dict[str, Any]], cfg: Config,
               gamma: float = 1.0) -> Dict[str, Any]:
    """Stage-1 (all windows), stage-2 (screened windows), unconditional prior,
    and HMM transitions, all from the given calibration videos."""
    f1, f2 = stage_features(cfg)
    rows1 = [r for v in videos for r in v["rows"]]
    y1 = np.concatenate([v["y"] for v in videos]) if videos else np.zeros(0, int)
    rows2 = [v["rows"][i] for v in videos for i in v["screened"]]
    y2 = np.array([int(v["y"][i]) for v in videos for i in v["screened"]], int)
    prior = float(y1.mean()) if y1.size else 0.5
    s1 = _fit_stage(rows1, y1, f1, cfg.fusion_l2) or _const_model(f1, prior)
    s2 = _fit_stage(rows2, y2, f2, cfg.fusion_l2)
    if s2 is None:
        s2 = _const_model(f2, float(y2.mean()) if y2.size else prior)
    hmm = hmm_transitions([v["y"].tolist() for v in videos], cfg.hmm_smoothing)
    return {"s1": s1, "s2": s2, "prior": prior, "hmm": hmm, "gamma": float(gamma),
            "use_hmm": bool(cfg.use_hmm), "version": VERSION}


def parse_gammas(cfg: Config) -> List[float]:
    g = [float(x) for x in str(cfg.hmm_gammas).split(",") if x.strip()]
    require(g and all(x > 0 for x in g), f"bad --hmm_gammas {cfg.hmm_gammas}")
    return g if cfg.use_hmm else [1.0]


def cross_fit_system(videos: Sequence[Dict[str, Any]], cfg: Config
                     ) -> Tuple[Dict[float, List[np.ndarray]], List[np.ndarray]]:
    """Game-grouped cross-fitting of the WHOLE system (both stages, prior and
    transitions): every video is scored by a system that never saw its game.
    Returns out-of-fold posteriors per gamma and the out-of-fold window probs."""
    fold = game_folds([v["game"] for v in videos], cfg.cv_folds, cfg.seed)
    gammas = parse_gammas(cfg)
    q = {g: [None] * len(videos) for g in gammas}
    probs: List[Optional[np.ndarray]] = [None] * len(videos)
    for f in sorted(set(fold.values())):
        te = [i for i, v in enumerate(videos) if fold[v["game"]] == f]
        tr = [videos[i] for i, v in enumerate(videos) if fold[v["game"]] != f]
        m = fit_system(tr, cfg)
        for i in te:
            p = window_probs(m, videos[i]["rows"], videos[i]["screened"])
            probs[i] = p
            for g in gammas:
                q[g][i] = posterior_from_probs(p, m, cfg.use_hmm, g)
    return q, probs


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


# =============================================================================
# G: verified galloping bisection with a likelihood presence oracle
# =============================================================================

def bisect_onset(present: Callable[[int], bool], lo: int, hi: int) -> int:
    """Smallest i in [lo, hi] with present(i), assuming present(hi) and monotone."""
    require(lo <= hi, "bisect_onset: empty range")
    while lo < hi:
        mid = (lo + hi) // 2
        if present(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def bisect_offset(present: Callable[[int], bool], lo: int, hi: int) -> int:
    require(lo <= hi, "bisect_offset: empty range")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if present(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def gallop_left(present: Callable[[int], bool], seed: int, floor_: int, steps: int) -> int:
    step, lo = 1, seed
    for _ in range(steps):
        cand = seed - step
        if cand < floor_:
            return floor_
        if not present(cand):
            return cand + 1
        lo, step = cand, step * 2
    return max(floor_, lo)


def gallop_right(present: Callable[[int], bool], seed: int, ceil_: int, steps: int) -> int:
    step, hi = 1, seed
    for _ in range(steps):
        cand = seed + step
        if cand > ceil_:
            return ceil_
        if not present(cand):
            return cand - 1
        hi, step = cand, step * 2
    return min(ceil_, hi)


def locate_onset(present, seed, floor_, cfg: Config, audit: Dict) -> int:
    lo = gallop_left(present, seed, floor_, cfg.gallop_max_steps)
    i = bisect_onset(present, lo, seed)
    if cfg.use_verified_bisection and i > lo and present(i - 1):
        audit["monotonicity_violations"] += 1
        while i - 1 >= lo and present(i - 1):
            i -= 1
    return i


def locate_offset(present, seed, ceil_, cfg: Config, audit: Dict) -> int:
    hi = gallop_right(present, seed, ceil_, cfg.gallop_max_steps)
    i = bisect_offset(present, seed, hi)
    if cfg.use_verified_bisection and i < hi and present(i + 1):
        audit["monotonicity_violations"] += 1
        while i + 1 <= hi and present(i + 1):
            i += 1
    return i


def extend_over_gaps(present, end: int, ceil_: int, gap: int, audit: Dict) -> int:
    cur = end
    while gap > 0:
        nxt = next((cur + d for d in range(1, gap + 2)
                    if cur + d <= ceil_ and present(cur + d)), None)
        if nxt is None:
            break
        audit["gaps_bridged"] += 1
        cur = nxt
    return cur


class PresenceOracle:
    """Event-specific likelihood oracle: present iff the calibrated log-odds of
    'this same glitch is visibly occurring' exceeds the margin. Callable on a
    window index; .image(path) scores any derived image of the same geometry."""

    def __init__(self, vlm: VLMBase, wins: Sequence[Window], game: str, desc: str,
                 entity: str, cfg: Config):
        self.vlm, self.cfg = vlm, cfg
        self.by_idx = {w.idx: w for w in wins}
        self.usr = presence_user(game, desc, entity)
        self.base = 0.0
        if cfg.use_contextual_calibration and wins:
            cf = content_free_path(wins) if vlm.needs_pixels else "__content_free__"
            self.base = binary_log_odds(vlm.score(P_PRESENCE_SYS, self.usr, [cf], YES_NO,
                                                  "g_calib"), "yes", "no")
        self.memo: Dict[Any, bool] = {}

    def image(self, path: str, stage: str = "g_frame") -> bool:
        if path not in self.memo:
            lo = binary_log_odds(self.vlm.score(P_PRESENCE_SYS, self.usr, [path], YES_NO, stage),
                                 "yes", "no")
            self.memo[path] = (lo - self.base) > self.cfg.presence_margin
        return self.memo[path]

    def __call__(self, i: int) -> bool:
        w = self.by_idx.get(i)
        return False if w is None else self.image(w.stitch_path, "g_presence")


def make_presence_oracle(vlm: VLMBase, wins: Sequence[Window], game: str, desc: str,
                         entity: str, cfg: Config) -> PresenceOracle:
    return PresenceOracle(vlm, wins, game, desc, entity, cfg)


def frame_onset(oracle: PresenceOracle, win: Window, audit: Dict, verify: bool = True) -> int:
    """First glitch frame of the onset window: the smallest position k such that
    the glitch is visible when only frames 0..k are kept. 'The prefix 0..k
    contains a glitch frame' is monotone non-decreasing in k, so bisection needs
    ceil(log2 n) oracle calls; k = n-1 is the full window (known present). A
    verified step repairs a violated monotonicity."""
    n = len(win.frame_ids)
    keep = lambda k: True if k == n - 1 else oracle.image(masked_window_path(win, 0, k))
    k = bisect_onset(keep, 0, n - 1)
    if verify and k > 0 and keep(k - 1):
        audit["monotonicity_violations"] += 1
        while k - 1 >= 0 and keep(k - 1):
            k -= 1
    return k


def frame_offset(oracle: PresenceOracle, win: Window, audit: Dict, verify: bool = True) -> int:
    """Last glitch frame of the offset window: the largest position k such that
    the glitch is visible when only frames k..n-1 are kept (mirror of
    frame_onset; the suffix predicate is monotone non-increasing in k)."""
    n = len(win.frame_ids)
    keep = lambda k: True if k == 0 else oracle.image(masked_window_path(win, k, n - 1))
    k = bisect_offset(keep, 0, n - 1)
    if verify and k < n - 1 and keep(k + 1):
        audit["monotonicity_violations"] += 1
        while k + 1 <= n - 1 and keep(k + 1):
            k += 1
    return k


def frame_time_bounds(win: Window, k_on: int, k_off: int, fps: float) -> Interval:
    """Decoded timestamps, not id / fps (the float accumulation of the sampler
    makes those differ)."""
    t = win.frame_times
    a = clamp(t[k_on], win.start, win.end)
    b = clamp(t[k_off] + 1.0 / fps, a, win.end)
    return (a, b)


REFINE_MODES = ("window", "frame", "shrink_guarded")


def apply_refine_mode(mode: str, w_iv: Interval, f_iv: Interval, guard: float) -> Interval:
    if mode == "window":
        return w_iv
    if mode == "frame":
        return f_iv
    wl = max(EPS, w_iv[1] - w_iv[0])
    if f_iv[1] - f_iv[0] >= guard * wl:
        return f_iv
    c, half = 0.5 * (f_iv[0] + f_iv[1]), 0.5 * guard * wl
    a = clamp(c - half, w_iv[0], w_iv[1])
    return (a, clamp(c + half, a, w_iv[1]))


def windows_to_spans(members: Sequence[int], wins: Sequence[Window]) -> List[Interval]:
    by = {w.idx: w for w in wins}
    return normalize_intervals([(by[m].start, by[m].end) for m in members if m in by])


def ground_event(vlm: VLMBase, wins: Sequence[Window], game: str, members: Sequence[int],
                 desc: str, entity: str, cfg: Config, duration: float
                 ) -> Tuple[Dict[str, List[Interval]], Dict[str, int]]:
    audit = {"monotonicity_violations": 0, "gaps_bridged": 0, "recurrences": 0,
             "frame_conflicts": 0, "unverified": 0}
    idxs = sorted(w.idx for w in wins)
    if not idxs or not members:
        return {m: [] for m in REFINE_MODES}, audit
    if not cfg.use_bisection:
        s = windows_to_spans(members, wins)
        return {m: s for m in REFINE_MODES}, audit
    lo_all, hi_all = idxs[0], idxs[-1]
    by = {w.idx: w for w in wins}
    seeds = sorted(set(members))
    seed_set = set(seeds)
    present = make_presence_oracle(vlm, wins, game, desc, entity, cfg)
    raw: List[Tuple[Interval, Interval]] = []
    consumed: set = set()
    cursor: Optional[int] = seeds[0]
    for _ in range(cfg.max_intervals_per_event):
        if cursor is None or cursor > hi_all:
            break
        if not present(cursor):
            # A detector-confirmed seed the presence oracle rejects keeps its own
            # contiguous run of seeds: the two queries are different questions,
            # and dropping the seed would turn a disagreement into a silent miss.
            s_idx = e_idx = cursor
            while e_idx + 1 in seed_set:
                e_idx += 1
        else:
            s_idx = locate_onset(present, cursor, lo_all, cfg, audit)
            e_idx = locate_offset(present, cursor, hi_all, cfg, audit)
            if cfg.use_gap_tolerance and cfg.gap_tolerance > 0:
                e2 = extend_over_gaps(present, e_idx, hi_all, cfg.gap_tolerance, audit)
                if e2 > e_idx:
                    e_idx = locate_offset(present, e2, hi_all, cfg, audit)
        consumed.update(range(s_idx, e_idx + 1))
        w_iv = (by[s_idx].start, by[e_idx].end)
        f_iv = w_iv
        if cfg.use_subframe and present(s_idx) and present(e_idx):
            v = cfg.use_verified_bisection
            k_on = frame_onset(present, by[s_idx], audit, v)
            k_off = frame_offset(present, by[e_idx], audit, v)
            a_ = frame_time_bounds(by[s_idx], k_on, k_on, cfg.fps)[0]
            b_ = frame_time_bounds(by[e_idx], k_off, k_off, cfg.fps)[1]
            if b_ > a_:
                f_iv = (a_, b_)
            else:
                audit["frame_conflicts"] += 1
        raw.append((w_iv, f_iv))
        nxt = next((s for s in seeds if s > e_idx and s not in consumed), None)
        if nxt is None and cfg.use_recurrence:
            probe = range(e_idx + 1 + cfg.gap_tolerance, hi_all + 1, max(1, cfg.recurrence_stride))
            nxt = next((c for c in probe if c not in consumed and present(c)), None)
            if nxt is not None:
                audit["recurrences"] += 1
        cursor = nxt
    if cfg.use_self_verification and not (any(present.memo.values())
                                          or any(present(i) for i in seeds)):
        # Description-conditioned self-verification: the model confirms its own
        # description on no window (seed or recurrence), so the event is treated
        # as a hallucination (cf. SelfCheckGPT, Manakul et al. 2023). Cost: at
        # most one memoised presence query per seed.
        audit["unverified"] = 1
        return {m: [] for m in REFINE_MODES}, audit
    out = {}
    for mode in REFINE_MODES:
        sp = [apply_refine_mode(mode, w, f, cfg.shrink_guard) for w, f in raw]
        out[mode] = normalize_intervals([(clamp(a, 0, duration), clamp(b, 0, duration))
                                         for a, b in sp])
    return out, audit


# =============================================================================
# Descriptions and entity-anchored event resolution
# =============================================================================

def describe_window(vlm: VLMBase, win: Window, game: str) -> Dict[str, str]:
    obj = vlm.generate_json(P_DESC_SYS, f"Game: {game}. Describe the glitch in this window.",
                            [win.stitch_path], 384, "describe") or {}
    d = str(obj.get("description", "")).strip()
    return {"description": d or "A visual or physics anomaly affecting an object in the scene.",
            "entity": str(obj.get("entity", "") or "entity"),
            "glitch_type": str(obj.get("glitch_type", "") or "other")}


def _fallback_events(obs: Sequence[Dict]) -> List[Dict]:
    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, o in enumerate(obs):
        key = (" ".join(sorted(_tokens(o["entity"]))) or o["entity"], o["glitch_type"].lower())
        buckets[key].append(i)
    return [{"member_ids": ids, "canonical_entity": obs[ids[0]]["entity"],
             "glitch_type": k[1], "description": obs[ids[0]]["description"]}
            for k, ids in buckets.items()]


def resolve_events(vlm: VLMBase, obs: Sequence[Dict], cfg: Config) -> List[Dict]:
    if len(obs) <= 1 or not cfg.use_event_resolution:
        return [{"member_ids": [i], "canonical_entity": o["entity"],
                 "glitch_type": o["glitch_type"], "description": o["description"]}
                for i, o in enumerate(obs)]
    payload = [{"id": i, "window": o["window"], "entity": o["entity"],
                "glitch_type": o["glitch_type"], "observation": o["description"][:300]}
               for i, o in enumerate(obs)]
    obj = vlm.generate_json(P_EAR_SYS, f"Observations (chronological):\n"
                                       f"{json.dumps(payload)[:4000]}", (), 768,
                            "resolution") or {}
    events, seen = [], set()
    for e in obj.get("events") or []:
        if not isinstance(e, dict):
            continue
        ids = []
        for i in e.get("member_ids") or []:
            try:
                ii = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= ii < len(obs) and ii not in seen and ii not in ids:
                ids.append(ii)
        if not ids:
            continue
        seen.update(ids)
        events.append({"member_ids": sorted(ids),
                       "canonical_entity": str(e.get("canonical_entity") or obs[ids[0]]["entity"]),
                       "glitch_type": str(e.get("glitch_type") or obs[ids[0]]["glitch_type"]),
                       "description": str(e.get("description") or obs[ids[0]]["description"])})
    left = [i for i in range(len(obs)) if i not in seen]
    for grp in _fallback_events([obs[i] for i in left]):
        grp["member_ids"] = [left[i] for i in grp["member_ids"]]
        events.append(grp)
    return events


def summarise_event(vlm: VLMBase, descs: Sequence[str], gtype: str,
                    spans: Sequence[Interval]) -> str:
    if len(descs) == 1:
        return descs[0]
    prompt = P_SUMMARY_TEMPLATE.format(
        descriptions="\n".join(f"- {d}" for d in descs), category=gtype, subtype="Unknown",
        time_range=", ".join(f"{a:.1f}s - {b:.1f}s" for a, b in spans))
    txt = vlm.generate("You are a video game glitch analyst.", prompt, (), 512, "summary")
    txt = re.sub(r"```.*?```", "", txt, flags=re.S).strip()
    return txt or descs[0]


def to_time_nodes(spans: Sequence[Interval], cfg: Config) -> List[List[float]]:
    if cfg.quantize_seconds:
        # GliDe summariser: int(frame // fps) on both ends.
        return [[int(math.floor(a)), int(math.floor(b))] for a, b in spans]
    return [[round(float(a), 2), round(float(b), 2)] for a, b in spans]


# =============================================================================
# CRUX end-to-end
# =============================================================================

def run_crux(vlm: VLMBase, rec: Record, wins: Sequence[Window], cfg: Config,
             emit_all_modes: bool = False,
             evidence: Optional[VideoEvidence] = None) -> Dict[str, Any]:
    require(cfg.fusion and "s1" in cfg.fusion,
            "run_crux needs a fitted v9 system model (run 'calibrate')")
    ev = evidence or collect_video_evidence(vlm, rec, wins, cfg)
    rows = [ev.features(i) for i in range(len(wins))]
    probs = window_probs(cfg.fusion, rows, ev.screened)
    q = posterior_from_probs(probs, cfg.fusion, cfg.use_hmm)
    runs = runs_above(q, cfg.decision_threshold)
    out: Dict[str, Any] = {"vid": rec.vid, "video_name": rec.video_name, "game": rec.game,
                           "n_windows": len(wins), "screened": ev.screened,
                           "window_probs": [float(x) for x in probs],
                           "posterior": [float(x) for x in q], "runs": runs,
                           "n_confirmed": int(sum(len(r) for r in runs)),
                           "monotonicity_violations": 0, "gaps_bridged": 0, "recurrences": 0,
                           "frame_conflicts": 0, "unverified": 0}
    by = {w.idx: w for w in wins}
    if not runs:
        out.update({"no_bugs": True, "bugs": [], "time_nodes": [], "reports": []})
        if emit_all_modes:
            out["reports_by_mode"] = {m: [] for m in REFINE_MODES}
        return out
    # One observation per posterior run, described at its peak window.
    obs = []
    for run in runs:
        peak = max(run, key=lambda i: (q[i], -i))
        d = describe_window(vlm, by[peak], rec.game)
        obs.append({"window": peak, "run": run, "prob": float(q[peak]), **d})
    events = resolve_events(vlm, obs, cfg)
    for e in events:
        e["score"] = float(max(obs[i]["prob"] for i in e["member_ids"]))
    events = sorted(events, key=lambda e: -e["score"])[:cfg.max_events_per_video]
    by_mode: Dict[str, List[Dict]] = {m: [] for m in REFINE_MODES}
    for e in events:
        members = sorted({w for i in e["member_ids"] for w in obs[i]["run"]})
        spans_by_mode, audit = ground_event(vlm, wins, rec.game, members, e["description"],
                                            e["canonical_entity"], cfg, rec.duration)
        for k in ("monotonicity_violations", "gaps_bridged", "recurrences", "frame_conflicts",
                  "unverified"):
            out[k] += audit[k]
        if audit["unverified"]:
            continue
        spans_final = spans_by_mode.get(cfg.refine_mode) or windows_to_spans(members, wins)
        desc = summarise_event(vlm, [obs[i]["description"] for i in e["member_ids"]],
                               e["glitch_type"], spans_final)
        for m in REFINE_MODES:
            sp = spans_by_mode.get(m) or windows_to_spans(members, wins)
            if sp:
                by_mode[m].append({"description": desc, "spans": sp, "score": e["score"],
                                   "entity": e["canonical_entity"], "members": members})
    reps = by_mode[cfg.refine_mode if cfg.refine_mode in by_mode else "frame"]
    out.update({"no_bugs": not reps, "reports": reps,
                "bugs": [r["description"] for r in reps],
                "time_nodes": [to_time_nodes(r["spans"], cfg) for r in reps]})
    if emit_all_modes:
        out["reports_by_mode"] = by_mode
    return out


# =============================================================================
# Baselines
# =============================================================================

def run_vanilla(vlm: VLMBase, rec: Record, wins: Sequence[Window], cfg: Config,
                max_images: int = 8) -> Dict[str, Any]:
    """In-house single-pass baseline (not part of the official repo). Windows are
    sampled evenly across the video and their true time ranges are stated, so the
    model is never asked to extrapolate timestamps it was not shown."""
    n = len(wins)
    pick = sorted(set(np.linspace(0, n - 1, min(n, max_images)).round().astype(int).tolist()))
    imgs = [wins[i].pair_path or wins[i].stitch_path for i in pick]
    ranges = "; ".join(f"image {k + 1} = [{wins[i].start:.1f}s, {wins[i].end:.1f}s]"
                       for k, i in enumerate(pick))
    obj = vlm.generate_json(P_VANILLA_SYS,
                            f"Game: {rec.game}. Video duration: {rec.duration:.1f} s. "
                            f"Images in temporal order: {ranges}. Report all glitches.",
                            imgs, 1024, "vanilla") or {}
    bugs, tns = [], []
    for b in obj.get("bugs") or []:
        if not isinstance(b, dict):
            continue
        sp = normalize_intervals([(clamp(float(a), 0, rec.duration), clamp(float(c), 0, rec.duration))
                                  for a, c in (b.get("time_nodes") or [])
                                  if isinstance(a, (int, float)) and isinstance(c, (int, float))])
        d = str(b.get("description", "")).strip()
        if d and sp:
            bugs.append(d)
            tns.append(to_time_nodes(sp, cfg))
    return {"vid": rec.vid, "video_name": rec.video_name, "game": rec.game,
            "no_bugs": not bugs, "bugs": bugs, "time_nodes": tns}


def import_glide_report(path: str, recs: Sequence[Record]) -> Tuple[List[Dict], Dict]:
    """Ingest an official GliDe batch_report.json verbatim: bugs and time_nodes are
    passed to the evaluator exactly as the official code wrote them (integer
    seconds from int(frame // fps)); nothing is re-normalised."""
    data = read_json(path)
    require(isinstance(data, list), f"{path} is not a GliDe batch_report (expected a list)")
    by_name = {r.video_name: r for r in recs}
    outs, unknown = [], 0
    for item in data:
        vn = str(item.get("video_name", "")).replace(".mp4", "")
        r = by_name.get(vn)
        if r is None:
            unknown += 1
            continue
        outs.append({"vid": r.vid, "video_name": vn, "game": r.game,
                     "no_bugs": bool(item.get("no_bugs", True)),
                     "bugs": list(item.get("bugs", [])),
                     "time_nodes": list(item.get("time_nodes", []))})
    missing = sorted(set(by_name) - {o["video_name"] for o in outs})
    info = {"n_report": len(data), "n_matched": len(outs), "n_not_in_split": unknown,
            "n_split_missing_from_report": len(missing), "missing_examples": missing[:10]}
    if missing:
        LOG.warning("GliDe report lacks %d of %d split videos; evaluation will use the "
                    "INTERSECTION of video sets for every method, as fairness requires",
                    len(missing), len(by_name))
    return outs, info


def to_batch_report(outputs: Sequence[Dict]) -> List[Dict]:
    return [{"video_name": o["video_name"], "game_name": o.get("game", "Unknown"),
             "no_bugs": bool(o.get("no_bugs", not o.get("bugs"))),
             "bugs": list(o.get("bugs", [])), "time_nodes": list(o.get("time_nodes", []))}
            for o in outputs]


# =============================================================================
# Official GliDe evaluation protocol (evaluation/evaluator.py), ported exactly:
#   * text-only LLM judge, official prompt, system message, key 'rating' (0-5)
#   * Hungarian matching on the RATING matrix only
#   * IoU computed for each matched pair afterwards (official compute_iou)
#   * EVERY Hungarian pair is appended, including rating-0 pairs
#   * P = sum(r)/(5 N_pred), R = sum(r)/(5 N_gt); *_iou use sum(r * IoU)
#   * mean_iou = mean over all matched pairs; only videos present in both files
# =============================================================================

JUDGE_SYSTEM = ("You must respond with a JSON object containing 'rating' (int 0-5) and "
                "'reasoning' (string).")
# Verbatim from GliDe evaluation/prompt.txt (MIT). Loaded from --glide_repo when given.
JUDGE_PROMPT = (
    "You are an expert in video game glitch analysis. Your task is to evaluate the quality "
    "of a predicted glitch description based on the given ground truth glitch description, "
    "focusing on the accurate identification/detection of the glitch.\n\n"
    "**Ground Truth Description**:\n{gt_description}\n\n"
    "**Predicted Description**:\n{pred_description}\n\n"
    "**Instructions**:\n\n"
    "- **Rate the quality of the predicted description on a scale from 0 to 5**, using the "
    "following criteria:\n\n"
    "- **0: No quality**.\n"
    "  - The predicted description is completely unrelated to the ground truth.\n"
    "  - The descriptions refer to entirely different events or glitches.\n"
    "  - The predicted description is irrelevant or nonsensical in the context of the glitch.\n\n"
    "- **1: Very low quality**.\n"
    "  - The predicted description mentions some elements, but they are incorrect or irrelevant.\n"
    "  - Key details about the glitch are missing or inaccurately described.\n"
    "  - The descriptions have minimal overlap in content or context.\n\n"
    "- **2: Low quality**.\n"
    "  - The predicted description includes minor aspects related to the glitch but misses "
    "the main behavior.\n"
    "  - Important elements are absent or significantly misrepresented.\n"
    "  - The description has some relevant terms but lacks context or specificity.\n\n"
    "- **3: Moderate quality**.\n"
    "  - The predicted description identifies the general type of glitch but lacks "
    "significant details.\n"
    "  - Some key behaviors or elements are mentioned, but important specifics are missing.\n"
    "  - The description captures part of the glitch but omits certain actions or effects.\n\n"
    "- **4: High quality**.\n"
    "  - The predicted description captures the main glitch and includes most key details.\n"
    "  - Minor differences exist (e.g., general terms used instead of specific names), but "
    "the overall glitch is accurately described.\n"
    "  - The description reflects the primary behaviors and impacts of the glitch on the "
    "game experience.\n\n"
    "- **5: Nearly identical**.\n"
    "  - The predicted description accurately identifies the glitch with all key details "
    "matching the ground truth.\n"
    "  - There are no significant differences; both descriptions refer to the same event "
    "with similar specificity.\n"
    "  - The description includes all important behaviors, entities involved, and impacts "
    "on gameplay.\n\n"
    "- **Provide a brief reasoning before your answer**, focusing on:\n"
    "  - How well the predicted description matches the ground truth in identifying the glitch.\n"
    "  - Specific similarities or differences in the behaviors, entities involved, and "
    "details mentioned.\n"
    "  - You should first identify what is the main abnormal behavior of the glitch; as long "
    "as the main behavior is predicted to be the same as the ground truth, you should give a "
    "high rating to the prediction.\n"
    "  - The impact of any missing or additional information on the accuracy of the glitch "
    "identification.\n\n"
    "- **Output Format**:\n\n"
    "Please provide your analysis and your rating in the following JSON format:\n\n"
    "```json\n{{\n  \"reasoning\": \"<brief reasoning of the evaluation>\",\n"
    "  \"rating\": <number from 0 to 5>\n}}\n```\n")


def load_judge_prompt(cfg: Config) -> str:
    if cfg.glide_repo:
        p = os.path.join(cfg.glide_repo, "evaluation", "prompt.txt")
        require(os.path.exists(p), f"--glide_repo given but {p} is missing")
        with open(p) as f:
            txt = f.read()
        if txt != JUDGE_PROMPT:
            LOG.warning("official prompt.txt differs from the embedded copy (sha1 %s vs %s); "
                        "using the repository file", sha1(txt)[:10], sha1(JUDGE_PROMPT)[:10])
        return txt
    return JUDGE_PROMPT


def official_iou(gt: List[List[float]], pred: List[List[float]]) -> float:
    """Exact port of Evaluator.compute_iou (including its conventions: both empty
    -> 1.0; one empty -> 0.0; pairwise intersections merged)."""
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0

    def merge(iv):
        if not iv:
            return []
        s = sorted(iv, key=lambda x: x[0])
        m = [list(s[0])]
        for cur in s[1:]:
            last = m[-1]
            if cur[0] <= last[1]:
                m[-1] = [last[0], max(last[1], cur[1])]
            else:
                m.append(list(cur))
        return m

    def dur(iv):
        return sum(e - s for s, e in merge(iv))

    inter = []
    for gs, ge in gt:
        for ps, pe in pred:
            a, b = max(gs, ps), min(ge, pe)
            if a < b:
                inter.append([a, b])
    g, p, i = dur(gt), dur(pred), dur(inter)
    u = g + p - i
    if u == 0:
        return 1.0 if i == 0 else 0.0
    return max(0.0, min(1.0, i / u))


class JudgeBase:
    name = "base"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prompt = load_judge_prompt(cfg)
        self.cache_dir = cfg.paths()["judge_cache"]
        self.calls = 0
        self.parse_failures = 0

    def rate(self, gt_desc: str, pred_desc: str) -> int:
        key = sha1(json.dumps([self.name, self.cfg.judge_model, self.cfg.judge_temperature,
                               sha1(self.prompt), gt_desc, pred_desc]))
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path) and self.name != "mock":
            return int(read_json(path)["rating"])
        content = self._chat(JUDGE_SYSTEM, self.prompt.format(gt_description=gt_desc,
                                                              pred_description=pred_desc))
        self.calls += 1
        rating = self._parse(content)
        if self.name != "mock":
            atomic_write_json(path, {"rating": rating, "raw": content})
        return rating

    def _parse(self, content: str) -> int:
        """Official order: json.loads first, then the text parser; failure -> 0."""
        for loader in (lambda s: json.loads(s), parse_json_from_text):
            try:
                d = loader(content)
            except (json.JSONDecodeError, TypeError):
                d = None
            if isinstance(d, dict):
                try:
                    return int(d.get("rating", 0))
                except (TypeError, ValueError):
                    break
        self.parse_failures += 1
        return 0

    def _chat(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAIJudge(JudgeBase):
    """Payload shape identical to GliDe LLMClient._chat_openai (model,
    temperature, max_tokens, messages); no extra sampling fields are sent."""
    name = "openai"

    def _chat(self, system, user) -> str:
        import requests
        hdr = {"Content-Type": "application/json"}
        if self.cfg.judge_api_key and self.cfg.judge_api_key != "EMPTY":
            hdr["Authorization"] = f"Bearer {self.cfg.judge_api_key}"
        payload = {"model": self.cfg.judge_model, "temperature": self.cfg.judge_temperature,
                   "max_tokens": self.cfg.judge_max_tokens,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        last = None
        for a in range(3):
            try:
                r = requests.post(f"{self.cfg.judge_api_base.rstrip('/')}/chat/completions",
                                  headers=hdr, json=payload, timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # pragma: no cover - network
                last = e
                time.sleep(2 ** a)
        LOG.warning("judge request failed (%s); official behaviour is rating 0", last)
        return ""


class HFJudge(JudgeBase):
    name = "hf"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._m = self._t = None

    def _chat(self, system, user) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        if self._m is None:
            self._t = AutoTokenizer.from_pretrained(self.cfg.judge_model)
            self._m = AutoModelForCausalLM.from_pretrained(
                self.cfg.judge_model, torch_dtype=torch.bfloat16, device_map="auto").eval()
        ids = self._t.apply_chat_template([{"role": "system", "content": system},
                                           {"role": "user", "content": user}],
                                          add_generation_prompt=True, return_tensors="pt"
                                          ).to(self._m.device)
        gc = GenerationConfig(max_new_tokens=self.cfg.judge_max_tokens, do_sample=True,
                              temperature=self.cfg.judge_temperature, top_p=1.0, top_k=0,
                              pad_token_id=self._t.eos_token_id)
        with torch.no_grad():
            out = self._m.generate(ids, generation_config=gc)
        return self._t.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


class MockJudge(JudgeBase):
    name = "mock"

    def __init__(self, cfg: Config, fn: Callable[[str, str], int]):
        super().__init__(cfg)
        self.fn = fn

    def _chat(self, system, user) -> str:
        gt = user.split("**Ground Truth Description**:\n", 1)[1].split("\n\n**Predicted", 1)[0]
        pr = user.split("**Predicted Description**:\n", 1)[1].split("\n\n**Instructions", 1)[0]
        return json.dumps({"reasoning": "mock", "rating": int(self.fn(gt, pr))})


def make_judge(cfg: Config, fn: Optional[Callable] = None) -> JudgeBase:
    if cfg.judge_backend == "openai":
        return OpenAIJudge(cfg)
    if cfg.judge_backend == "hf":
        return HFJudge(cfg)
    require(fn is not None, "mock judge needs a scoring function")
    return MockJudge(cfg, fn)


def official_eval_video(judge: JudgeBase, gt: Dict, pred: Dict) -> Dict[str, Any]:
    gb, pb = list(gt.get("bugs", [])), list(pred.get("bugs", []))
    gtn, ptn = list(gt.get("time_nodes", [])), list(pred.get("time_nodes", []))
    res = {"gt_num": len(gb), "pred_num": len(pb), "matched_scores": [], "matched_ious": [],
           "matches": []}
    if not gb or not pb:
        return res
    S = np.zeros((len(pb), len(gb)))
    for i, pd in enumerate(pb):
        for j, gd in enumerate(gb):
            S[i, j] = judge.rate(gd, pd)
    pi, gi = linear_sum_assignment(-S)
    for a, b in zip(pi, gi):
        s = int(S[a][b])
        iou = official_iou(gtn[b] if b < len(gtn) else [], ptn[a] if a < len(ptn) else [])
        res["matched_scores"].append(s)
        res["matched_ious"].append(iou)
        res["matches"].append({"pred_idx": int(a), "gt_idx": int(b), "score": s, "iou": iou})
    res["score_matrix"] = S.tolist()
    return res


def official_aggregate(vr: Sequence[Dict]) -> Dict[str, float]:
    N = sum(v["pred_num"] for v in vr)
    M = sum(v["gt_num"] for v in vr)
    sc = [s for v in vr for s in v["matched_scores"]]
    io = [i for v in vr for i in v["matched_ious"]]
    sxi = [s * i for v in vr for s, i in zip(v["matched_scores"], v["matched_ious"])]
    P = sum(sc) / (5 * N) if N else 0.0
    R = sum(sc) / (5 * M) if M else 0.0
    Pi = sum(sxi) / (5 * N) if N else 0.0
    Ri = sum(sxi) / (5 * M) if M else 0.0
    hm = lambda a, b: 2 * a * b / (a + b) if a + b > 0 else 0.0
    return {"num_videos": len(vr), "num_gt_bugs": M, "num_pred_bugs": N,
            "num_matched": len(sc), "mean_score": float(np.mean(sc)) if sc else 0.0,
            "mean_iou": float(np.mean(io)) if io else 0.0,
            "precision": P, "recall": R, "f1": hm(P, R),
            "precision_iou": Pi, "recall_iou": Ri, "f1_iou": hm(Pi, Ri)}


def load_gt_official(cfg: Config, recs: Sequence[Record]) -> Dict[str, Dict]:
    """Ground truth as the official evaluator sees it. With --gt_json the official
    groundtruth.json is used verbatim; otherwise the loaded annotations are
    converted (and a warning says so, because v7 preferred time_nodes_validated)."""
    if cfg.gt_json:
        out = {}
        for it in read_json(cfg.gt_json):
            out[str(it["video_name"]).replace(".mp4", "")] = {
                "game_name": it.get("game_name", "Unknown"), "bugs": it.get("bugs", []),
                "time_nodes": it.get("time_nodes", []), "no_bugs": it.get("no_bugs", True)}
        return out
    LOG.warning("--gt_json not given: ground truth is taken from %s. For numbers that are "
                "comparable with the GliDe paper pass the official groundtruth.json.",
                cfg.metadata_name)
    return {r.video_name: {"game_name": r.game, "bugs": list(r.bugs),
                           "time_nodes": [[list(iv) for iv in sp] for sp in r.time_nodes],
                           "no_bugs": r.no_bugs} for r in recs}


# =============================================================================
# Statistics over per-video evaluation results
# =============================================================================

def per_video_f1_iou(v: Dict) -> float:
    return official_aggregate([v])["f1_iou"]


def bootstrap_ci(vr: Sequence[Dict], key: str, B: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(vr)
    vals = [official_aggregate([vr[i] for i in rng.integers(0, n, n)])[key] for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(a: Sequence[Dict], b: Sequence[Dict], key: str, B: int, seed: int,
                     groups: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Paired over videos, or over GAMES when groups is given (cluster bootstrap)."""
    require(len(a) == len(b), "paired bootstrap needs aligned lists")
    rng = np.random.default_rng(seed)
    obs = official_aggregate(a)[key] - official_aggregate(b)[key]
    if groups is None:
        units = [[i] for i in range(len(a))]
    else:
        gi: Dict[str, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            gi[g].append(i)
        units = list(gi.values())
    d = []
    for _ in range(B):
        take = [i for u in rng.integers(0, len(units), len(units)) for i in units[u]]
        d.append(official_aggregate([a[i] for i in take])[key]
                 - official_aggregate([b[i] for i in take])[key])
    d = np.asarray(d)
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    # The percentile 'p' is reported for continuity with v8 only; it is NOT a
    # test of H0 (see paired_cluster_permutation) and is never used for decisions.
    return {"delta": obs, "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)), "p_percentile_not_a_test": clamp(p, 0.0, 1.0)}


def paired_cluster_permutation(a: Sequence[Dict], b: Sequence[Dict], key: str, B: int,
                               seed: int, groups: Optional[Sequence[str]] = None
                               ) -> Dict[str, float]:
    """Fisher randomisation test of H0: the two methods' per-video results are
    exchangeable within each cluster (game). Each cluster's A/B assignment is
    flipped jointly; the statistic is the difference of the official aggregate
    (a ratio of sums, so per-video differences cannot simply be sign-flipped).
    Exact enumeration when 2^G <= B, otherwise Monte Carlo with the (1+k)/(1+B)
    correction. The minimum attainable two-sided p is 2^(1-G): with G = 2 games
    no difference can ever be significant at 0.05, which the bootstrap hid."""
    require(len(a) == len(b), "permutation test needs aligned lists")
    if groups is None:
        units = [[i] for i in range(len(a))]
    else:
        gi: Dict[str, List[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            gi[g].append(i)
        units = list(gi.values())
    G = len(units)
    stat = lambda A, Bv: official_aggregate(A)[key] - official_aggregate(Bv)[key]
    obs = stat(a, b)

    def flipped(mask: Sequence[bool]) -> float:
        A, Bv = list(a), list(b)
        for u, f in zip(units, mask):
            if f:
                for i in u:
                    A[i], Bv[i] = b[i], a[i]
        return stat(A, Bv)

    tol = 1e-12
    if G <= 20 and 2 ** G <= B:
        import itertools
        ds = [flipped(m) for m in itertools.product((False, True), repeat=G)]
        p = float(np.mean([abs(d) >= abs(obs) - tol for d in ds]))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        k = sum(abs(flipped(rng.random(G) < 0.5)) >= abs(obs) - tol for _ in range(B))
        p = (1.0 + k) / (1.0 + B)
        exact = False
    return {"delta": obs, "p": float(clamp(p, 0.0, 1.0)), "n_clusters": G, "exact": exact,
            "min_attainable_p": float(2.0 ** (1 - G)) if G > 0 else 1.0}


def paired_wilcoxon(a: Sequence[Dict], b: Sequence[Dict]) -> Dict[str, float]:
    xa = np.array([per_video_f1_iou(v) for v in a])
    xb = np.array([per_video_f1_iou(v) for v in b])
    nz = int(np.sum(np.abs(xa - xb) > 1e-12))
    if nz < 10:
        return {"stat": float("nan"), "p": 1.0, "n_nonzero": nz}
    st, p = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(st), "p": float(p), "n_nonzero": nz}


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, still = len(items), {}, True
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        still = still and p <= thr
        out[k] = {"p": p, "threshold": thr, "reject": bool(still)}
    return out


# =============================================================================
# Evidence collection and the calibration design
# =============================================================================

def gather_evidence(cfg: Config, vlm: VLMBase, recs: Sequence[Record],
                    wins_map: Dict[str, List[Window]]) -> Dict[str, VideoEvidence]:
    evs, t0 = {}, time.time()
    for i, r in enumerate(recs):
        evs[r.vid] = collect_video_evidence(vlm, r, wins_map[r.vid], cfg)
        if (i + 1) % 25 == 0:
            LOG.info("evidence %d/%d videos | %d logical calls (%d run, %d cached, %d "
                     "top-logprob truncations) | %.1f min", i + 1, len(recs), vlm.budget.calls,
                     vlm.budget.model_calls, vlm.budget.cache_hits,
                     vlm.budget.logprob_truncations, (time.time() - t0) / 60)
    return evs


def build_design(cfg: Config, recs: Sequence[Record], wins_map: Dict[str, List[Window]],
                 evs: Dict[str, VideoEvidence]) -> List[Dict[str, Any]]:
    """One entry per calibration video: feature rows for every window, window
    labels, the screened positions and the game (the cross-fitting group)."""
    out = []
    for r in recs:
        ev, wins = evs[r.vid], wins_map[r.vid]
        out.append({"vid": r.vid, "game": r.game,
                    "rows": [ev.features(i) for i in range(len(wins))],
                    "y": window_labels(r, wins, cfg.label_min_overlap),
                    "screened": list(ev.screened)})
    return out


def _flat(videos: Sequence[Dict[str, Any]], key: str, screened_only: bool = False
          ) -> Tuple[List[float], np.ndarray, List[str]]:
    s, y, g = [], [], []
    for v in videos:
        idx = v["screened"] if screened_only else range(len(v["rows"]))
        for i in idx:
            s.append(float(v["rows"][i][key]))
            y.append(int(v["y"][i]))
            g.append(v["game"])
    return s, np.asarray(y, int), g


def feature_diagnostics(videos: Sequence[Dict[str, Any]], cfg: Config) -> Dict[str, Dict]:
    f1, f2 = stage_features(cfg)
    out = {}
    for f in dict.fromkeys(f1 + f2):
        scr = f in SCREENED_ONLY
        s, y, g = _flat(videos, f, scr)
        ci = cluster_bootstrap_auc(s, y, g, min(cfg.n_bootstrap, 1000), cfg.seed)
        out[f] = {"scope": "screened" if scr else "all_windows", **ci,
                  "ap": average_precision(s, y), "constant": bool(np.std(s) < 1e-9)}
    # Within-screened comparison of every feature: the fair test of whether the
    # screened-only evidence adds anything beyond lo_abs on the same windows.
    for f in dict.fromkeys(f2):
        s, y, g = _flat(videos, f, True)
        out[f]["screened_auc"] = roc_auc(s, y)
    return out


def assess(cfg: Config, videos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Out-of-fold system posteriors, evidence temperature, threshold and the
    degeneracy decision. Every number here comes from a system that did not
    see the scored video's game."""
    y_all = np.concatenate([v["y"] for v in videos])
    g_all = [v["game"] for v in videos for _ in range(len(v["y"]))]
    n_scr = sum(len(v["screened"]) for v in videos)
    require(n_scr >= 30, f"only {n_scr} screened windows; need >= 30")
    require(len({v["game"] for v in videos}) >= 3, "calibration needs >= 3 games")
    require(0 < int(y_all.sum()) < len(y_all), "calibration windows contain a single class")
    qg, probs = cross_fit_system(videos, cfg)
    ll = {g: log_loss(np.concatenate(q), y_all) for g, q in qg.items()}
    gamma = min(ll, key=lambda g: (ll[g], abs(g - 1.0)))
    q = np.concatenate(qg[gamma])
    p = np.concatenate(probs)
    thr, rep = select_threshold_mcc(q, y_all)
    auc = cluster_bootstrap_auc(q, y_all, g_all, cfg.n_bootstrap, cfg.seed)
    auc_p = cluster_bootstrap_auc(p, y_all, g_all, min(cfg.n_bootstrap, 1000), cfg.seed + 1)
    scr_mask = np.concatenate([np.isin(np.arange(len(v["y"])), v["screened"]) for v in videos])
    screen_recall = float(y_all[scr_mask].sum()) / max(1, int(y_all.sum()))
    reasons = []
    if math.isnan(auc["lo"]) or auc["lo"] <= cfg.min_auc_ci_lo:
        reasons.append(f"out-of-fold posterior ROC-AUC {auc['auc']:.3f} with game-cluster 95% CI "
                       f"[{auc['lo']:.3f}, {auc['hi']:.3f}] does not exclude chance")
    if rep["mcc"] < cfg.min_mcc:
        reasons.append(f"best out-of-fold MCC {rep['mcc']:.3f} < {cfg.min_mcc:.2f}")
    base = float(y_all.mean())
    if not (0.1 * base < rep["pos_rate"] < 0.98):
        reasons.append(f"positive rate {rep['pos_rate']:.3f} at the chosen threshold "
                       f"(window base rate {base:.3f})")
    pf = np.array([float(v["y"].mean()) if len(v["y"]) else 0.0 for v in videos])
    return {"threshold": thr, "gamma": gamma, "gamma_log_loss": ll,
            "oof_selection": {k: v for k, v in rep.items() if k != "curve"},
            "oof_auc": auc, "oof_auc_without_hmm": auc_p,
            "oof_log_loss_without_hmm": log_loss(p, y_all),
            "screen_recall": screen_recall, "n_screened": n_scr, "n_windows": int(y_all.size),
            "sparsity_check": {"median_video_positive_fraction": float(np.median(pf)),
                               "videos_over_half_positive": float(np.mean(pf > 0.5))},
            "degenerate": bool(reasons), "degeneracy_reasons": reasons, "curve": rep["curve"]}


def log_diagnostics(diag: Dict[str, Dict], A: Dict[str, Any]) -> None:
    LOG.info("per-feature ROC-AUC with game-cluster bootstrap 95%% CI:")
    for f, d in diag.items():
        flag = "SIGNAL" if (not math.isnan(d["lo"]) and d["lo"] > 0.5) else "no evidence"
        LOG.info("   %-8s [%-11s] AUC %.3f  CI [%.3f, %.3f]  AP %.3f  screened-AUC %.3f -> %s",
                 f, d["scope"], d["auc"], d["lo"], d["hi"], d["ap"],
                 d.get("screened_auc", float("nan")), flag)
    a, b = A["oof_auc"], A["oof_auc_without_hmm"]
    LOG.info("system (out-of-fold, all windows): posterior AUC %.3f CI [%.3f, %.3f] | without "
             "HMM %.3f | gamma %.2f | log-loss %.4f vs %.4f without HMM", a["auc"], a["lo"],
             a["hi"], b["auc"], A["gamma"], A["gamma_log_loss"][A["gamma"]],
             A["oof_log_loss_without_hmm"])
    LOG.info("decision: MCC %.3f at t=%.3f | precision %.3f vs base %.3f | recall %.3f | "
             "screening recall %.3f", A["oof_selection"]["mcc"], A["threshold"],
             A["oof_selection"]["precision"], A["oof_selection"]["base_rate"],
             A["oof_selection"]["recall"], A["screen_recall"])
    sc = A["sparsity_check"]
    LOG.info("self-reference assumption: median positive fraction per video %.3f; %.1f%% of "
             "videos are more than half glitch (C2/C3 are weakest there)",
             sc["median_video_positive_fraction"], 100 * sc["videos_over_half_positive"])


def _calib_path(cfg: Config, tag: str = "") -> str:
    return os.path.join(cfg.paths()["params"], f"calibration{('_' + tag) if tag else ''}.json")


def _calib_recs(cfg: Config, splits: Dict[str, List[Record]], names: str) -> List[Record]:
    names_l = [s.strip() for s in names.split(",") if s.strip()]
    require("test" not in names_l, "calibration may never read the test split")
    recs = []
    for n in names_l:
        rr = splits[n][:cfg.limit] if cfg.limit else splits[n]
        recs.extend(rr)
    return recs


def cmd_probe(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]], split: str) -> Dict:
    """Cheap go/no-go: evidence only (no descriptions, no grounding, no judge).
    Run on ~50-100 dev videos BEFORE any full run; if no feature's CI clears 0.5
    the backbone does not see these glitches and no downstream module can fix it."""
    require(split != "test", "probe must not touch the test split")
    recs = splits[split][:cfg.limit] if cfg.limit else splits[split]
    wm = prepare_split(recs, cfg)
    evs = gather_evidence(cfg, vlm, recs, wm)
    V = build_design(cfg, recs, wm, evs)
    diag = feature_diagnostics(V, cfg)
    try:
        A = assess(cfg, V)
        log_diagnostics(diag, A)
    except RuntimeError as e:
        A = {"error": str(e)}
        LOG.error("probe could not fit the system: %s", e)
    go = any(not math.isnan(d["lo"]) and d["lo"] > 0.5 for d in diag.values())
    LOG.info("PROBE VERDICT: %s", "GO -- at least one feature carries signal" if go else
             "NO-GO -- no feature separates glitch windows; do not launch full runs")
    out = {"feature_diagnostics": diag, "assessment": {k: v for k, v in A.items() if k != "curve"},
           "go": go, "n_videos": len(recs), "budget": asdict(vlm.budget)}
    atomic_write_json(os.path.join(cfg.paths()["params"], f"probe_{split}.json"), out)
    return out


def cmd_calibrate(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]],
                  calib_splits: str = "dev", tag: str = "") -> Dict:
    recs = _calib_recs(cfg, splits, calib_splits)
    LOG.info("calibration pool: %d videos, %d games (splits: %s)", len(recs),
             len({r.game for r in recs}), calib_splits)
    wm = prepare_split(recs, cfg)
    evs = gather_evidence(cfg, vlm, recs, wm)
    V = build_design(cfg, recs, wm, evs)
    diag = feature_diagnostics(V, cfg)
    A = assess(cfg, V)
    log_diagnostics(diag, A)
    final = fit_system(V, cfg, A["gamma"])
    LOG.info("final system: stage-1 w=%s | stage-2 w=%s | prior %.3f | HMM a=%.3f b=%.3f | gamma %.2f",
             {f: round(w, 3) for f, w in zip(final["s1"]["features"], final["s1"]["w"])},
             {f: round(w, 3) for f, w in zip(final["s2"]["features"], final["s2"]["w"])},
             final["prior"], final["hmm"]["a"], final["hmm"]["b"], final["gamma"])
    out: Dict[str, Any] = {"version": VERSION, "fusion": final, "decision_threshold": A["threshold"],
                           "feature_diagnostics": diag,
                           "assessment": {k: v for k, v in A.items() if k != "curve"},
                           "selection_curve": A["curve"], "calib_splits": calib_splits,
                           "n_videos": len(recs), "config_switches": {
                               k: getattr(cfg, k) for k in asdict(cfg) if k.startswith("use_")}}
    if A["degenerate"]:
        path = _calib_path(cfg, (tag + "_" if tag else "") + "degenerate")
        atomic_write_json(path, out)
        msg = ("DEGENERATE DETECTOR: " + "; ".join(A["degeneracy_reasons"])
               + f". Diagnostics: {path}. Per-feature CIs above say which evidence is dead.")
        if cfg.fail_on_degenerate:
            raise RuntimeError(msg + " (--allow_degenerate exists for ablation bookkeeping, "
                                     "not for headline numbers)")
        LOG.error(msg)
    cal_cfg = Config(**{**asdict(cfg), "fusion": final, "decision_threshold": A["threshold"]})
    refine, mious = cfg.refine_mode, {}
    if cfg.select_refine_mode and cfg.use_bisection:
        # Judge-free choice of a discrete boundary policy on calibration videos.
        # (In-sample for the detector; only 3 candidates, so the optimism is small.)
        num = {m: 0.0 for m in REFINE_MODES}
        den = {m: 0 for m in REFINE_MODES}
        for r in recs:
            o = run_crux(vlm, r, wm[r.vid], cal_cfg, emit_all_modes=True, evidence=evs[r.vid])
            gt = r.gt_reports()
            for m in REFINE_MODES:
                reps = o["reports_by_mode"][m]
                if not reps or not gt:
                    continue
                M = np.array([[official_iou([list(x) for x in g["spans"]],
                                            [list(x) for x in p["spans"]]) for g in gt]
                              for p in reps])
                ri, ci = linear_sum_assignment(-M)
                num[m] += float(M[ri, ci].sum())
                den[m] += len(ri)
        mious = {m: num[m] / den[m] if den[m] else 0.0 for m in REFINE_MODES}
        refine = max(REFINE_MODES, key=lambda m: (mious[m], m == cfg.refine_mode))
        LOG.info("boundary policy %s selected on calibration IoU (judge-free): %s", refine,
                 {m: round(v, 3) for m, v in mious.items()})
    out.update({"refine_mode": refine, "calib_miou_by_mode": mious})
    atomic_write_json(_calib_path(cfg, tag), out)
    return out


def apply_calibration(cfg: Config, tag: str = "") -> Config:
    path = _calib_path(cfg, tag)
    require(os.path.exists(path), f"calibration missing at {path}: run 'calibrate' first")
    cal = read_json(path)
    require(str(cal.get("version", "")).startswith("9"),
            f"{path} was not written by CRUX v9; re-run 'calibrate'")
    sw = cal.get("config_switches", {})
    diff = {k: (v, getattr(cfg, k)) for k, v in sw.items() if getattr(cfg, k) != v}
    require(not diff, f"component switches differ from calibration {path}: {diff}")
    return Config(**{**asdict(cfg), "fusion": cal["fusion"],
                     "decision_threshold": float(cal["decision_threshold"]),
                     "refine_mode": str(cal.get("refine_mode", cfg.refine_mode))})


# =============================================================================
# run / import / export
# =============================================================================

def cmd_run(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]], method: str,
            split: str, tag: str = "") -> str:
    recs = splits[split][:cfg.limit] if cfg.limit else splits[split]
    wm = prepare_split(recs, cfg)
    if method == "crux":
        cfg = apply_calibration(cfg, tag)
    b0 = asdict(vlm.budget)
    outs = []
    for i, r in enumerate(recs):
        before, st0 = vlm.budget.calls, dict(vlm.budget.by_stage)
        if method == "crux":
            o = run_crux(vlm, r, wm[r.vid], cfg)
        elif method == "vanilla":
            o = run_vanilla(vlm, r, wm[r.vid], cfg)
        else:
            raise ValueError(f"unknown method {method} (GliDe is ingested with import_glide)")
        o["calls"] = vlm.budget.calls - before
        o["calls_by_stage"] = {k: v - st0.get(k, 0) for k, v in vlm.budget.by_stage.items()
                               if v - st0.get(k, 0) > 0}
        o["duration"] = r.duration
        outs.append(o)
        if (i + 1) % 25 == 0:
            LOG.info("%s %d/%d | %.1f logical calls/video", method, i + 1, len(recs),
                     (vlm.budget.calls - b0["calls"]) / (i + 1))
    name = f"{method}{('_' + tag) if tag else ''}_{split}.json"
    path = os.path.join(cfg.paths()["runs"], name)
    atomic_write_json(path, {"method": method + (f"_{tag}" if tag else ""), "split": split,
                             "config": asdict(cfg), "outputs": outs,
                             "calls_per_video": (vlm.budget.calls - b0["calls"]) / max(1, len(recs)),
                             "logprob_truncations": vlm.budget.logprob_truncations
                             - b0["logprob_truncations"]})
    atomic_write_json(path.replace(".json", "_batch_report.json"), to_batch_report(outs))
    LOG.info("wrote %s (+ official-format batch report)", path)
    return path


def cmd_import_glide(cfg: Config, splits: Dict[str, List[Record]], split: str,
                     batch_report: str) -> str:
    recs = splits[split][:cfg.limit] if cfg.limit else splits[split]
    outs, info = import_glide_report(batch_report, recs)
    path = os.path.join(cfg.paths()["runs"], f"glide_official_{split}.json")
    atomic_write_json(path, {"method": "glide_official", "split": split, "outputs": outs,
                             "source": os.path.abspath(batch_report), "import_info": info,
                             "calls_per_video": None})
    LOG.info("imported official GliDe report: %s -> %s", info, path)
    return path


def cmd_export(run_file: str) -> str:
    data = read_json(run_file)
    out = run_file.replace(".json", "_batch_report.json")
    atomic_write_json(out, to_batch_report(data["outputs"]))
    LOG.info("wrote %s (feed to GliDe's evaluation/run.py for an independent check)", out)
    return out


# =============================================================================
# evaluate / ablate
# =============================================================================

def cmd_evaluate(cfg: Config, judge: JudgeBase, splits: Dict[str, List[Record]], split: str,
                 run_files: Sequence[str]) -> Dict:
    recs = splits[split][:cfg.limit] if cfg.limit else splits[split]
    gt = load_gt_official(cfg, recs)
    split_names = {r.video_name for r in recs}
    game_of = {r.video_name: r.game for r in recs}
    runs = []
    for rf in run_files:
        path = rf if os.path.isabs(rf) or os.path.exists(rf) else os.path.join(cfg.paths()["runs"], rf)
        d = read_json(path)
        runs.append((d["method"], {o["video_name"]: o for o in d["outputs"]}, d))
    common = sorted(split_names & set(gt) & set.intersection(*[set(m) for _, m, _ in runs]))
    require(common, "no video is shared by the split, the ground truth and every run file")
    LOG.info("evaluating %d videos common to all %d runs (official protocol; judge=%s @ T=%.1f)",
             len(common), len(runs), cfg.judge_model, cfg.judge_temperature)
    per: Dict[str, List[Dict]] = {}
    for name, m, _ in runs:
        per[name] = [dict(official_eval_video(judge, gt[v], m[v]), video_name=v) for v in common]
    groups = [game_of.get(v, "unknown") for v in common]
    table, cmp_ = {}, {}
    keys = ("precision", "recall", "f1", "mean_iou", "precision_iou", "recall_iou", "f1_iou")
    for name, _, d in runs:
        agg = official_aggregate(per[name])
        ci = {k: bootstrap_ci(per[name], k, cfg.n_bootstrap, cfg.seed) for k in ("f1", "f1_iou")}
        table[name] = {**agg, "ci": ci, "calls_per_video": d.get("calls_per_video")}
    ref = runs[0][0]
    pv = {}
    for name, _, _ in runs[1:]:
        c = {"bootstrap_ci_video": paired_bootstrap(per[ref], per[name], "f1_iou",
                                                    cfg.n_bootstrap, cfg.seed),
             "bootstrap_ci_game": paired_bootstrap(per[ref], per[name], "f1_iou",
                                                   cfg.n_bootstrap, cfg.seed, groups),
             "permutation_game": paired_cluster_permutation(per[ref], per[name], "f1_iou",
                                                            cfg.n_permutations, cfg.seed, groups),
             "wilcoxon": paired_wilcoxon(per[ref], per[name])}
        cmp_[f"{ref} vs {name}"] = c
        pv[f"{ref} vs {name}"] = c["permutation_game"]["p"]
    holm = holm_bonferroni(pv) if pv else {}
    print("\n" + "=" * 112)
    print(f"Official GliDe protocol | split={split} | videos={len(common)} | judge={cfg.judge_model}")
    print("-" * 112)
    print(f"{'method':<24}" + "".join(f"{k:>12}" for k in keys) + f"{'calls/vid':>12}")
    for name, row in table.items():
        cpv = row.get("calls_per_video")
        print(f"{name:<24}" + "".join(f"{100 * row[k] if k != 'mean_iou' else row[k]:>12.2f}"
                                     for k in keys) + f"{(cpv if cpv is not None else float('nan')):>12.1f}")
    for k, c in cmp_.items():
        g, pm = c["bootstrap_ci_game"], c["permutation_game"]
        print(f"  {k}: dF1xIoU {100 * g['delta']:+.2f} game-bootstrap 95% CI "
              f"[{100 * g['ci_lo']:+.2f}, {100 * g['ci_hi']:+.2f}] | game sign-flip permutation "
              f"p={pm['p']:.4f} ({'exact' if pm['exact'] else 'MC'}, G={pm['n_clusters']}, "
              f"min p={pm['min_attainable_p']:.4f}; Holm reject={holm.get(k, {}).get('reject')}) "
              f"| Wilcoxon p={c['wilcoxon']['p']:.4f}")
    print("=" * 112)
    res = {"split": split, "n_videos": len(common), "table": table, "comparisons": cmp_,
           "holm": holm, "judge": {"model": cfg.judge_model, "temperature": cfg.judge_temperature,
                                   "calls": judge.calls, "parse_failures": judge.parse_failures},
           "per_video": per}
    atomic_write_json(os.path.join(cfg.paths()["eval"], f"eval_{split}.json"), res)
    return res


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_ctx_calib": {"use_contextual_calibration": False},
    "no_relz": {"use_rel_z": False},
    "no_static_contrast": {"use_static_contrast": False},
    "no_pairwise": {"use_pairwise": False},
    "no_matched_refs": {"use_matched_refs": False},
    "no_hmm": {"use_hmm": False},
    "no_bisection": {"use_bisection": False},
    "no_subframe": {"use_subframe": False},
    "no_recurrence": {"use_recurrence": False},
    "no_event_resolution": {"use_event_resolution": False},
    "no_self_verification": {"use_self_verification": False},
}


def cmd_ablate(cfg: Config, vlm: VLMBase, judge: JudgeBase, splits: Dict[str, List[Record]],
               split: str, calib_splits: str) -> Dict:
    """Each variant is RE-CALIBRATED (fusion refit, threshold reselected) on the
    calibration splits; VLM evidence is cached, so this costs almost nothing but
    the grounding calls. Degenerate variants are reported, not hidden."""
    full = cmd_run(cfg, vlm, splits, "crux", split)
    files = [full]
    for tag, over in ABLATIONS.items():
        vcfg = Config(**{**asdict(cfg), **over, "fail_on_degenerate": False})
        cmd_calibrate(vcfg, vlm, splits, calib_splits, tag)
        files.append(cmd_run(vcfg, vlm, splits, "crux", split, tag))
    return cmd_evaluate(cfg, judge, splits, split, files)


# =============================================================================
# Self-tests: unit tests, parity with the official repository, end-to-end mock
# =============================================================================

def _noise(key: str) -> float:
    """Deterministic ~N(0,1) from a string (sum of 12 uniforms - 6)."""
    h = hashlib.sha256(key.encode()).digest()
    return float(sum(b / 255.0 for b in h[:12]) - 6.0)


def _write_video(path: str, seconds: float, fps: int = 10, size=(96, 64), seed: int = 0) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, 3)
    for k in range(int(seconds * fps)):
        fr = np.zeros((size[1], size[0], 3), np.uint8)
        fr[:] = base
        x = (k * 3) % (size[0] - 10)
        fr[20:30, x:x + 10] = (255, 255, 255)
        vw.write(fr)
    vw.release()


def _build_synthetic_benchmark(root: str) -> None:
    games = {"dev": ["AlphaRace", "BetaFight", "GammaSim", "DeltaShoot", "IotaPlat",
                     "KappaPuzzle", "LambdaRPG", "MuSport"],
             "test": ["EpsRace", "ZetaSim"], "train": ["EtaRPG"]}
    rows, splits = [], defaultdict(list)
    k = 0
    for split, gl in games.items():
        for g in gl:
            for v in range(3 if split in ("dev", "test") else 1):
                name = f"{split}_{g}_{v}"
                buggy = (v % 3) != 2
                _write_video(os.path.join(root, "raw", g, name + ".mp4"), 12.0, seed=k)
                k += 1
                rows.append({"video_name": name + ".mp4", "game_name": g,
                             "bugs": ["The red car floats above the road surface."] if buggy else [],
                             "time_nodes": [[[4, 8]]] if buggy else [], "no_bugs": not buggy})
                splits[split].append(name)
    atomic_write_json(os.path.join(root, "annotations.json"), rows)
    for s, ids in splits.items():
        atomic_write_json(os.path.join(root, "splits", f"{s}.json"), ids)


_KEEP_RE = re.compile(r"_keep_(\d+)_(\d+)\.jpg$")


def _mock_oracle(label_of: Dict[str, int], game_bias: Dict[str, float], signal: bool,
                 salt: str = "rnd", frame_label_of: Optional[Dict[Tuple[str, int], int]] = None):
    """Scripted VLM. Detector log-odds = 2*label + game bias + shared CONTENT
    noise (identical for a window and its static counterfactual) + independent
    noise; the static counterfactual keeps 0.5*label (a partly static glitch).
    This encodes the modelling assumptions of C1b/C2 so that the self-test can
    check that the code exploits them; it says nothing about real VLMs."""
    frame_label_of = frame_label_of or {}

    def win_key(path: str) -> str:
        ap = os.path.abspath(path)
        stem = re.sub(r"_(stitched|pair|static|keep_\d+_\d+)\.jpg$", "", ap)
        return stem + "_stitched.jpg"

    def rel(path: str) -> str:
        # Noise keys use the last two path components only, so the self-test is
        # reproducible although its data live in a random temporary directory.
        return "/".join(os.path.abspath(path).split(os.sep)[-2:])

    def lab(path: str) -> int:
        if not signal:
            return int(_noise(salt + rel(win_key(path))) > 0)
        m = _KEEP_RE.search(path)
        if m:
            d = os.path.dirname(os.path.abspath(path))
            return int(any(frame_label_of.get((d, f), 0)
                           for f in range(int(m.group(1)), int(m.group(2)) + 1)))
        return int(label_of.get(win_key(path), 0))

    def game_of(user: str) -> str:
        m = re.search(r"Game: ([^.\n]+)", user)
        return m.group(1) if m else ""

    def oracle(kind, system, user, images, options):
        if kind == "score":
            if system == P_DETECT_SYS:
                img = images[0]
                if img == "__content_free__":
                    # The grey probe captures the prompt prior but NOT the game's
                    # footage-level bias; that residual is what C2 must remove.
                    lo = 0.6
                else:
                    shared = game_bias.get(game_of(user), 0.0) + 0.8 * _noise(salt + "c" + rel(win_key(img)))
                    if img.endswith("_static.jpg"):
                        lo = 0.5 * lab(img) + shared
                    else:
                        lo = 2.0 * lab(img) + shared + 0.4 * _noise(salt + "d" + rel(img))
                return {"yes": lo / 2, "no": -lo / 2}
            if system == P_PAIR_SYS:
                lo = 0.8 + 1.5 * (lab(images[0]) - lab(images[1])) + 0.3 * _noise(salt + "p" + rel(images[0]) + rel(images[1]))
                return {"A": lo / 2, "B": -lo / 2}
            if system == P_PRESENCE_SYS:
                lo = -1.0 if images[0] == "__content_free__" else 3.0 * lab(images[0]) - 1.5
                return {"yes": lo / 2, "no": -lo / 2}
            raise AssertionError(f"unexpected score prompt: {system[:40]}")
        if system == P_DESC_SYS:
            pos = lab(images[0])
            return json.dumps({"description": "The red car floats above the road surface."
                               if pos else "A wall texture flickers briefly.",
                               "entity": "red car" if pos else "wall",
                               "glitch_type": "floating" if pos else "texture"})
        if system == P_EAR_SYS:
            return json.dumps({"events": []})
        if system == P_VANILLA_SYS:
            return json.dumps({"no_bugs": False, "bugs": [
                {"description": "A car floats in the air.", "time_nodes": [[3, 9]]}]})
        return "The red car floats above the road surface for several seconds."
    return oracle


def selftest(glide_repo: str = "", verbose: bool = True) -> None:
    import tempfile
    fails, n = [], [0]

    def chk(name: str, cond: bool, detail: str = "") -> None:
        n[0] += 1
        if not cond:
            fails.append(f"{name}: {detail}")
        if verbose:
            print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))

    if not LOG.handlers:
        logging.basicConfig(level=logging.WARNING)
    print("== unit tests ==")
    chk("official_iou both empty = 1", official_iou([], []) == 1.0)
    chk("official_iou one empty = 0", official_iou([[0, 1]], []) == 0.0)
    chk("official_iou half", abs(official_iou([[0, 4]], [[0, 2]]) - 0.5) < 1e-12)
    chk("official_iou multi", abs(official_iou([[0, 2], [4, 6]], [[1, 5]]) - 2 / 6) < 1e-12)
    chk("official_iou zero-length pred", official_iou([[3, 4]], [[3, 3]]) == 0.0)
    chk("normalize merges", normalize_intervals([(3, 4), (0, 1), (0.5, 2)]) == [(0, 2), (3, 4)])
    chk("json fenced", parse_json_from_text('x ```json\n{"a": 1}\n``` y') == {"a": 1})
    chk("json repaired", parse_json_from_text("{'a': True,}") == {"a": True})
    chk("json none", parse_json_from_text("no json here") is None)

    c0 = Config(temperature=0.5, top_k=-1)
    g = build_generation_kwargs(c0, 32)
    chk("sampling: top_k disabled explicitly (v7 inherited top_k=1)", g["top_k"] == 0 and g["do_sample"])
    g = build_generation_kwargs(Config(temperature=0.0), 32)
    chk("greedy: fully specified", g["do_sample"] is False and g["top_k"] == 0)

    top = [{"token": "Yes", "logprob": -0.2}, {"token": " yes", "logprob": -2.0},
           {"token": "Maybe", "logprob": -3.0}]
    m, tr = match_top_logprobs(top, YES_NO)
    chk("top_logprobs: variants summed", abs(m["yes"] - float(logsumexp([-0.2, -2.0]))) < 1e-9)
    chk("top_logprobs: absent option -> floor + flag", m["no"] == -3.0 and tr)
    nl = normalise_option_logprobs({"a": -1.0, "b": -2.0})
    chk("option normalisation", abs(math.exp(nl["a"]) + math.exp(nl["b"]) - 1) < 1e-12)

    b, d = 0.9, -0.7
    chk("pair debiasing cancels position bias", abs(debiased_pair_log_odds(b + d, b - d) - d) < 1e-12)
    z = robust_z(np.array([0, 0, 0, 0, 10.0]))
    chk("robust z: outlier stands out, floor on MAD", z[-1] > 5 and abs(z[0]) < 1e-9)
    refs = choose_references(np.array([5., 1., 0., 3., 2., 4.]), 2, 2, 2)
    chk("references respect temporal gap", refs == [4, 5], str(refs))
    chk("screen picks top-k by score", screen_indices(np.array([1., 5., 3., 4.]), 2, 1) == [1, 3])
    emb = np.eye(6)
    emb[4] = emb[2] * 0.9 + emb[4] * 0.1
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    lo6 = np.array([5., 1., 0., 3., 2., 4.])
    mr = choose_matched_references(lo6, emb, 2, 2, 2)
    chk("matched refs: most similar eligible window first", mr[0] == 4, str(mr))
    chk("matched refs: never above the video median unless padding",
        all(lo6[j] <= np.median(lo6) for j in mr[:1]) and len(mr) == 2, str(mr))

    # ---- HMM: forward-backward equals brute-force enumeration --------------------
    import itertools
    rng_h = np.random.default_rng(7)
    for trial in range(3):
        T = 6
        llr = rng_h.normal(0, 2, T)
        ah, bh = float(rng_h.uniform(0.05, 0.4)), float(rng_h.uniform(0.05, 0.6))
        A_ = np.array([[1 - ah, ah], [bh, 1 - bh]])
        p1 = ah / (ah + bh)
        num_, den_ = np.zeros(T), 0.0
        for path in itertools.product((0, 1), repeat=T):
            w_ = (p1 if path[0] else 1 - p1) * math.exp(llr[0] * path[0])
            for t in range(1, T):
                w_ *= A_[path[t - 1], path[t]] * math.exp(llr[t] * path[t])
            den_ += w_
            num_ += w_ * np.array(path)
        chk(f"HMM forward-backward == enumeration (trial {trial})",
            np.allclose(forward_backward(llr, ah, bh), num_ / den_, atol=1e-10))
    llr = rng_h.normal(size=5)
    chk("HMM with independent states reduces to Bayes' rule",
        np.allclose(forward_backward(llr, 0.3, 0.7), expit(llr + math.log(0.3 / 0.7)), atol=1e-10))
    tr_ = hmm_transitions([[0, 0, 1, 1, 0], [0, 1]], k=0.0)
    chk("HMM transition MLE", abs(tr_["a"] - 2 / 3) < 1e-12 and abs(tr_["b"] - 1 / 2) < 1e-12, str(tr_))
    chk("runs above threshold", runs_above([.1, .6, .7, .2, .9], .5) == [[1, 2], [4]])

    # ---- cluster permutation test validity ---------------------------------------
    mk = lambda sc, io: {"gt_num": 1, "pred_num": 1, "matched_scores": [sc], "matched_ious": [io]}
    A2 = [mk(5, 1.0) for _ in range(6)]
    B2 = [mk(0, 0.0) for _ in range(6)]
    pm2 = paired_cluster_permutation(A2, B2, "f1_iou", 1000, 0, ["g1"] * 3 + ["g2"] * 3)
    chk("permutation: G=2 can never reach p<0.05 (v8 bootstrap reported 0.0000)",
        pm2["exact"] and pm2["p"] >= 0.5, str(pm2))
    g8 = [f"g{i // 2}" for i in range(16)]
    pm8 = paired_cluster_permutation([mk(5, 1.0)] * 16, [mk(0, 0.0)] * 16, "f1_iou", 1000, 0, g8)
    chk("permutation: G=8 exact, maximal effect reaches min p = 2^(1-G)",
        pm8["exact"] and abs(pm8["p"] - 2 ** -7) < 1e-12, str(pm8))
    rng_p = np.random.default_rng(3)
    Ar = [mk(int(rng_p.integers(0, 6)), float(rng_p.random())) for _ in range(40)]
    Br = [mk(int(rng_p.integers(0, 6)), float(rng_p.random())) for _ in range(40)]
    gr = [f"g{i % 20}" for i in range(40)]
    pmr = paired_cluster_permutation(Ar, Br, "f1_iou", 999, 0, gr)
    chk("permutation: Monte-Carlo branch, null data not rejected", (not pmr["exact"]) and pmr["p"] > 0.05,
        str(pmr))

    # ---- derived images and frame-level likelihood bisection ----------------------
    if cv2 is not None:
        td0 = tempfile.mkdtemp()
        frs = []
        for k in range(8):
            f_ = np.full((120, 200, 3), 60, np.uint8)
            if k == 3:
                f_[70:110, 120:190] = 255         # transient glitch in ONE frame
            frs.append(f_)
        ids = list(range(40, 48))
        sp_ = os.path.join(td0, "window_0005_stitched.jpg")
        cv2.imwrite(sp_, stitch_official(frs, ids), [cv2.IMWRITE_JPEG_QUALITY, 95])
        w5 = Window(5, 10.0, 12.0, sp_, ids, [10.0 + 0.25 * k for k in range(8)])
        st_ = cv2.imread(static_counterfactual_path(w5))
        cells_, *_ = _split_cells(st_, 8)
        orig_, *_ = _split_cells(cv2.imread(sp_), 8)
        chk("static counterfactual removes a transient (1/8 frames) event",
            float(orig_[3][75:105, 125:185].mean()) > 200 and
            abs(float(cells_[3][75:105, 125:185].mean()) - 60) < 6,
            f"{orig_[3][75:105, 125:185].mean():.1f} -> {cells_[3][75:105, 125:185].mean():.1f}")
        mk_ = cv2.imread(masked_window_path(w5, 2, 4))
        mc_, *_ = _split_cells(mk_, 8)
        chk("masked window greys frames outside the kept range",
            all(abs(float(mc_[k][60:110, 10:110].mean()) - (60 if 2 <= k <= 4 else 128)) < 6
                for k in range(8)))

        class _FrameStub:
            def __init__(self, glitch_ids):
                self.g, self.calls = set(glitch_ids), 0

            def image(self, path, stage="x"):
                self.calls += 1
                m = _KEEP_RE.search(path)
                return any(f in self.g for f in range(int(m.group(1)), int(m.group(2)) + 1))
        for gl_ids in ([43, 44, 45], [40], [47], [41, 42, 43, 44, 45, 46, 47]):
            stub = _FrameStub(gl_ids)
            au_ = {"monotonicity_violations": 0}
            on_, off_ = frame_onset(stub, w5, au_), frame_offset(stub, w5, au_)
            chk(f"frame bisection recovers first/last glitch frame {gl_ids[0]}..{gl_ids[-1]}",
                ids[on_] == min(gl_ids) and ids[off_] == max(gl_ids) and stub.calls <= 8,
                f"{ids[on_]}..{ids[off_]} in {stub.calls} calls")
        chk("frame times use decoded timestamps", frame_time_bounds(w5, 3, 5, 4.0) == (10.75, 11.5))

    rng = np.random.default_rng(0)
    s, y = rng.normal(size=200), rng.integers(0, 2, 200)
    brute = np.mean([(a > c) + 0.5 * (a == c) for a, ya in zip(s, y) if ya == 1
                     for c, yc in zip(s, y) if yc == 0])
    chk("roc_auc equals brute force", abs(roc_auc(s, y) - brute) < 1e-12)
    chk("roc_auc constant = 0.5", roc_auc(np.ones(10), [0, 1] * 5) == 0.5)
    X = rng.normal(size=(4000, 2))
    yy = (rng.random(4000) < expit(X @ np.array([1.5, -1.0]) + 0.3)).astype(int)
    w, bb = fit_logistic(X, yy, l2=1e-6)
    chk("IRLS recovers logistic weights", np.allclose(w, [1.5, -1.0], atol=0.15) and abs(bb - 0.3) < 0.15,
        f"{w}, {bb}")
    gms = [f"g{i % 7}" for i in range(70)]
    fd = game_folds(gms, 5, 1)
    chk("game folds are game-disjoint", all(len({fd[gm]}) == 1 for gm in set(gms)))
    grp = [f"g{i % 10}" for i in range(400)]
    ss = rng.normal(size=400)
    yy2 = (ss + rng.normal(size=400) > 0).astype(int)
    cb = cluster_bootstrap_auc(ss, yy2, grp, 300, 0)
    chk("cluster bootstrap CI brackets point", cb["lo"] <= cb["auc"] <= cb["hi"] and cb["lo"] > 0.5)
    cb0 = cluster_bootstrap_auc(rng.normal(size=400), yy2, grp, 300, 0)
    chk("cluster bootstrap CI includes 0.5 under no signal", cb0["lo"] <= 0.5 <= cb0["hi"])

    pres = lambda i: 10 <= i <= 20
    au = {"monotonicity_violations": 0, "gaps_bridged": 0, "recurrences": 0}
    cT = Config()
    chk("onset", locate_onset(pres, 15, 0, cT, au) == 10)
    chk("offset", locate_offset(pres, 15, 40, cT, au) == 20)
    nm = lambda i: i in (3, 5, 6, 7)
    au2 = dict(au)
    chk("verified bisection repairs non-monotone onset", locate_onset(nm, 7, 0, cT, au2) <= 5)
    gp = lambda i: i in (5, 6, 8, 9)
    chk("gap bridging", extend_over_gaps(gp, 6, 20, 1, dict(au)) == 9)
    sg = apply_refine_mode("shrink_guarded", (0, 2), (0.9, 1.0), 0.5)
    chk("refine shrink guard keeps >= guard of window", abs(sg[0] - 0.45) < 1e-9 and abs(sg[1] - 1.45) < 1e-9, str(sg))

    # ---- official evaluator port on a hand-computed case -------------------------
    cfgj = Config(judge_backend="mock", work_subdir="selftest_work", root=tempfile.mkdtemp())
    table = {("gA", "pA"): 5, ("gB", "pC"): 4}
    judge = MockJudge(cfgj, lambda gt_, pr_: table.get((gt_, pr_), 0))
    v1 = official_eval_video(judge, {"bugs": ["gA", "gB"], "time_nodes": [[[0, 4]], [[10, 12]]]},
                             {"bugs": ["pA", "pB", "pC"],
                              "time_nodes": [[[0, 2]], [[20, 22]], [[10, 12]]]})
    v2 = official_eval_video(judge, {"bugs": [], "time_nodes": []},
                             {"bugs": ["pX"], "time_nodes": [[[0, 1]]]})
    v3 = official_eval_video(judge, {"bugs": ["gQ"], "time_nodes": [[[0, 5]]]},
                             {"bugs": ["pQ"], "time_nodes": [[[0, 5]]]})
    agg = official_aggregate([v1, v2, v3])
    N, M = 5, 3
    P, R = 9 / (5 * N), 9 / (5 * M)
    Pi, Ri = 6.5 / (5 * N), 6.5 / (5 * M)
    chk("port: rating-0 pairs are counted", agg["num_matched"] == 3 and v3["matched_scores"] == [0])
    chk("port: P/R/F1", abs(agg["precision"] - P) < 1e-12 and abs(agg["recall"] - R) < 1e-12
        and abs(agg["f1"] - 2 * P * R / (P + R)) < 1e-12, str(agg))
    chk("port: *_iou", abs(agg["f1_iou"] - 2 * Pi * Ri / (Pi + Ri)) < 1e-12)
    chk("port: mean_iou over all pairs", abs(agg["mean_iou"] - (0.5 + 1.0 + 1.0) / 3) < 1e-12)

    repo = glide_repo or os.environ.get("GLIDE_REPO", "")
    if repo and os.path.isdir(repo):
        print("== parity with the official GliDe repository ==")
        sys.path.insert(0, repo)
        try:
            from evaluation.evaluator import Evaluator as OfficialEvaluator  # type: ignore

            class _Stub:
                def chat(self, system_msg, user_msg, images=None):
                    gt_ = user_msg.split("**Ground Truth Description**:\n", 1)[1].split("\n\n**Pred", 1)[0]
                    pr_ = user_msg.split("**Predicted Description**:\n", 1)[1].split("\n\n**Instr", 1)[0]
                    return json.dumps({"reasoning": "", "rating": table.get((gt_, pr_), 0)})
            td = tempfile.mkdtemp()
            gtf, prf = os.path.join(td, "gt.json"), os.path.join(td, "pr.json")
            atomic_write_json(gtf, [
                {"video_name": "v1.mp4", "bugs": ["gA", "gB"], "time_nodes": [[[0, 4]], [[10, 12]]], "no_bugs": False},
                {"video_name": "v2.mp4", "bugs": [], "time_nodes": [], "no_bugs": True},
                {"video_name": "v3.mp4", "bugs": ["gQ"], "time_nodes": [[[0, 5]]], "no_bugs": False}])
            atomic_write_json(prf, [
                {"video_name": "v1", "bugs": ["pA", "pB", "pC"], "time_nodes": [[[0, 2]], [[20, 22]], [[10, 12]]]},
                {"video_name": "v2", "bugs": ["pX"], "time_nodes": [[[0, 1]]]},
                {"video_name": "v3", "bugs": ["pQ"], "time_nodes": [[[0, 5]]]}])
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                off = OfficialEvaluator(llm_client=_Stub(), verbose=False).evaluate(gtf, prf).to_dict()
            keys = ("num_gt_bugs", "num_pred_bugs", "num_matched", "mean_iou", "precision",
                    "recall", "f1", "precision_iou", "recall_iou", "f1_iou")
            chk("evaluator port == official Evaluator (all metrics)",
                all(abs(float(off[k]) - float(agg[k])) < 1e-12 for k in keys),
                str({k: (off[k], agg[k]) for k in keys}))
            with open(os.path.join(repo, "evaluation", "prompt.txt")) as f:
                chk("embedded judge prompt == official prompt.txt", f.read() == JUDGE_PROMPT)
            with open(os.path.join(repo, "summarizer", "system_prompt.txt")) as f:
                chk("embedded summariser prompt == official", f.read().strip() == P_SUMMARY_TEMPLATE.strip())
            if cv2 is not None:
                from preprocess.video_preprocessor import VideoPreprocessor  # type: ignore
                vp = os.path.join(td, "clip.mp4")
                _write_video(vp, 5.3, fps=30, size=(160, 96), seed=3)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    VideoPreprocessor(output_path=__import__("pathlib").Path(td), target_fps=4.0,
                                      window_size=8).process_video(vp)
                rr = Record("clip", "clip", "g", "u", vp, False, [], [])
                cP = Config(root=td, work_subdir="w")
                ws = prepare_record(rr, cP, force=True)
                offw = sorted(os.listdir(os.path.join(td, "windows", "clip")))
                offw = [x for x in offw if x.endswith("_stitched.jpg")]
                chk("preprocessing: same number of windows", len(offw) == len(ws), f"{len(offw)} vs {len(ws)}")
                diffs = []
                for w_, name_ in zip(ws, offw):
                    a_ = cv2.imread(w_.stitch_path).astype(int)
                    b_ = cv2.imread(os.path.join(td, "windows", "clip", name_)).astype(int)
                    diffs.append(np.abs(a_ - b_).max() if a_.shape == b_.shape else 999)
                chk("preprocessing: stitched windows pixel-identical", max(diffs) == 0, str(diffs))
        except Exception as e:  # pragma: no cover
            chk("official repository parity checks ran", False, f"{type(e).__name__}: {e}")
        finally:
            sys.path.remove(repo)
    else:
        print("== parity with the official repository: SKIPPED (pass --glide_repo) ==")

    if cv2 is None:
        print("== end-to-end mock pipeline: SKIPPED (opencv missing) ==")
    else:
        print("== end-to-end mock pipeline ==")
        root = tempfile.mkdtemp()
        _build_synthetic_benchmark(root)
        base = Config(root=root, backend="mock", judge_backend="mock", n_bootstrap=300,
                      cv_folds=4, pair_budget=6)
        recs = load_records(base)
        spl = load_splits(base, recs)
        wm = prepare_split(spl["dev"] + spl["test"] + spl["train"], base)
        label_of, frame_label_of = {}, {}
        for r in spl["dev"] + spl["test"]:
            gt_iv = [iv for spans in r.time_nodes for iv in spans]
            for w_, y_ in zip(wm[r.vid], window_labels(r, wm[r.vid], base.label_min_overlap)):
                label_of[os.path.abspath(w_.stitch_path)] = int(y_)
                d_ = os.path.dirname(os.path.abspath(w_.stitch_path))
                for fid, ft in zip(w_.frame_ids, w_.frame_times):
                    frame_label_of[(d_, fid)] = int(any(a_ <= ft < b_ for a_, b_ in gt_iv))
        chk("synthetic labels contain both classes", 0 < sum(label_of.values()) < len(label_of))
        bias = {"AlphaRace": 2.0, "BetaFight": -1.0, "GammaSim": 1.0, "DeltaShoot": 0.0,
                "IotaPlat": -2.0, "KappaPuzzle": 1.5, "LambdaRPG": -0.5, "MuSport": 0.5,
                "EpsRace": 2.5, "ZetaSim": -1.5}
        vlm = make_vlm(base, _mock_oracle(label_of, bias, True, frame_label_of=frame_label_of))
        judge_fn = lambda gt_, pr_: 5 if ("float" in pr_.lower() and "float" in gt_.lower()) else 0
        cal = cmd_calibrate(base, vlm, spl, "dev")
        chk("signal oracle: detector not degenerate", not cal["assessment"]["degenerate"],
            str(cal["assessment"].get("degeneracy_reasons")))
        chk("C2 removes game bias (rel_z AUC > lo_abs AUC)",
            cal["feature_diagnostics"]["rel_z"]["auc"] > cal["feature_diagnostics"]["lo_abs"]["auc"],
            f"{cal['feature_diagnostics']['rel_z']['auc']:.3f} vs "
            f"{cal['feature_diagnostics']['lo_abs']['auc']:.3f}")
        fd_ = cal["feature_diagnostics"]
        chk("C1b static contrast removes content noise (screened AUC lo_dyn > lo_abs)",
            fd_["lo_dyn"]["screened_auc"] > fd_["lo_abs"]["screened_auc"],
            f"{fd_['lo_dyn']['screened_auc']:.3f} vs {fd_['lo_abs']['screened_auc']:.3f}")
        asm = cal["assessment"]
        chk("HMM gamma chosen from the grid by out-of-fold log-loss",
            asm["gamma"] in parse_gammas(base) and len(asm["gamma_log_loss"]) == len(parse_gammas(base)))
        print(f"  [info] out-of-fold log-loss: HMM {asm['gamma_log_loss'][asm['gamma']]:.4f} vs "
              f"no HMM {asm['oof_log_loss_without_hmm']:.4f}; AUC {asm['oof_auc']['auc']:.3f} vs "
              f"{asm['oof_auc_without_hmm']['auc']:.3f} (mock; not evidence about real data)")
        f_crux = cmd_run(base, vlm, spl, "crux", "test")
        run_ = read_json(f_crux)
        chk("frame-level bisection recovers the synthetic [4, 8] s span",
            all(o["time_nodes"] == [[[4.0, 8.0]]] for o in run_["outputs"] if o["time_nodes"]) and
            any(o["time_nodes"] for o in run_["outputs"]),
            str([o["time_nodes"] for o in run_["outputs"]]))
        chk("clean test videos produce no report",
            all(not o["bugs"] for o, r in zip(run_["outputs"], spl["test"]) if r.no_bugs))
        f_van = cmd_run(base, vlm, spl, "vanilla", "test")
        gl = [{"video_name": r.video_name, "game_name": r.game, "no_bugs": False,
               "bugs": ["Something odd happens."], "time_nodes": [[[0, 1]]]} for r in spl["test"][:-1]]
        gpath = os.path.join(root, "glide_batch_report.json")
        atomic_write_json(gpath, gl)
        f_gl = cmd_import_glide(base, spl, "test", gpath)
        res = cmd_evaluate(base, make_judge(base, judge_fn), spl, "test", [f_crux, f_van, f_gl])
        chk("evaluation restricted to common videos", res["n_videos"] == len(spl["test"]) - 1)
        chk("crux beats a content-free baseline on the mock",
            res["table"]["crux"]["f1_iou"] > res["table"]["glide_official"]["f1_iou"])
        chk("batch report written in official format",
            os.path.exists(f_crux.replace(".json", "_batch_report.json")))
        vh = Config(**{**asdict(base), "use_hmm": False})
        cal_h = cmd_calibrate(vh, vlm, spl, "dev", "no_hmm")
        chk("ablation without HMM calibrates and runs",
            cal_h["fusion"]["use_hmm"] is False and os.path.exists(cmd_run(vh, vlm, spl, "crux", "test", "no_hmm")))
        vlm0 = make_vlm(base, _mock_oracle(label_of, bias, False))
        try:
            cmd_calibrate(base, vlm0, spl, "dev")
            chk("no-signal oracle is rejected as degenerate", False, "calibrate did not raise")
        except RuntimeError as e:
            chk("no-signal oracle is rejected as degenerate", "DEGENERATE" in str(e), str(e)[:200])
        ab = Config(**{**asdict(base), "use_pairwise": False})
        try:
            apply_calibration(ab)
            chk("switch mismatch with calibration is refused", False)
        except RuntimeError:
            chk("switch mismatch with calibration is refused", True)

    print(f"\n{n[0] - len(fails)}/{n[0]} checks passed")
    if fails:
        for f in fails:
            print("  FAIL:", f)
        raise SystemExit(1)


# =============================================================================
# CLI
# =============================================================================

COMMANDS = ("selftest", "prepare", "probe", "calibrate", "run", "import_glide", "export",
            "evaluate", "ablate")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CRUX v9 (see module docstring)")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--calib_splits", default="dev", help="comma list, never 'test'")
    ap.add_argument("--method", default="crux", choices=("crux", "vanilla"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--run_files", nargs="*", default=[])
    ap.add_argument("--run_file", default="")
    ap.add_argument("--batch_report", default="")
    ap.add_argument("--allow_degenerate", action="store_true")
    skip = {"fusion", "limit", "fail_on_degenerate"}
    for f in Config.__dataclass_fields__.values():
        if f.name in skip:
            continue
        default = f.default
        if isinstance(default, bool):
            ap.add_argument(f"--{f.name}", default=None, action=argparse.BooleanOptionalAction)
        elif isinstance(default, (int, float, str)):
            ap.add_argument(f"--{f.name}", default=None, type=type(default))
    ap.add_argument("--limit", type=int, default=None)
    return ap


def cfg_from_args(a: argparse.Namespace) -> Config:
    base = Config()
    if a.backbone:
        base = base.with_backbone(a.backbone)
    over = {k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__
            and v is not None and k != "backbone"}
    cfg = Config(**{**asdict(base), **over})
    cfg.fail_on_degenerate = not a.allow_degenerate
    return cfg


def main() -> None:
    a = build_parser().parse_args()
    if a.command == "selftest":
        selftest(a.glide_repo or "")
        return
    cfg = cfg_from_args(a)
    setup_logging(cfg.paths()["work"], a.command)
    set_seed(cfg.seed)
    LOG.info("CRUX v%s | %s | backbone=%s backend=%s model=%s", VERSION, a.command,
             cfg.backbone, cfg.backend, cfg.model)
    if a.command == "export":
        require(a.run_file, "--run_file is required")
        cmd_export(a.run_file)
        return
    recs = load_records(cfg)
    splits = load_splits(cfg, recs)
    if a.command == "prepare":
        prepare_split(splits[a.split][:cfg.limit] if cfg.limit else splits[a.split], cfg)
        return
    if a.command == "import_glide":
        require(a.batch_report, "--batch_report is required")
        cmd_import_glide(cfg, splits, a.split, a.batch_report)
        return
    if a.command == "evaluate":
        require(a.run_files, "--run_files is required (first file = reference method)")
        cmd_evaluate(cfg, make_judge(cfg), splits, a.split, a.run_files)
        return
    vlm = make_vlm(cfg)
    if a.command == "probe":
        cmd_probe(cfg, vlm, splits, a.split)
    elif a.command == "calibrate":
        cmd_calibrate(cfg, vlm, splits, a.calib_splits, a.tag)
    elif a.command == "run":
        cmd_run(cfg, vlm, splits, a.method, a.split, a.tag)
    elif a.command == "ablate":
        cmd_ablate(cfg, vlm, make_judge(cfg), splits, a.split, a.calib_splits)
    LOG.info("budget: %s", asdict(vlm.budget))


if __name__ == "__main__":
    main()
