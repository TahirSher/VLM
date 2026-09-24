"""
CRUX v10 -- Hypothesise, Verify, Segment, Select: calibrated likelihood
verification for open-ended video-game glitch detection and temporal grounding
(VideoGlitchBench; baseline and evaluation parity with the official GliDe code,
https://github.com/SandyyyZheng/GliDe, arXiv:2604.07818).

Task (GliDe problem statement): for each gameplay video output every glitch as
a free-text description plus time span(s); scored by an LLM judge (0-5) with
Hungarian matching on the ratings, and by rating x temporal IoU.

Why v9 failed (probe on 72 dev videos / 12 games / 709 windows)
--------------------------------------------------------------
  * The generic question "does this window show a glitch?" measures the VIDEO,
    not the window: sd within videos 0.44 < sd between videos 0.60 logits;
    pooled AUC 0.557-0.569, per-video median AUC 0.583. Per-video AUC is
    identical with and without the content-free term, because that term is
    a per-video constant. This agrees with GlitchBench (Taesiri et al., CVPR 2024):
    LMMs are poor at OPEN glitch detection.
  * The self-reference assumption of v9 is false for this benchmark: the window
    base rate is 0.518, the median video is 50% glitch and 47% of videos are
    more than half glitch. Median normalisation (C2), 'normal-half' references
    (C3) and the HMM prior all assume that glitches are rare within a video.
  * C3 (pairwise) spent 67% of all calls for an AUC CI that contains 0.5;
    the HMM LOWERED the AUC (0.483 vs 0.529) with gamma pinned at the grid
    minimum and a log-loss of ln 2, i.e. uninformative emissions.
  * With ~50% coverage the trivial span "whole video" already reaches IoU ~0.5;
    v9 never reported that baseline.

v10 method: turn open detection into calibrated verification of hypotheses
---------------------------------------------------------------------------
  H  Hypothesise: every window is described with an explicit "no glitch" option;
     observations are merged into entity-anchored hypotheses and ranked by
     SUPPORT (how many windows produced them independently; self-consistency,
     Wang et al., ICLR 2023). If nothing is reported, one forced hypothesis from
     the top generic window is kept and the selector decides.
  V  Verify: for hypothesis h and every window w
        s_h(w) = logit P(Yes | w, "is <h> visible?") - logit P(Yes | grey, same)
     A specific, entity-anchored question is a verification task, which VLMs
     do far better than open anomaly detection. The subtracted term is h's own
     prompt prior (contextual calibration per query, Zhao et al., ICML 2021),
     so profiles of different hypotheses share one scale.
  S  Segment: exact O(T) dynamic programme for
        argmax over non-empty sets of disjoint segments of
        sum_k sum_{t in seg_k} (s_h(t) - tau)  -  lambda (K - 1),
     i.e. penalised-likelihood change-point segmentation (Kadane when
     lambda -> inf). tau, lambda are fitted on correct hypotheses by IoU with
     game-grouped cross-validation. Boundary windows are refined to frames by
     masked-frame likelihood bisection (from v9). The whole-video span competes
     as a refine mode, so the calibration output SAYS when localisation does
     not beat the trivial baseline.
  U  Select: a fractional-response logistic model (Papke & Wooldridge, 1996)
     predicts u_h = E[rating/5 x IoU] from profile, support and prior
     features. Labels come from the official protocol on dev: Hungarian on judge
     ratings, then IoU. Because the official F1xIoU is 2 S / (5 (N + M)),
     adding a prediction helps iff u_h > F*/2. This generalises the result of
     Lipton, Elkan & Naryanaswamy (2014) that the F1-optimal threshold is
     half the optimal F1. The threshold is the exact maximiser on out-of-fold
     predictions. The selector is declared degenerate unless it ranks correct
     hypotheses above chance (game-cluster CI of the out-of-fold AUC > 0.5).

  Statistics (kept from v9): exact / Monte-Carlo game sign-flip permutation test
  with Holm correction; game-cluster bootstrap CIs; official evaluator port.
  New diagnostics: ICC(1) variance decomposition and within-video (median-
  centred) AUC with game-cluster CIs, i.e. the numbers of the v9 probe, computed
  reproducibly for every evidence type; paired per-video AUC test of the
  hypothesis profile against the generic detector.

Honest scope: nothing is trained on pixels. The self-test verifies mechanics
(exactness of the segmentation DP, the F*/2 property, ICC recovery, test
validity) with a mock that reproduces the v9 probe pattern; whether a real
VLM verifies hypotheses better than it detects glitches is exactly what
'probe' measures, and must be established before any full run.

Commands
--------
  python crux_v10.py selftest [--glide_repo /path/to/GliDe]
  python crux_v10.py diagnose  --split dev --limit 72     # generic detector only (cached)
  python crux_v10.py probe     --split dev --limit 72     # v10 go/no-go (no judge needed)
  python crux_v10.py calibrate --calib_splits dev         # needs the judge server
  python crux_v10.py run       --method crux    --split test
  python crux_v10.py run       --method vanilla --split test
  python crux_v10.py import_glide --batch_report /path/batch_report.json --split test
  python crux_v10.py export    --run_file <runs/..../crux_test.json>
  python crux_v10.py evaluate  --split test --run_files crux_test.json glide_official_test.json
  python crux_v10.py ablate    --split test

--limit always samples round-robin over games (--no-stratify_limit disables it).

Fair-comparison recipe (same backbone, same server, same judge, same videos)
  1. vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --max-logprobs 20
  2. official GliDe, unmodified:
       cd GliDe && python run.py --video-dir <root>/raw --api-base http://localhost:8000/v1 \\
                                 --model Qwen/Qwen2.5-VL-7B-Instruct
  3. python crux_v10.py import_glide --split test --batch_report GliDe/data/results/batch_report.json
  4. python crux_v10.py probe --split dev --limit 72     (stop if the verdict is NO-GO)
  5. vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001
  6. python crux_v10.py calibrate --calib_splits dev
  7. python crux_v10.py run --method crux --split test ; run --method vanilla --split test
  8. python crux_v10.py evaluate --split test --gt_json <official groundtruth.json> \\
       --run_files crux_test.json glide_official_test.json vanilla_test.json
     (first file = reference; evaluation uses the intersection of video sets)

The work directory defaults to crux_v8_work so that the windows and the generic
detector calls cached by v8/v9 are reused (identical prompts); v10 writes its
calibrations, runs and evaluations under a separate 'v10' sub-tree.

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
VERSION = "10.0"

# Backbones for the comparison matrix. With backend=openai the same names are
# what the vLLM server must be serving (e.g. `vllm serve Qwen/Qwen2.5-VL-7B-Instruct`).
BACKBONES: Dict[str, Dict[str, str]] = {
    "qwen2.5-vl-3b": {"hf": "Qwen/Qwen2.5-VL-3B-Instruct", "family": "qwen"},
    "qwen2.5-vl-7b": {"hf": "Qwen/Qwen2.5-VL-7B-Instruct", "family": "qwen"},
    "internvl2.5-4b": {"hf": "OpenGVLab/InternVL2_5-4B", "family": "internvl"},
    "internvl2.5-8b": {"hf": "OpenGVLab/InternVL2_5-8B", "family": "internvl"},
}
# Hypothesis-level features used by the utility model U (section "U" below).
HYP_FEATURES = ("support", "rank_inv", "forced", "prior", "prof_max", "prof_top2",
                "prof_mean", "prof_contrast", "seg_frac", "gen_lo", "gen_z")
REFINE_MODES = ("window", "frame", "whole")
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
    work_subdir: str = "crux_v8_work"   # shared with v8/v9: windows and VLM caches are reused
    metadata_name: str = "annotations.json"
    gt_json: str = ""            # official groundtruth.json; used verbatim by evaluate
    glide_repo: str = ""         # optional clone of the official GliDe repo

    # ---- preprocessing: official GliDe defaults (config.py) ------------------
    fps: float = 4.0
    window_size: int = 8
    jpeg_quality: int = 95       # official writes frames and stitches at q=95
    pair_max_side: int = 2688    # downscaled copies (used by the vanilla baseline)

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
    desc_max_tokens: int = 256
    summary_max_tokens: int = 512
    top_logprobs: int = 20

    # ---- evaluation judge: official protocol ------------------------------------
    judge_backend: str = "openai"    # 'openai' | 'hf' | 'mock'
    judge_api_base: str = "http://localhost:8001/v1"
    judge_api_key: str = "EMPTY"
    judge_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    judge_temperature: float = 0.3   # Evaluator default in the official repo
    judge_max_tokens: int = 512

    # ---- CRUX v10 components (each is an ablation switch) -------------------------
    use_contextual_calibration: bool = True   # per-query content-free prior removal
    use_hypothesis_scoring: bool = True       # V: hypothesis-conditioned profile (else generic)
    use_event_resolution: bool = True         # merge window observations into hypotheses
    use_segmentation: bool = True             # S: exact penalised segmentation (else member runs)
    use_subframe: bool = True                 # frame-level masked likelihood bisection
    use_verified_bisection: bool = True
    use_utility_selection: bool = True        # U: F1xIoU-optimal hypothesis selection

    describe_budget: int = 0      # windows described per video (0 = all; else top-k by lo_abs)
    max_hypotheses: int = 3       # hypotheses verified per video (ranked by support)
    max_intervals_per_event: int = 4
    max_events_per_video: int = 6
    seg_taus: str = "-1.0,-0.5,0.0,0.5,1.0,1.5,2.0"   # grid for the presence level tau
    seg_penalties: str = "0.0,0.5,1.0,2.0,4.0,1e9"    # grid for the segment-start cost lambda
    seg_tau: float = 0.0          # fitted by 'calibrate'
    seg_penalty: float = 1.0      # fitted by 'calibrate'
    refine_mode: str = "frame"    # 'window' | 'frame' | 'whole'; fitted by 'calibrate'
    min_correct_rating: int = 3   # judge rating that counts a hypothesis as correct

    # ---- selection / calibration ---------------------------------------------------
    calib_label: str = "judge"    # 'judge' (official judge on dev) | 'iou' (description-blind)
    label_min_overlap: float = 0.25
    fusion_l2: float = 1.0
    cv_folds: int = 5
    decision_threshold: float = 0.0   # utility threshold, fitted by 'calibrate'
    fusion: Dict[str, Any] = field(default_factory=dict)
    min_auc_ci_lo: float = 0.5
    fail_on_degenerate: bool = True
    quantize_seconds: bool = False   # emulate GliDe's int(frame // fps) output
    stratify_limit: bool = True      # --limit samples round-robin over games

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
                "runs": os.path.join(w, "v10", "runs", b),
                "params": os.path.join(w, "v10", "params", b),
                "eval": os.path.join(w, "v10", "eval", b)}

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
    fh = logging.FileHandler(os.path.join(work_dir, f"crux_v10_{tag}.log"))
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


def take_limit(recs: Sequence[Record], limit: Optional[int], stratify: bool = True
               ) -> List[Record]:
    """First `limit` records, drawn round-robin over games when stratify is set.
    Split files are usually grouped by game, so a plain prefix of 60 videos can
    contain one or two games and makes game-grouped cross-fitting impossible."""
    recs = list(recs)
    if not limit or limit >= len(recs):
        return recs
    if not stratify:
        return recs[:limit]
    by: Dict[str, List[Record]] = defaultdict(list)
    for r in recs:
        by[r.game].append(r)
    games, out, k = sorted(by), [], 0
    while len(out) < limit:
        for g in games:
            if k < len(by[g]) and len(out) < limit:
                out.append(by[g][k])
        k += 1
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
P_DESC_OPEN_SYS = (
    "You are a video game QA analyst. The image is a stitched sequence of consecutive "
    "gameplay frames labelled '#<frame index>'. " + _GLITCH_DEF + " If no glitch is visible, "
    "say so; do not invent one. If a glitch is visible, name the affected object or "
    "character by its visual appearance and state exactly what is abnormal, in 1-2 "
    "sentences, without frame numbers. Reply with one JSON object only, keys: glitch "
    "(true or false), description, entity, glitch_type.")
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


def presence_user(game: str, desc: str, entity: str) -> str:
    return (f"Game: {game}.\nGlitch: {desc}\nAffected entity: {entity}\n"
            f"Is this same glitch visibly occurring in this window? "
            f"Answer with exactly one word: Yes or No.")


# =============================================================================
# E: generic evidence (C1, C2). Weak on VideoGlitchBench (see module docstring),
#    kept only because it is cheap, cached from v8/v9, and useful as a feature.
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


@dataclass
class GenericEvidence:
    lo_raw: List[float]
    lo_cf: float
    lo_abs: List[float]
    rel_z: List[float]


def collect_generic_evidence(vlm: VLMBase, rec: Record, wins: Sequence[Window],
                             cfg: Config) -> GenericEvidence:
    usr = detect_user(rec.game)

    def lo_of(path: str, stage: str) -> float:
        return binary_log_odds(vlm.score(P_DETECT_SYS, usr, [path], YES_NO, stage), "yes", "no")

    lo_raw = np.array([lo_of(w.stitch_path, "c1_detect") for w in wins])
    lo_cf = 0.0
    if cfg.use_contextual_calibration:
        lo_cf = lo_of(content_free_path(wins) if vlm.needs_pixels else "__content_free__",
                      "c1_calib")
    lo_abs = lo_raw - lo_cf
    return GenericEvidence(lo_raw.tolist(), float(lo_cf), lo_abs.tolist(),
                           robust_z(lo_abs).tolist())


# =============================================================================
# H + V: hypotheses and hypothesis-conditioned likelihood profiles
#   H  every window is described with an explicit "no glitch" option; the
#      observations are merged into hypotheses (entity + anomaly), ranked by
#      support (how many windows independently produced them).
#   V  for hypothesis h and window w:
#        s_h(w) = logit P(Yes | w, "is <h> visible?") - logit P(Yes | grey, same question)
#      The second term is h's own prompt prior (contextual calibration per query),
#      so profiles of different hypotheses are on a common scale.
# =============================================================================

@dataclass
class Hypothesis:
    description: str
    entity: str
    glitch_type: str
    members: List[int]
    forced: bool
    prior: float
    profile: List[float]


@dataclass
class VideoHypotheses:
    vid: str
    generic: GenericEvidence
    hyps: List[Hypothesis]
    n_described: int


def verify_profile(vlm: VLMBase, rec: Record, wins: Sequence[Window], desc: str, entity: str,
                   cfg: Config) -> Tuple[float, List[float]]:
    usr = presence_user(rec.game, desc, entity)

    def lo_of(path: str, stage: str) -> float:
        return binary_log_odds(vlm.score(P_PRESENCE_SYS, usr, [path], YES_NO, stage), "yes", "no")

    prior = 0.0
    if cfg.use_contextual_calibration:
        prior = lo_of(content_free_path(wins) if vlm.needs_pixels else "__content_free__",
                      "v_calib")
    return prior, [lo_of(w.stitch_path, "v_profile") - prior for w in wins]


def collect_video(vlm: VLMBase, rec: Record, wins: Sequence[Window], cfg: Config
                  ) -> VideoHypotheses:
    gen = collect_generic_evidence(vlm, rec, wins, cfg)
    lo = np.asarray(gen.lo_abs)
    order = list(range(len(wins)))
    if cfg.describe_budget and cfg.describe_budget < len(wins):
        order = sorted(int(i) for i in np.argsort(-lo, kind="mergesort")[:cfg.describe_budget])
    obs = []
    for i in order:
        d = describe_window_open(vlm, wins[i], rec.game)
        if d is not None:
            obs.append({"window": i, **d})
    forced = not obs
    if forced:
        # The model reports no glitch anywhere. Keep one forced hypothesis from
        # the window the generic detector ranks highest; U decides whether it is
        # worth emitting (the 'forced' feature lets it learn how often it is).
        i = int(np.argmax(lo)) if lo.size else 0
        obs = [{"window": i, **describe_window(vlm, wins[i], rec.game)}]
    events = resolve_events(vlm, obs, cfg)
    rz = np.asarray(gen.rel_z)
    for e in events:
        e["members"] = sorted({obs[k]["window"] for k in e["member_ids"]})
        e["support"] = len(e["members"])
        e["peak_z"] = float(max(rz[m] for m in e["members"])) if rz.size else 0.0
    events.sort(key=lambda e: (-e["support"], -e["peak_z"], e["members"][0]))
    hyps = []
    for e in events[:cfg.max_hypotheses]:
        descs = [obs[k]["description"] for k in e["member_ids"]]
        desc = summarise_event(vlm, descs, e["glitch_type"], windows_to_spans(e["members"], wins))
        if cfg.use_hypothesis_scoring:
            prior, prof = verify_profile(vlm, rec, wins, desc, e["canonical_entity"], cfg)
        else:
            prior, prof = 0.0, list(gen.lo_abs)
        hyps.append(Hypothesis(desc, e["canonical_entity"], e["glitch_type"], e["members"],
                               forced, float(prior), [float(x) for x in prof]))
    return VideoHypotheses(rec.vid, gen, hyps, len(order))


# =============================================================================
# S: exact penalised segmentation of a likelihood profile
#   argmax over NON-EMPTY sets of disjoint segments of
#        sum_k sum_{t in seg_k} (s_t - tau)  -  lambda * (K - 1)
#   tau is the presence level (log-odds units), lambda the cost of every segment
#   after the first. Exact O(T) dynamic programme with three states: before the
#   first segment, inside a segment, between segments. lambda -> inf gives the
#   single maximum-sum segment (Kadane); lambda = 0 gives every run with s > tau.
# =============================================================================

def segment_profile(s: Sequence[float], tau: float, lam: float) -> List[Tuple[int, int]]:
    x = np.asarray(s, float) - float(tau)
    T = x.size
    if T == 0:
        return []
    NEG = -np.inf
    V = np.full((T, 3), NEG)          # 0 before, 1 inside, 2 between
    bp = np.zeros((T, 3), int)
    V[0, 0], V[0, 1] = 0.0, x[0]
    for t in range(1, T):
        V[t, 0] = 0.0
        cand = (V[t - 1, 1], V[t - 1, 0], V[t - 1, 2] - lam)    # continue, first, new
        k = int(np.argmax(cand))
        V[t, 1], bp[t, 1] = cand[k] + x[t], (1, 0, 2)[k]
        if V[t - 1, 2] >= V[t - 1, 1]:
            V[t, 2], bp[t, 2] = V[t - 1, 2], 2
        else:
            V[t, 2], bp[t, 2] = V[t - 1, 1], 1
    st = 1 if V[T - 1, 1] >= V[T - 1, 2] else 2
    inside = np.zeros(T, bool)
    for t in range(T - 1, -1, -1):
        inside[t] = st == 1
        if t > 0:
            st = bp[t, st] if st != 0 else 0
    segs, a = [], None
    for t in range(T):
        if inside[t] and a is None:
            a = t
        if a is not None and (not inside[t] or t == T - 1):
            segs.append((a, t if inside[t] else t - 1))
            a = None
    return segs


def segment_score(s: Sequence[float], segs: Sequence[Tuple[int, int]], tau: float,
                  lam: float) -> float:
    x = np.asarray(s, float) - tau
    return float(sum(x[a:b + 1].sum() for a, b in segs) - lam * max(0, len(segs) - 1))


def member_runs(members: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for m in sorted(set(members)):
        if out and m == out[-1][1] + 1:
            out[-1] = (out[-1][0], m)
        else:
            out.append((m, m))
    return out


def hypothesis_segments(h: Hypothesis, cfg: Config, tau: float, lam: float
                        ) -> List[Tuple[int, int]]:
    if not cfg.use_segmentation:
        return member_runs(h.members)
    segs = segment_profile(h.profile, tau, lam)
    if len(segs) > cfg.max_intervals_per_event:
        x = np.asarray(h.profile) - tau
        segs = sorted(sorted(segs, key=lambda ab: -x[ab[0]:ab[1] + 1].sum())
                      [:cfg.max_intervals_per_event])
    return segs


def hypothesis_features(h: Hypothesis, gen: GenericEvidence, rank: int, cfg: Config,
                        tau: float, lam: float) -> Dict[str, float]:
    s = np.asarray(h.profile, float)
    n = max(1, s.size)
    segs = hypothesis_segments(h, cfg, tau, lam)
    inside = np.zeros(s.size, bool)
    for a, b in segs:
        inside[a:b + 1] = True
    contrast = float(s[inside].mean() - s[~inside].mean()) if inside.any() and (~inside).any() else 0.0
    lo, rz = np.asarray(gen.lo_abs), np.asarray(gen.rel_z)
    top = np.sort(s)[::-1]
    return {"support": len(h.members) / n, "rank_inv": 1.0 / rank, "forced": float(h.forced),
            "prior": h.prior, "prof_max": float(top[0]) if s.size else 0.0,
            "prof_top2": float(top[:2].mean()) if s.size else 0.0,
            "prof_mean": float(s.mean()) if s.size else 0.0, "prof_contrast": contrast,
            "seg_frac": float(inside.mean()) if s.size else 0.0,
            "gen_lo": float(lo[h.members].mean()) if lo.size else 0.0,
            "gen_z": float(rz[h.members].max()) if rz.size else 0.0}


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


def icc_oneway(values: Sequence[Sequence[float]]) -> Dict[str, float]:
    """One-way random-effects ANOVA decomposition of a window score into a
    between-video and a within-video component (unbalanced groups, ANOVA
    estimator with the n0 correction). ICC(1) near 1 means the score is a
    property of the VIDEO and cannot localise anything inside it."""
    groups = [np.asarray(v, float) for v in values if len(v) >= 2]
    if len(groups) < 2:
        return {"icc": float("nan"), "sd_within": float("nan"), "sd_between": float("nan"),
                "n_videos": len(groups)}
    k, N = len(groups), sum(g.size for g in groups)
    grand = float(np.concatenate(groups).mean())
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - sum(g.size ** 2 for g in groups) / N) / (k - 1)
    vb = max(0.0, (msb - msw) / n0)
    return {"icc": float(vb / (vb + msw)) if vb + msw > 0 else float("nan"),
            "sd_within": float(math.sqrt(msw)), "sd_between": float(math.sqrt(vb)),
            "n_videos": k}


def within_video_scores(scores: Sequence[Sequence[float]]) -> List[np.ndarray]:
    """Centre every video's scores on its median: a pooled AUC of the centred
    scores measures WITHIN-video discrimination only (the 'when' signal)."""
    return [np.asarray(v, float) - (float(np.median(v)) if len(v) else 0.0) for v in scores]


def per_video_auc(scores: Sequence[Sequence[float]], labels: Sequence[Sequence[int]]) -> np.ndarray:
    out = [roc_auc(s, y) for s, y in zip(scores, labels)]
    return np.asarray([a for a in out if not math.isnan(a)], float)


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


# =============================================================================
# G: frame-level grounding with a likelihood presence oracle
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


class PresenceOracle:
    """Event-specific likelihood oracle: present iff the calibrated log-odds of
    'this same glitch is visibly occurring' exceeds the margin. Callable on a
    window index; .image(path) scores any derived image of the same geometry."""

    def __init__(self, vlm: VLMBase, wins: Sequence[Window], game: str, desc: str,
                 entity: str, cfg: Config, margin: float = 0.0):
        self.vlm, self.cfg, self.margin = vlm, cfg, float(margin)
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
            self.memo[path] = (lo - self.base) > self.margin
        return self.memo[path]

    def __call__(self, i: int) -> bool:
        w = self.by_idx.get(i)
        return False if w is None else self.image(w.stitch_path, "g_presence")


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


def windows_to_spans(members: Sequence[int], wins: Sequence[Window]) -> List[Interval]:
    by = {w.idx: w for w in wins}
    return normalize_intervals([(by[m].start, by[m].end) for m in members if m in by])


def hypothesis_spans(vlm: VLMBase, rec: Record, wins: Sequence[Window], h: Hypothesis,
                     cfg: Config, tau: float, lam: float, mode: str
                     ) -> Tuple[List[Interval], Dict[str, int]]:
    """Time spans of a hypothesis under a refine mode:
         window  segments of the exact segmentation, at window resolution
         frame   same segments, boundaries refined by masked-frame bisection with
                 the hypothesis' own presence oracle at level tau
         whole   the entire video (the trivial baseline; kept as a candidate so
                 that calibration can report honestly when localisation does not
                 beat it)"""
    audit = {"monotonicity_violations": 0, "frame_conflicts": 0}
    dur = float(rec.duration or (wins[-1].end if wins else 0.0))
    if mode == "whole" or not wins:
        return ([(0.0, dur)] if dur > 0 else []), audit
    segs = hypothesis_segments(h, cfg, tau, lam)
    refine = mode == "frame" and cfg.use_subframe and cfg.use_hypothesis_scoring
    oracle: Optional[PresenceOracle] = None
    out = []
    for a, b in segs:
        lo_t, hi_t = wins[a].start, wins[b].end
        if refine:
            oracle = oracle or PresenceOracle(vlm, wins, rec.game, h.description, h.entity,
                                              cfg, tau)
            v = cfg.use_verified_bisection
            on_t, off_t = lo_t, hi_t
            if h.profile[a] > tau:
                k = frame_onset(oracle, wins[a], audit, v)
                on_t = frame_time_bounds(wins[a], k, k, cfg.fps)[0]
            if h.profile[b] > tau:
                k = frame_offset(oracle, wins[b], audit, v)
                off_t = frame_time_bounds(wins[b], k, k, cfg.fps)[1]
            if off_t > on_t:
                lo_t, hi_t = on_t, off_t
            else:
                audit["frame_conflicts"] += 1
        out.append((clamp(lo_t, 0.0, dur), clamp(hi_t, 0.0, dur)))
    return normalize_intervals(out), audit


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


def describe_window_open(vlm: VLMBase, win: Window, game: str) -> Optional[Dict[str, str]]:
    """Open description with an explicit no-glitch option. Returns None when the
    model reports no glitch or returns nothing usable."""
    obj = vlm.generate_json(P_DESC_OPEN_SYS, f"Game: {game}. Is there a glitch in this window?",
                            [win.stitch_path], 256, "describe") or {}
    g = obj.get("glitch")
    if isinstance(g, str):
        g = g.strip().lower() in ("true", "yes", "1")
    d = str(obj.get("description", "") or "").strip()
    if not g or not d:
        return None
    return {"description": d, "entity": str(obj.get("entity", "") or "entity"),
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

def hypothesis_rows(vh: VideoHypotheses, cfg: Config, tau: float, lam: float
                    ) -> List[Dict[str, float]]:
    return [hypothesis_features(h, vh.generic, k + 1, cfg, tau, lam) for k, h in enumerate(vh.hyps)]


def run_crux(vlm: VLMBase, rec: Record, wins: Sequence[Window], cfg: Config,
             vh: Optional[VideoHypotheses] = None) -> Dict[str, Any]:
    require(not cfg.use_utility_selection or str(cfg.fusion.get("version", "")).startswith("10"),
            "run_crux needs a CRUX v10 utility model (run 'calibrate')")
    vh = vh or collect_video(vlm, rec, wins, cfg)
    tau, lam = cfg.seg_tau, cfg.seg_penalty
    rows = hypothesis_rows(vh, cfg, tau, lam)
    util = (fusion_predict(cfg.fusion, rows) if rows and cfg.use_utility_selection
            else np.ones(len(rows)))
    keep = [i for i in np.argsort(-util, kind="mergesort")
            if not cfg.use_utility_selection or util[i] >= cfg.decision_threshold]
    keep = [int(i) for i in keep][:cfg.max_events_per_video]
    out: Dict[str, Any] = {"vid": rec.vid, "video_name": rec.video_name, "game": rec.game,
                           "n_windows": len(wins), "n_described": vh.n_described,
                           "hypotheses": [{"description": h.description, "entity": h.entity,
                                           "members": h.members, "forced": h.forced,
                                           "prior": h.prior, "profile": h.profile,
                                           "utility": float(u), **r}
                                          for h, u, r in zip(vh.hyps, util, rows)],
                           "monotonicity_violations": 0, "frame_conflicts": 0}
    reps = []
    for i in keep:
        h = vh.hyps[i]
        spans, audit = hypothesis_spans(vlm, rec, wins, h, cfg, tau, lam, cfg.refine_mode)
        for k in audit:
            out[k] += audit[k]
        if spans:
            reps.append({"description": h.description, "spans": spans, "utility": float(util[i]),
                         "entity": h.entity})
    out.update({"no_bugs": not reps, "reports": reps, "bugs": [r["description"] for r in reps],
                "time_nodes": [to_time_nodes(r["spans"], cfg) for r in reps]})
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
# Evidence collection, diagnostics and calibration
# =============================================================================

def gather(cfg: Config, vlm: VLMBase, recs: Sequence[Record], wins_map: Dict[str, List[Window]],
           hypotheses: bool = True) -> Dict[str, Any]:
    out, t0 = {}, time.time()
    for i, r in enumerate(recs):
        w = wins_map[r.vid]
        out[r.vid] = (collect_video(vlm, r, w, cfg) if hypotheses
                      else collect_generic_evidence(vlm, r, w, cfg))
        if (i + 1) % 25 == 0:
            LOG.info("evidence %d/%d videos | %d logical calls (%d run, %d cached, %d "
                     "top-logprob truncations) | %.1f min", i + 1, len(recs), vlm.budget.calls,
                     vlm.budget.model_calls, vlm.budget.cache_hits,
                     vlm.budget.logprob_truncations, (time.time() - t0) / 60)
    return out


def _signal_report(name: str, scores: List[np.ndarray], labels: List[np.ndarray],
                   games: List[str], cfg: Config) -> Dict[str, Any]:
    s_all = np.concatenate(scores)
    y_all = np.concatenate(labels)
    g_all = [g for g, v in zip(games, scores) for _ in range(len(v))]
    B = min(cfg.n_bootstrap, 1000)
    pooled = cluster_bootstrap_auc(s_all, y_all, g_all, B, cfg.seed)
    within = cluster_bootstrap_auc(np.concatenate(within_video_scores(scores)), y_all, g_all, B,
                                   cfg.seed + 1)
    pv = per_video_auc(scores, labels)
    return {"name": name, "pooled_auc": pooled, "within_video_auc": within,
            "per_video_auc_median": float(np.median(pv)) if pv.size else float("nan"),
            "per_video_auc_mean": float(pv.mean()) if pv.size else float("nan"),
            "n_videos_both_classes": int(pv.size), "variance": icc_oneway(scores)}


def cmd_diagnose(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]], split: str,
                 hypotheses: bool) -> Dict[str, Any]:
    """Signal audit on one split (never 'test'). Without hypotheses it only reads
    the generic detector, which is cached from v8/v9, so it costs no model calls
    on a warm cache. With hypotheses it is the v10 go/no-go probe: is the
    hypothesis-conditioned profile a better WITHIN-video ('when') signal than
    the generic detector on the same windows?"""
    require(split != "test", "diagnostics must not touch the test split")
    recs = take_limit(splits[split], cfg.limit, cfg.stratify_limit)
    LOG.info("diagnose: %d videos over %d games", len(recs), len({r.game for r in recs}))
    wm = prepare_split(recs, cfg)
    ev = gather(cfg, vlm, recs, wm, hypotheses)
    labels = [window_labels(r, wm[r.vid], cfg.label_min_overlap) for r in recs]
    games = [r.game for r in recs]
    gen = [e.generic if hypotheses else e for e in (ev[r.vid] for r in recs)]
    series = {"lo_raw": [np.asarray(g.lo_raw) for g in gen],
              "lo_abs": [np.asarray(g.lo_abs) for g in gen]}
    if hypotheses:
        series["hyp_max"] = [np.max(np.stack([h.profile for h in ev[r.vid].hyps]), axis=0)
                             if ev[r.vid].hyps else np.asarray(ev[r.vid].generic.lo_abs)
                             for r in recs]
        series["hyp_top1"] = [np.asarray(ev[r.vid].hyps[0].profile) if ev[r.vid].hyps
                              else np.asarray(ev[r.vid].generic.lo_abs) for r in recs]
    rep = {k: _signal_report(k, v, labels, games, cfg) for k, v in series.items()}
    y_all = np.concatenate(labels)
    pos = np.array([float(y.mean()) if y.size else 0.0 for y in labels])
    whole = []
    for r in recs:
        gt = normalize_intervals([iv for sp in r.time_nodes for iv in sp])
        if gt:
            whole.append(official_iou([list(x) for x in gt], [[0.0, float(r.duration)]]))
    base = {"n_windows": int(y_all.size), "base_rate": float(y_all.mean()),
            "median_video_positive_fraction": float(np.median(pos)),
            "videos_over_half_positive": float(np.mean(pos > 0.5)),
            "whole_video_span_iou_mean": float(np.mean(whole)) if whole else float("nan")}
    LOG.info("windows %d | base rate %.3f | median positive fraction per video %.3f | %.1f%% "
             "of videos over half glitch", base["n_windows"], base["base_rate"],
             base["median_video_positive_fraction"], 100 * base["videos_over_half_positive"])
    LOG.info("TRIVIAL GROUNDING BASELINE: span = whole video gives mean IoU %.3f with the "
             "union of GT spans; any localisation must beat this", base["whole_video_span_iou_mean"])
    for k, d in rep.items():
        v = d["variance"]
        LOG.info("%-8s pooled AUC %.3f [%.3f, %.3f] | within-video AUC %.3f [%.3f, %.3f] | "
                 "per-video AUC median %.3f (n=%d) | sd within %.3f between %.3f ICC %.3f", k,
                 d["pooled_auc"]["auc"], d["pooled_auc"]["lo"], d["pooled_auc"]["hi"],
                 d["within_video_auc"]["auc"], d["within_video_auc"]["lo"],
                 d["within_video_auc"]["hi"], d["per_video_auc_median"],
                 d["n_videos_both_classes"], v["sd_within"], v["sd_between"], v["icc"])
    cmp_ = {}
    if hypotheses:
        a_h = [roc_auc(s, y) for s, y in zip(series["hyp_max"], labels)]
        a_g = [roc_auc(s, y) for s, y in zip(series["lo_abs"], labels)]
        keep = [i for i in range(len(recs)) if not (math.isnan(a_h[i]) or math.isnan(a_g[i]))]
        d = np.array([a_h[i] - a_g[i] for i in keep])
        gk = [games[i] for i in keep]
        if d.size >= 10 and np.any(np.abs(d) > 1e-12):
            _, pw = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        else:
            pw = 1.0
        rng = np.random.default_rng(cfg.seed)
        gi: Dict[str, List[int]] = defaultdict(list)
        for i, g in enumerate(gk):
            gi[g].append(i)
        units = list(gi.values())
        boot = [float(np.mean(np.concatenate([d[units[u]] for u in
                                              rng.integers(0, len(units), len(units))])))
                for _ in range(min(cfg.n_bootstrap, 2000))] if units else [float("nan")]
        cmp_ = {"mean_delta_per_video_auc": float(d.mean()) if d.size else float("nan"),
                "game_bootstrap_ci": [float(np.percentile(boot, 2.5)),
                                      float(np.percentile(boot, 97.5))],
                "wilcoxon_p": float(pw), "n_videos": int(d.size)}
        LOG.info("hyp_max vs lo_abs per-video AUC: delta %+.3f, game-bootstrap 95%% CI "
                 "[%+.3f, %+.3f], Wilcoxon p=%.4f (n=%d)", cmp_["mean_delta_per_video_auc"],
                 *cmp_["game_bootstrap_ci"], cmp_["wilcoxon_p"], cmp_["n_videos"])
    key = "hyp_max" if hypotheses else "lo_abs"
    w = rep[key]["within_video_auc"]
    go = not math.isnan(w["lo"]) and w["lo"] > 0.5
    LOG.info("VERDICT (%s within-video AUC): %s", key,
             "GO -- the score localises glitches inside videos" if go else
             "NO-GO -- no evidence that the score localises glitches inside videos")
    out = {"split": split, "n_videos": len(recs), "hypotheses": hypotheses, "base": base,
           "signals": rep, "hyp_vs_generic": cmp_, "go": go, "budget": asdict(vlm.budget)}
    atomic_write_json(os.path.join(cfg.paths()["params"],
                                   f"{'probe' if hypotheses else 'diagnose'}_{split}.json"), out)
    return out


def parse_grid(txt: str) -> List[float]:
    g = [float(x) for x in str(txt).split(",") if x.strip()]
    require(g, f"empty grid '{txt}'")
    return g


def gt_bug_list(rec: Record) -> List[Dict[str, Any]]:
    return [{"description": b, "spans": [list(iv) for iv in rec.time_nodes[i]]}
            for i, b in enumerate(rec.bugs) if i < len(rec.time_nodes) and rec.time_nodes[i]]


def match_hypotheses(cfg: Config, judge: Optional[JudgeBase], rec: Record, wins: Sequence[Window],
                     vh: VideoHypotheses) -> Dict[int, Tuple[int, int]]:
    """Official protocol applied to the CANDIDATE set: Hungarian assignment on
    the judge's 0-5 rating matrix (spans play no role in the matching, exactly
    as in evaluation/evaluator.py). Returns {hyp_idx: (gt_idx, rating)}.
    calib_label='iou' is a description-blind fallback when no judge is served:
    Hungarian on window-level IoU, rating 5 for any overlap. It ignores what
    the judge rewards and is only for debugging the pipeline."""
    gts = gt_bug_list(rec)
    if not gts or not vh.hyps:
        return {}
    R = np.zeros((len(vh.hyps), len(gts)))
    if cfg.calib_label == "judge":
        require(judge is not None, "calib_label=judge needs a judge")
        for i, h in enumerate(vh.hyps):
            for j, g in enumerate(gts):
                R[i, j] = judge.rate(g["description"], h.description)
    else:
        for i, h in enumerate(vh.hyps):
            sp = windows_to_spans([t for a, b in hypothesis_segments(h, cfg, 0.0, 1.0)
                                   for t in range(a, b + 1)], wins)
            for j, g in enumerate(gts):
                R[i, j] = official_iou(g["spans"], [list(x) for x in sp])
    pi, gi = linear_sum_assignment(-R)
    if cfg.calib_label == "judge":
        return {int(a): (int(b), int(R[a, b])) for a, b in zip(pi, gi)}
    return {int(a): (int(b), 5 if R[a, b] > 0 else 0) for a, b in zip(pi, gi)}


def select_f1_threshold(u_hat: np.ndarray, u: np.ndarray, M: int) -> Dict[str, Any]:
    """Exact maximiser over thresholds of the plug-in dataset F1xIoU
         F(t) = 2 sum_{u_hat >= t} u / (N(t) + M),   u = rating/5 * IoU in [0, 1].
    Lipton, Elkan & Naryanaswamy (2014) showed that for calibrated probabilities
    the F1-optimal threshold is F*/2; the same argument with graded utilities
    gives 'emit iff E[u] > F*/2', reported as a calibration check."""
    order = np.argsort(-u_hat, kind="mergesort")
    cs = np.cumsum(u[order])
    Ns = np.arange(1, u.size + 1)
    F = 2 * cs / (Ns + M) if M > 0 else np.zeros(u.size)
    if F.size == 0 or F.max() <= 0:
        return {"threshold": float("inf"), "f1_iou": 0.0, "n_emitted": 0, "lipton_ratio": float("nan")}
    k = int(np.argmax(F))
    t = float(u_hat[order[k]])
    return {"threshold": t, "f1_iou": float(F[k]), "n_emitted": int(k + 1),
            "lipton_ratio": float(t / F[k]) if F[k] > 0 else float("nan")}


def _const_util_model(feats: Sequence[str], p: float) -> Dict[str, Any]:
    p = clamp(p, 1e-4, 1 - 1e-4)
    return {"features": list(feats), "mu": [0.0] * len(feats), "sd": [1.0] * len(feats),
            "w": [0.0] * len(feats), "b": float(math.log(p / (1 - p))), "l2": 0.0}


def fit_utility(rows: Sequence[Dict[str, float]], u: np.ndarray, cfg: Config) -> Dict[str, Any]:
    """Fractional-response logistic regression (Papke & Wooldridge 1996): the
    Bernoulli quasi-likelihood with targets in [0, 1] is consistent for
    E[u | x]; IRLS as in fit_logistic."""
    if len(rows) < 5 or float(np.max(u)) <= 0 or float(np.min(u)) >= 1:
        return _const_util_model(HYP_FEATURES, float(np.mean(u)) if len(u) else 0.0)
    m = fusion_fit(rows, u, HYP_FEATURES, cfg.fusion_l2)
    m["version"] = VERSION
    return m


def _iou_of(spans: Sequence[Interval], gt: Sequence[Sequence[float]]) -> float:
    return official_iou([list(x) for x in gt], [list(x) for x in spans])


def cmd_calibrate(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]],
                  calib_splits: str = "dev", tag: str = "",
                  judge: Optional[JudgeBase] = None) -> Dict:
    recs = _calib_recs(cfg, splits, calib_splits)
    LOG.info("calibration pool: %d videos, %d games (splits: %s, labels: %s)", len(recs),
             len({r.game for r in recs}), calib_splits, cfg.calib_label)
    require(len({r.game for r in recs}) >= 3, "calibration needs >= 3 games")
    if cfg.calib_label == "iou":
        LOG.warning("calib_label=iou ignores descriptions; use it for debugging only")
    wm = prepare_split(recs, cfg)
    ev = gather(cfg, vlm, recs, wm, True)
    by = {r.vid: r for r in recs}
    match = {r.vid: match_hypotheses(cfg, judge, r, wm[r.vid], ev[r.vid]) for r in recs}
    M = sum(len(gt_bug_list(r)) for r in recs)

    # ---- 1. segmentation (tau, lambda) on CORRECT hypotheses, by game-grouped CV --
    correct = [(r.vid, i, j) for r in recs for i, (j, rt) in match[r.vid].items()
               if rt >= cfg.min_correct_rating]
    LOG.info("candidate hypotheses: %d over %d GT bugs; correct (rating >= %d): %d",
             sum(len(ev[r.vid].hyps) for r in recs), M, cfg.min_correct_rating, len(correct))
    taus, lams = parse_grid(cfg.seg_taus), parse_grid(cfg.seg_penalties)
    grid = [(t, l) for t in taus for l in lams]

    def win_iou(item, t, l) -> float:
        vid, i, j = item
        sp, _ = hypothesis_spans(vlm, by[vid], wm[vid], ev[vid].hyps[i], cfg, t, l, "window")
        return _iou_of(sp, gt_bug_list(by[vid])[j]["spans"])

    seg_table, tau, lam, seg_oof = {}, cfg.seg_tau, cfg.seg_penalty, float("nan")
    if correct and cfg.use_segmentation:
        mat = np.array([[win_iou(it, t, l) for (t, l) in grid] for it in correct])
        seg_table = {f"{t:g}|{l:g}": float(v) for (t, l), v in zip(grid, mat.mean(0))}
        k = int(np.argmax(mat.mean(0)))
        tau, lam = grid[k]
        fold = game_folds([by[v].game for v, _, _ in correct], cfg.cv_folds, cfg.seed)
        oof = []
        for f in sorted(set(fold.values())):
            te = [n for n, (v, _, _) in enumerate(correct) if fold[by[v].game] == f]
            tr = [n for n, (v, _, _) in enumerate(correct) if fold[by[v].game] != f]
            kk = int(np.argmax(mat[tr].mean(0))) if tr else k
            oof.extend(mat[te, kk].tolist())
        seg_oof = float(np.mean(oof)) if oof else float("nan")
        LOG.info("segmentation: tau=%g lambda=%g | window IoU on correct hypotheses %.3f "
                 "(game-CV estimate %.3f)", tau, lam, float(mat[:, k].mean()), seg_oof)

    # ---- 2. refine mode: localised spans versus the trivial whole-video span -----
    mode_iou = {}
    for mode in REFINE_MODES:
        vals = []
        for vid, i, j in correct:
            sp, _ = hypothesis_spans(vlm, by[vid], wm[vid], ev[vid].hyps[i], cfg, tau, lam, mode)
            vals.append(_iou_of(sp, gt_bug_list(by[vid])[j]["spans"]))
        mode_iou[mode] = float(np.mean(vals)) if vals else float("nan")
    valid = {m: v for m, v in mode_iou.items() if not math.isnan(v)}
    refine = max(valid, key=lambda m: (valid[m], m == cfg.refine_mode)) if valid else cfg.refine_mode
    LOG.info("refine mode %s selected; mean IoU on correct hypotheses: %s", refine,
             {m: round(v, 3) for m, v in mode_iou.items()})
    if refine == "whole":
        LOG.warning("the WHOLE-VIDEO span beats every localised span on calibration: the "
                    "profile does not localise glitches better than the trivial baseline")

    # ---- 3. utility labels, fractional-logistic model, F1xIoU-optimal threshold --
    rows, u, correct_flag, games = [], [], [], []
    for r in recs:
        vh, gts = ev[r.vid], gt_bug_list(r)
        for k, row in enumerate(hypothesis_rows(vh, cfg, tau, lam)):
            val = 0.0
            if k in match[r.vid] and match[r.vid][k][1] > 0:
                j, rt = match[r.vid][k]
                sp, _ = hypothesis_spans(vlm, r, wm[r.vid], vh.hyps[k], cfg, tau, lam, refine)
                val = rt / 5.0 * _iou_of(sp, gts[j]["spans"])
            rows.append(row)
            u.append(val)
            correct_flag.append(int(k in match[r.vid] and match[r.vid][k][1] >= cfg.min_correct_rating))
            games.append(r.game)
    u = np.asarray(u, float)
    require(len(rows) >= 20, f"only {len(rows)} candidate hypotheses; need >= 20")
    fold = game_folds(games, cfg.cv_folds, cfg.seed)
    oof = np.zeros(len(rows))
    for f in sorted(set(fold.values())):
        te = np.array([fold[g] == f for g in games])
        m = fit_utility([rows[i] for i in np.where(~te)[0]], u[~te], cfg)
        oof[te] = fusion_predict(m, [rows[i] for i in np.where(te)[0]])
    sel = select_f1_threshold(oof, u, M)
    auc = cluster_bootstrap_auc(oof, correct_flag, games, cfg.n_bootstrap, cfg.seed)
    top1 = np.array([rw["rank_inv"] == 1.0 for rw in rows])
    f_all = float(2 * u.sum() / (u.size + M)) if M else 0.0
    f_top1 = float(2 * u[top1].sum() / (int(top1.sum()) + M)) if M else 0.0
    LOG.info("utility model (out-of-fold): AUC for 'correct' %.3f CI [%.3f, %.3f] | F1xIoU "
             "proxy %.4f at t=%.4f with %d emitted (t/F* = %.2f, theory 0.5) | emit-all %.4f | "
             "emit-top1 %.4f", auc["auc"], auc["lo"], auc["hi"], sel["f1_iou"], sel["threshold"],
             sel["n_emitted"], sel["lipton_ratio"], f_all, f_top1)
    reasons = []
    if math.isnan(auc["lo"]) or auc["lo"] <= cfg.min_auc_ci_lo:
        reasons.append(f"utility model cannot rank correct hypotheses: AUC {auc['auc']:.3f} "
                       f"CI [{auc['lo']:.3f}, {auc['hi']:.3f}]")
    if sel["f1_iou"] <= max(f_all, f_top1) + 1e-12:
        LOG.warning("selection does not beat emitting all / top-1 hypotheses on calibration")
    final = fit_utility(rows, u, cfg)
    LOG.info("final utility weights (standardised): %s",
             {f: round(w, 3) for f, w in zip(final["features"], final["w"])})
    out: Dict[str, Any] = {
        "version": VERSION, "fusion": final, "decision_threshold": sel["threshold"],
        "seg_tau": tau, "seg_penalty": lam, "refine_mode": refine,
        "segmentation": {"table": seg_table, "oof_window_iou": seg_oof},
        "refine_mode_iou": mode_iou,
        "selection": {**sel, "f1_iou_emit_all": f_all, "f1_iou_emit_top1": f_top1,
                      "oof_auc_correct": auc},
        "n_hypotheses": len(rows), "n_correct": int(sum(correct_flag)), "n_gt_bugs": M,
        "calib_splits": calib_splits, "calib_label": cfg.calib_label, "n_videos": len(recs),
        "degenerate": bool(reasons), "degeneracy_reasons": reasons,
        "config_switches": {k: getattr(cfg, k) for k in asdict(cfg) if k.startswith("use_")}}
    if reasons:
        path = _calib_path(cfg, (tag + "_" if tag else "") + "degenerate")
        atomic_write_json(path, out)
        msg = "DEGENERATE SELECTOR: " + "; ".join(reasons) + f". Diagnostics: {path}."
        if cfg.fail_on_degenerate:
            raise RuntimeError(msg + " (--allow_degenerate exists for ablation bookkeeping, "
                                     "not for headline numbers)")
        LOG.error(msg)
    atomic_write_json(_calib_path(cfg, tag), out)
    return out


def _calib_path(cfg: Config, tag: str = "") -> str:
    return os.path.join(cfg.paths()["params"], f"calibration{('_' + tag) if tag else ''}.json")


def _calib_recs(cfg: Config, splits: Dict[str, List[Record]], names: str) -> List[Record]:
    names_l = [s.strip() for s in names.split(",") if s.strip()]
    require("test" not in names_l, "calibration may never read the test split")
    recs: List[Record] = []
    for n in names_l:
        recs.extend(splits[n])
    return take_limit(recs, cfg.limit, cfg.stratify_limit)


def apply_calibration(cfg: Config, tag: str = "") -> Config:
    path = _calib_path(cfg, tag)
    require(os.path.exists(path), f"calibration missing at {path}: run 'calibrate' first")
    cal = read_json(path)
    require(str(cal.get("version", "")).startswith("10"),
            f"{path} was not written by CRUX v10; re-run 'calibrate'")
    sw = cal.get("config_switches", {})
    diff = {k: (v, getattr(cfg, k)) for k, v in sw.items() if getattr(cfg, k) != v}
    require(not diff, f"component switches differ from calibration {path}: {diff}")
    return Config(**{**asdict(cfg), "fusion": cal["fusion"],
                     "decision_threshold": float(cal["decision_threshold"]),
                     "seg_tau": float(cal["seg_tau"]), "seg_penalty": float(cal["seg_penalty"]),
                     "refine_mode": str(cal["refine_mode"])})


# =============================================================================
# run / import / export
# =============================================================================

def cmd_run(cfg: Config, vlm: VLMBase, splits: Dict[str, List[Record]], method: str,
            split: str, tag: str = "") -> str:
    recs = take_limit(splits[split], cfg.limit, cfg.stratify_limit)
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
    recs = take_limit(splits[split], cfg.limit, cfg.stratify_limit)
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
    recs = take_limit(splits[split], cfg.limit, cfg.stratify_limit)
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
    "no_hypothesis_scoring": {"use_hypothesis_scoring": False},   # generic profile
    "no_event_resolution": {"use_event_resolution": False},
    "no_segmentation": {"use_segmentation": False},               # member runs
    "no_subframe": {"use_subframe": False},
    "no_utility_selection": {"use_utility_selection": False},     # emit every hypothesis
}


def cmd_ablate(cfg: Config, vlm: VLMBase, judge: JudgeBase, splits: Dict[str, List[Record]],
               split: str, calib_splits: str) -> Dict:
    """Each variant is RE-CALIBRATED (utility model, threshold, segmentation and
    refine mode) on the calibration splits; VLM and judge calls are cached, so
    a variant costs only the calls its own switches add. Degenerate variants are
    reported, not hidden."""
    full = cmd_run(cfg, vlm, splits, "crux", split)
    files = [full]
    for tag, over in ABLATIONS.items():
        vcfg = Config(**{**asdict(cfg), **over, "fail_on_degenerate": False})
        cmd_calibrate(vcfg, vlm, splits, calib_splits, tag, judge)
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


def _mock_oracle(label_of: Dict[str, int], video_bias: Callable[[str], float], signal: bool,
                 salt: str = "rnd", frame_label_of: Optional[Dict[Tuple[str, int], int]] = None):
    """Scripted VLM that reproduces the CRUX v9 probe on VideoGlitchBench:
      * the GENERIC detector is dominated by a per-video offset (sd 1.5) and
        only weakly tied to the window label (+0.3 logits), i.e. high ICC and a
        near-chance within-video AUC;
      * open descriptions name the true glitch on glitch windows and sometimes
        hallucinate an unrelated one on normal windows;
      * the HYPOTHESIS-conditioned presence query is informative for a true
        hypothesis and flat for a hallucinated one.
    With signal=False descriptions and presence are independent of the labels.
    The self-test therefore checks that v10 USES such evidence when it exists
    and REFUSES to calibrate when it does not; it says nothing about how much
    of it a real VLM provides (that is what 'probe' measures)."""
    frame_label_of = frame_label_of or {}

    def win_key(path: str) -> str:
        ap = os.path.abspath(path)
        stem = re.sub(r"_(stitched|pair|keep_\d+_\d+)\.jpg$", "", ap)
        return stem + "_stitched.jpg"

    def rel(path: str) -> str:
        # Noise keys use the last two path components only, so the self-test is
        # reproducible although its data live in a random temporary directory.
        return "/".join(os.path.abspath(path).split(os.sep)[-2:])

    def vid_of(path: str) -> str:
        return os.path.basename(os.path.dirname(os.path.abspath(path)))

    def lab(path: str) -> int:
        m = _KEEP_RE.search(path)
        if m:
            d = os.path.dirname(os.path.abspath(path))
            return int(any(frame_label_of.get((d, f), 0)
                           for f in range(int(m.group(1)), int(m.group(2)) + 1)))
        return int(label_of.get(win_key(path), 0))

    CAR = "The red car floats above the road surface."
    WALL = "A wall texture flickers briefly."

    def oracle(kind, system, user, images, options):
        if kind == "score":
            img = images[0]
            if system == P_DETECT_SYS:
                if img == "__content_free__":
                    lo = 0.6
                else:
                    lo = (0.6 + 1.5 * video_bias(vid_of(img)) + 0.3 * lab(img)
                          + 0.4 * _noise(salt + "d" + rel(img)))
                return {"yes": lo / 2, "no": -lo / 2}
            if system == P_PRESENCE_SYS:
                if img == "__content_free__":
                    lo = -1.0
                elif not signal:
                    lo = -1.0 + 1.0 * _noise(salt + "v" + rel(img) + user[-40:])
                elif "red car" in user:
                    lo = 3.0 * lab(img) - 1.5 + 0.3 * _noise(salt + "v" + rel(img))
                else:
                    lo = -2.0 + 0.3 * _noise(salt + "w" + rel(img))
                return {"yes": lo / 2, "no": -lo / 2}
            raise AssertionError(f"unexpected score prompt: {system[:40]}")
        if system in (P_DESC_OPEN_SYS, P_DESC_SYS):
            img = images[0]
            if signal:
                pos = lab(img)
                hall = _noise(salt + "h" + rel(img)) > 0.9
            else:
                pos = int(_noise(salt + "r" + rel(img)) > 0.3)
                hall = not pos and _noise(salt + "h" + rel(img)) > 0.0
            if system == P_DESC_SYS:        # forced description: always a glitch
                pos = pos or not hall
            if pos:
                return json.dumps({"glitch": True, "description": CAR, "entity": "red car",
                                   "glitch_type": "floating"})
            if hall:
                return json.dumps({"glitch": True, "description": WALL, "entity": "wall",
                                   "glitch_type": "texture"})
            return json.dumps({"glitch": False, "description": "", "entity": "", "glitch_type": ""})
        if system == P_EAR_SYS:
            return json.dumps({"events": []})
        if system == P_VANILLA_SYS:
            return json.dumps({"no_bugs": False, "bugs": [
                {"description": "A car floats in the air.", "time_nodes": [[3, 9]]}]})
        m = re.search(r"^- (.+)$", user, flags=re.M)      # summariser: echo the first input
        return m.group(1) if m else CAR
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

    z = robust_z(np.array([0, 0, 0, 0, 10.0]))
    chk("robust z: outlier stands out, floor on MAD", z[-1] > 5 and abs(z[0]) < 1e-9)

    # ---- S: exact penalised segmentation == brute force -------------------------
    import itertools

    def brute_seg(x, tau, lam):
        T, best = len(x), -np.inf
        for mask in itertools.product((0, 1), repeat=T):
            if not any(mask):
                continue
            segs = member_runs([t for t in range(T) if mask[t]])
            best = max(best, segment_score(x, segs, tau, lam))
        return best
    rng_s = np.random.default_rng(11)
    ok_seg = True
    for trial in range(40):
        T = int(rng_s.integers(1, 8))
        x = rng_s.normal(0, 1.5, T)
        tau, lam = float(rng_s.choice([-0.5, 0.0, 0.7])), float(rng_s.choice([0.0, 0.8, 3.0, 1e9]))
        segs = segment_profile(x, tau, lam)
        ok_seg &= bool(segs) and abs(segment_score(x, segs, tau, lam) - brute_seg(x, tau, lam)) < 1e-9
        ok_seg &= all(b2 >= a2 for a2, b2 in segs) and all(
            segs[k + 1][0] > segs[k][1] + 1 for k in range(len(segs) - 1))
    chk("segmentation DP attains the brute-force optimum (40 random profiles)", ok_seg)
    xk = np.array([1., -3., 2., 2., -0.5, 1., -4., 0.5])
    chk("lambda -> inf gives the single maximum-sum segment (Kadane)",
        segment_profile(xk, 0.0, 1e9) == [(2, 5)], str(segment_profile(xk, 0.0, 1e9)))
    chk("lambda = 0 gives every run above tau",
        segment_profile(xk, 0.0, 0.0) == [(0, 0), (2, 3), (5, 5), (7, 7)], str(segment_profile(xk, 0.0, 0.0)))
    chk("all-negative profile still yields its best window",
        segment_profile([-3., -1., -2.], 0.0, 1.0) == [(1, 1)])

    # ---- U: F1xIoU-optimal selection --------------------------------------------
    rng_u = np.random.default_rng(5)
    uh = rng_u.random(60)
    uu = (rng_u.random(60) < uh) * rng_u.random(60)
    sel = select_f1_threshold(uh, uu, 25)
    brute = max(2 * uu[uh >= t].sum() / ((uh >= t).sum() + 25) for t in uh)
    chk("F1xIoU threshold equals brute-force maximum", abs(sel["f1_iou"] - brute) < 1e-12)
    pc = rng_u.random(200000)
    yc = (rng_u.random(200000) < pc).astype(float)
    selc = select_f1_threshold(pc, yc, int(yc.sum()))
    chk("calibrated scores: optimal threshold = F*/2 (Lipton et al. 2014)",
        abs(selc["lipton_ratio"] - 0.5) < 0.02, f"{selc['lipton_ratio']:.3f}")
    Xf = rng_u.normal(size=(3000, 2))
    uf = expit(Xf @ np.array([1.2, -0.8]) - 0.5)
    wf, bf = fit_logistic(Xf, uf, l2=1e-6)
    chk("fractional logistic recovers E[u|x] for targets in [0, 1]",
        np.allclose(wf, [1.2, -0.8], atol=1e-3) and abs(bf + 0.5) < 1e-3, f"{wf}, {bf}")

    # ---- diagnostics: variance decomposition and within-video AUC ----------------
    vids = [rng_u.normal(0, 1) + rng_u.normal(0, 1, 30) for _ in range(400)]
    ic = icc_oneway(vids)
    chk("ICC(1) recovers 0.5 for equal variance components", abs(ic["icc"] - 0.5) < 0.05, str(ic))
    offs = rng_u.normal(0, 3, 200)
    labs = [(rng_u.random(10) < 0.5).astype(int) for _ in range(200)]
    scs = [o + 1.0 * y_ + rng_u.normal(0, 0.5, 10) for o, y_ in zip(offs, labs)]
    raw_auc = roc_auc(np.concatenate(scs), np.concatenate(labs))
    cen_auc = roc_auc(np.concatenate(within_video_scores(scs)), np.concatenate(labs))
    chk("within-video centring removes video offsets", cen_auc > raw_auc + 0.1,
        f"{raw_auc:.3f} -> {cen_auc:.3f}")
    rr_ = [Record(f"v{i}", f"v{i}", g_, "u", "", False, [], []) for i, g_ in
           enumerate(["A"] * 6 + ["B"] * 6 + ["C"] * 6)]
    chk("stratified --limit covers every game", [r_.game for r_ in take_limit(rr_, 5)]
        == ["A", "B", "C", "A", "B"])

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
                      cv_folds=4)
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
        vbias = lambda v: _noise("video-bias" + v)
        vlm = make_vlm(base, _mock_oracle(label_of, vbias, True, frame_label_of=frame_label_of))
        judge_fn = lambda gt_, pr_: 5 if ("float" in pr_.lower() and "float" in gt_.lower()) else 0
        judge = make_judge(base, judge_fn)

        dg = cmd_diagnose(base, vlm, spl, "dev", hypotheses=True)
        sg_ = dg["signals"]
        chk("diagnose: generic detector is video-dominated (ICC > 0.5), as in the v9 probe",
            sg_["lo_abs"]["variance"]["icc"] > 0.5, str(sg_["lo_abs"]["variance"]))
        chk("diagnose: hypothesis profile localises better within videos than the generic score",
            sg_["hyp_max"]["within_video_auc"]["auc"] > sg_["lo_abs"]["within_video_auc"]["auc"] + 0.1,
            f"{sg_['hyp_max']['within_video_auc']['auc']:.3f} vs "
            f"{sg_['lo_abs']['within_video_auc']['auc']:.3f}")
        chk("diagnose: trivial whole-video IoU reported",
            abs(dg["base"]["whole_video_span_iou_mean"] - 4 / 12) < 0.05,
            str(dg["base"]["whole_video_span_iou_mean"]))

        cal = cmd_calibrate(base, vlm, spl, "dev", "", judge)
        chk("signal oracle: selector not degenerate", not cal["degenerate"], str(cal["degeneracy_reasons"]))
        chk("localised spans beat the whole-video span on calibration",
            cal["refine_mode"] != "whole" and cal["refine_mode_iou"]["whole"] < cal["refine_mode_iou"]["frame"],
            str(cal["refine_mode_iou"]))
        chk("selection beats emitting every hypothesis on calibration",
            cal["selection"]["f1_iou"] > cal["selection"]["f1_iou_emit_all"], str(cal["selection"]))
        f_crux = cmd_run(base, vlm, spl, "crux", "test")
        run_ = read_json(f_crux)
        byname = {r.video_name: r for r in spl["test"]}
        buggy = [o for o in run_["outputs"] if not byname[o["video_name"]].no_bugs]
        chk("frame-level grounding recovers the synthetic [4, 8] s span on every glitch video",
            all(o["time_nodes"] == [[[4.0, 8.0]]] for o in buggy), str([o["time_nodes"] for o in buggy]))
        chk("clean test videos produce no report",
            all(not o["bugs"] for o in run_["outputs"] if byname[o["video_name"]].no_bugs))
        f_van = cmd_run(base, vlm, spl, "vanilla", "test")
        gl = [{"video_name": r.video_name, "game_name": r.game, "no_bugs": False,
               "bugs": ["Something odd happens."], "time_nodes": [[[0, 1]]]} for r in spl["test"][:-1]]
        gpath = os.path.join(root, "glide_batch_report.json")
        atomic_write_json(gpath, gl)
        f_gl = cmd_import_glide(base, spl, "test", gpath)
        res = cmd_evaluate(base, judge, spl, "test", [f_crux, f_van, f_gl])
        chk("evaluation restricted to common videos", res["n_videos"] == len(spl["test"]) - 1)
        chk("crux beats a content-free baseline on the mock",
            res["table"]["crux"]["f1_iou"] > res["table"]["glide_official"]["f1_iou"])
        chk("batch report written in official format",
            os.path.exists(f_crux.replace(".json", "_batch_report.json")))
        vg = Config(**{**asdict(base), "use_hypothesis_scoring": False, "fail_on_degenerate": False})
        cmd_calibrate(vg, vlm, spl, "dev", "no_hypothesis_scoring", judge)
        chk("ablation with the generic profile calibrates and runs",
            os.path.exists(cmd_run(vg, vlm, spl, "crux", "test", "no_hypothesis_scoring")))
        vlm0 = make_vlm(base, _mock_oracle(label_of, vbias, False, frame_label_of=frame_label_of))
        try:
            cmd_calibrate(base, vlm0, spl, "dev", "", judge)
            chk("no-signal oracle is rejected as degenerate", False, "calibrate did not raise")
        except RuntimeError as e:
            chk("no-signal oracle is rejected as degenerate", "DEGENERATE" in str(e), str(e)[:200])
        ab = Config(**{**asdict(base), "use_segmentation": False})
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

COMMANDS = ("selftest", "prepare", "diagnose", "probe", "calibrate", "run", "import_glide",
            "export", "evaluate", "ablate")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CRUX v10 (see module docstring)")
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
    require(cfg.calib_label in ("judge", "iou"), "--calib_label must be 'judge' or 'iou'")
    require(cfg.refine_mode in REFINE_MODES, f"--refine_mode must be one of {REFINE_MODES}")
    LOG.info("CRUX v%s | %s | backbone=%s backend=%s model=%s", VERSION, a.command,
             cfg.backbone, cfg.backend, cfg.model)
    if a.command == "export":
        require(a.run_file, "--run_file is required")
        cmd_export(a.run_file)
        return
    recs = load_records(cfg)
    splits = load_splits(cfg, recs)
    if a.command == "prepare":
        prepare_split(take_limit(splits[a.split], cfg.limit, cfg.stratify_limit), cfg)
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
    if a.command in ("diagnose", "probe"):
        cmd_diagnose(cfg, vlm, splits, a.split, hypotheses=a.command == "probe")
    elif a.command == "calibrate":
        cmd_calibrate(cfg, vlm, splits, a.calib_splits, a.tag,
                      make_judge(cfg) if cfg.calib_label == "judge" else None)
    elif a.command == "run":
        cmd_run(cfg, vlm, splits, a.method, a.split, a.tag)
    elif a.command == "ablate":
        cmd_ablate(cfg, vlm, make_judge(cfg), splits, a.split, a.calib_splits)
    LOG.info("budget: %s", asdict(vlm.budget))


if __name__ == "__main__":
    main()
