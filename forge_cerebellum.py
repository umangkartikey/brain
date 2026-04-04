"""
FORGE Cerebellum — forge_cerebellum.py
========================================
AI analog of the brain's cerebellum.

The cerebellum is the brain's precision engine — a massive
structure containing more neurons than the rest of the brain
combined, dedicated entirely to one thing: making everything
smoother, faster, and more accurate through practice.

It doesn't initiate actions. It doesn't make decisions.
It watches what the rest of the brain does, builds an
internal model of how things work, and then provides
real-time corrections that make execution effortless.

Key insight: The cerebellum is why a skilled pianist doesn't
think about finger placement. It has built a forward model
so accurate that it can predict the sensory consequence of
every action before it happens — and correct errors before
they even occur.

Four core functions:

  1. FORWARD MODEL LEARNING (predictive motor control)
     Every time a module produces an output, the cerebellum
     records: what was the input? what was the output?
     what was the actual result?
     Over thousands of repetitions, it builds a forward model:
     given this input, this action produces this result.
     The model gets more accurate with every cycle.

  2. ERROR CORRECTION (real-time adjustment)
     Before a module sends its output, the cerebellum
     compares the predicted outcome vs the likely actual.
     If they diverge, it sends a correction signal.
     This is the cerebellar "climbing fiber" error signal —
     the most powerful learning signal in the brain.

  3. TIMING (precise temporal coordination)
     The cerebellum is the brain's clock.
     It coordinates the timing of signals across modules —
     ensuring that prefrontal, amygdala, hippocampus all
     fire in the right sequence rather than a chaotic pile-up.
     Poor timing = cognitive dysrhythmia = errors.

  4. SKILL CONSOLIDATION (automaticity)
     Repeated patterns get compressed into automatic routines.
     What once required conscious attention becomes effortless.
     The cerebellum is why expertise feels like intuition —
     it has offloaded the computation to a fast, automatic layer.

Architecture:
  ForwardModelLibrary  → stores learned input→output mappings
  ErrorCorrectionEngine→ computes and applies corrections
  TimingCoordinator    → manages inter-module timing
  SkillConsolidator    → compresses repeated patterns
  CerebellarPredictor  → generates pre-movement predictions
  CerebellarOutput     → smooth signal to downstream modules
"""

import json
import time
import uuid
import sqlite3
import threading
import math
from datetime import datetime
from collections import deque, defaultdict
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.columns import Columns
    from rich.rule import Rule
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# ─── Constants ────────────────────────────────────────────────────────────────

DB_PATH  = "forge_cerebellum.db"
API_PORT = 7800
VERSION  = "1.0.0"

# Learning rates
FORWARD_MODEL_LR    = 0.12   # how fast forward models update
ERROR_SIGNAL_DECAY  = 0.15   # climbing fiber signal decay per cycle
SKILL_THRESHOLD     = 8      # repetitions before skill consolidation
AUTOMATICITY_MAX    = 0.95   # asymptote for automaticity score

# Timing
TIMING_WINDOW       = 12     # cycles for timing analysis
TIMING_JITTER_MAX   = 0.30   # jitter above this = dysrhythmia

# Correction
MAX_CORRECTION      = 0.35   # max adjustment cerebellum can apply
CORRECTION_DEADBAND = 0.05   # below this — no correction applied

# Prediction confidence
MIN_SAMPLES_TO_PREDICT = 3   # need this many samples before predicting

console = Console() if HAS_RICH else None

# ─── Enums ────────────────────────────────────────────────────────────────────

class SkillLevel(Enum):
    NAIVE        = "NAIVE"        # first encounter
    LEARNING     = "LEARNING"     # actively building model
    COMPETENT    = "COMPETENT"    # model reliable
    PROFICIENT   = "PROFICIENT"   # smooth, fast, low-error
    AUTOMATIC    = "AUTOMATIC"    # fully consolidated — effortless

class TimingState(Enum):
    SYNCHRONOUS  = "SYNCHRONOUS"  # modules firing in good sequence
    DRIFTING     = "DRIFTING"     # slight timing irregularity
    DYSRHYTHMIC  = "DYSRHYTHMIC"  # significant timing problems
    DESYNCHRONIZED="DESYNCHRONIZED" # modules out of phase

class CorrectionType(Enum):
    NONE         = "NONE"
    MINOR        = "MINOR"        # < 10% adjustment
    MODERATE     = "MODERATE"     # 10-25% adjustment
    SIGNIFICANT  = "SIGNIFICANT"  # > 25% adjustment
    OVERRIDE     = "OVERRIDE"     # cerebellum strongly disagrees

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class ForwardModel:
    """A learned mapping: context → expected output."""
    id:            str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_key:   str   = ""      # hashed context fingerprint
    module:        str   = ""      # which module this models
    samples:       int   = 0       # training examples seen
    predicted_val: float = 0.0     # current prediction
    actual_mean:   float = 0.0     # running mean of actual outputs
    error_mean:    float = 0.0     # running mean of prediction error
    accuracy:      float = 0.0     # 1 - normalized_error
    skill_level:   str   = SkillLevel.NAIVE.value
    automaticity:  float = 0.0     # 0-1 automaticity score
    last_updated:  str   = field(default_factory=lambda: datetime.now().isoformat())
    created:       str   = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class CorrectionEvent:
    """A cerebellar correction applied to a module output."""
    id:             str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:      str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:          int   = 0
    module:         str   = ""
    raw_value:      float = 0.0
    corrected_value:float = 0.0
    correction_delta:float= 0.0
    correction_type:str   = CorrectionType.NONE.value
    pattern_key:    str   = ""
    model_accuracy: float = 0.0

@dataclass
class TimingEvent:
    """A timing coordination event."""
    id:          str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:   str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:       int   = 0
    state:       str   = TimingState.SYNCHRONOUS.value
    jitter:      float = 0.0
    modules_late: list = field(default_factory=list)
    correction_applied: bool = False

# ─── Database ─────────────────────────────────────────────────────────────────

class CerebellumDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS forward_models (
                    id TEXT PRIMARY KEY, pattern_key TEXT, module TEXT,
                    samples INTEGER, predicted_val REAL, actual_mean REAL,
                    error_mean REAL, accuracy REAL, skill_level TEXT,
                    automaticity REAL, last_updated TEXT, created TEXT
                );
                CREATE TABLE IF NOT EXISTS corrections (
                    id TEXT PRIMARY KEY, timestamp TEXT, cycle INTEGER,
                    module TEXT, raw_value REAL, corrected_value REAL,
                    correction_delta REAL, correction_type TEXT,
                    pattern_key TEXT, model_accuracy REAL
                );
                CREATE TABLE IF NOT EXISTS timing_events (
                    id TEXT PRIMARY KEY, timestamp TEXT, cycle INTEGER,
                    state TEXT, jitter REAL, modules_late TEXT,
                    correction_applied INTEGER
                );
                CREATE TABLE IF NOT EXISTS skill_consolidations (
                    id TEXT PRIMARY KEY, timestamp TEXT, cycle INTEGER,
                    module TEXT, pattern_key TEXT, from_level TEXT,
                    to_level TEXT, automaticity REAL, samples INTEGER
                );
            """)
            self.conn.commit()

    def save_model(self, m: ForwardModel):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO forward_models VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (m.id, m.pattern_key, m.module, m.samples,
                  m.predicted_val, m.actual_mean, m.error_mean,
                  m.accuracy, m.skill_level, m.automaticity,
                  m.last_updated, m.created))
            self.conn.commit()

    def save_correction(self, c: CorrectionEvent):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO corrections VALUES
                (?,?,?,?,?,?,?,?,?,?)
            """, (c.id, c.timestamp, c.cycle, c.module,
                  c.raw_value, c.corrected_value, c.correction_delta,
                  c.correction_type, c.pattern_key, c.model_accuracy))
            self.conn.commit()

    def save_timing(self, t: TimingEvent):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO timing_events VALUES
                (?,?,?,?,?,?,?)
            """, (t.id, t.timestamp, t.cycle, t.state,
                  t.jitter, json.dumps(t.modules_late),
                  int(t.correction_applied)))
            self.conn.commit()

    def log_consolidation(self, cycle: int, module: str, key: str,
                          from_lvl: str, to_lvl: str,
                          automaticity: float, samples: int):
        with self.lock:
            self.conn.execute("""
                INSERT INTO skill_consolidations VALUES (?,?,?,?,?,?,?,?,?)
            """, (str(uuid.uuid4())[:8], datetime.now().isoformat(),
                  cycle, module, key, from_lvl, to_lvl, automaticity, samples))
            self.conn.commit()

    def get_models(self, min_samples=1):
        with self.lock:
            return self.conn.execute("""
                SELECT module, pattern_key, samples, accuracy,
                       skill_level, automaticity, error_mean
                FROM forward_models WHERE samples >= ?
                ORDER BY automaticity DESC
            """, (min_samples,)).fetchall()

    def get_recent_corrections(self, limit=15):
        with self.lock:
            return self.conn.execute("""
                SELECT cycle, module, raw_value, corrected_value,
                       correction_delta, correction_type, model_accuracy
                FROM corrections ORDER BY cycle DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_timing_history(self, limit=10):
        with self.lock:
            return self.conn.execute("""
                SELECT cycle, state, jitter, correction_applied
                FROM timing_events ORDER BY cycle DESC LIMIT ?
            """, (limit,)).fetchall()


# ─── Forward Model Library ────────────────────────────────────────────────────

class ForwardModelLibrary:
    """
    Stores and updates learned forward models.

    A forward model maps: context → predicted output
    The cerebellum maintains one model per module per context type.

    Learning algorithm: running weighted average.
    New samples update the model with weight FORWARD_MODEL_LR.
    Old knowledge decays gently — the model stays current.

    Key property: the model's prediction improves monotonically
    with experience, but never reaches perfect accuracy.
    There is always residual uncertainty — which is healthy.
    """

    def __init__(self, db: CerebellumDB):
        self.db     = db
        self.models: dict[str, ForwardModel] = {}

    def _key(self, module: str, pattern: str) -> str:
        return f"{module}::{pattern}"

    def update(self, module: str, pattern: str,
               actual: float, cycle: int) -> ForwardModel:
        """Update forward model with new observation."""
        key = self._key(module, pattern)

        if key not in self.models:
            model = ForwardModel(
                pattern_key  = pattern,
                module       = module,
                samples      = 0,
                predicted_val= actual,
                actual_mean  = actual,
                error_mean   = 0.0,
                accuracy     = 0.5,
            )
            self.models[key] = model
        else:
            model = self.models[key]

        prev_predicted = model.predicted_val
        error          = abs(actual - prev_predicted)

        # Running weighted average update
        α = FORWARD_MODEL_LR
        model.actual_mean  = round(model.actual_mean  * (1-α) + actual * α, 5)
        model.predicted_val= round(model.predicted_val* (1-α) + actual * α, 5)
        model.error_mean   = round(model.error_mean   * (1-α) + error  * α, 5)
        model.samples     += 1

        # Accuracy: inverse of normalized error
        model.accuracy = round(
            max(0.0, 1.0 - min(1.0, model.error_mean * 3.0)), 4
        )

        # Automaticity grows with samples and accuracy
        target_auto = min(AUTOMATICITY_MAX,
                         (model.samples / (model.samples + 15.0))
                         * model.accuracy)
        model.automaticity = round(
            model.automaticity * 0.85 + target_auto * 0.15, 4
        )

        # Skill level
        prev_level    = model.skill_level
        model.skill_level = self._skill_level(model.samples, model.automaticity).value
        model.last_updated= datetime.now().isoformat()

        self.db.save_model(model)

        # Log consolidation events
        if prev_level != model.skill_level:
            self.db.log_consolidation(
                cycle, module, pattern,
                prev_level, model.skill_level,
                model.automaticity, model.samples
            )

        return model

    def predict(self, module: str, pattern: str) -> Optional[tuple[float, float]]:
        """
        Returns (predicted_value, confidence) or None if insufficient data.
        """
        key = self._key(module, pattern)
        if key not in self.models:
            return None
        m = self.models[key]
        if m.samples < MIN_SAMPLES_TO_PREDICT:
            return None
        return m.predicted_val, m.accuracy

    def _skill_level(self, samples: int, automaticity: float) -> SkillLevel:
        if automaticity >= 0.85:              return SkillLevel.AUTOMATIC
        if automaticity >= 0.65:              return SkillLevel.PROFICIENT
        if automaticity >= 0.40:              return SkillLevel.COMPETENT
        if samples >= 2:                      return SkillLevel.LEARNING
        return SkillLevel.NAIVE

    def best_models(self, n=5) -> list[ForwardModel]:
        return sorted(
            self.models.values(),
            key=lambda m: m.automaticity, reverse=True
        )[:n]

    def total_automatic(self) -> int:
        return sum(1 for m in self.models.values()
                   if m.skill_level == SkillLevel.AUTOMATIC.value)


# ─── Error Correction Engine ──────────────────────────────────────────────────

class ErrorCorrectionEngine:
    """
    Computes and applies real-time corrections to module outputs.

    This models the cerebellar climbing fiber system:
    the inferior olive sends error signals to Purkinje cells,
    which then adjust their output to minimize future errors.

    The correction has two components:
      PREDICTIVE  — based on what the forward model says should happen
      REACTIVE    — based on what actually happened vs what was predicted

    Corrections are scaled by model accuracy:
    a low-accuracy model applies small, tentative corrections.
    A high-accuracy model applies confident, large corrections.
    """

    def __init__(self):
        self.climbing_fiber_signal = 0.0   # current error signal strength
        self.total_corrections     = 0
        self.total_magnitude       = 0.0

    def correct(self, module: str, raw_value: float,
                model: Optional[ForwardModel],
                cycle: int) -> tuple[float, CorrectionEvent]:
        """
        Apply cerebellar correction to raw module output.
        Returns (corrected_value, correction_event).
        """
        if model is None or model.samples < MIN_SAMPLES_TO_PREDICT:
            # No model yet — pass through unchanged
            event = CorrectionEvent(
                cycle           = cycle,
                module          = module,
                raw_value       = raw_value,
                corrected_value = raw_value,
                correction_delta= 0.0,
                correction_type = CorrectionType.NONE.value,
                pattern_key     = "",
                model_accuracy  = 0.0
            )
            return raw_value, event

        # Predictive correction
        prediction = model.predicted_val
        raw_error  = raw_value - prediction

        # Scale correction by model confidence and cap at MAX_CORRECTION
        correction_magnitude = abs(raw_error) * model.accuracy
        correction_magnitude = min(MAX_CORRECTION, correction_magnitude)

        # Don't correct tiny errors (deadband)
        if correction_magnitude < CORRECTION_DEADBAND:
            corrected = raw_value
            delta     = 0.0
            ctype     = CorrectionType.NONE
        else:
            # Pull raw value toward predicted
            direction = -1 if raw_value > prediction else 1
            delta     = round(direction * correction_magnitude, 5)
            corrected = round(max(0.0, min(1.0, raw_value + delta)), 5)

            if abs(delta) < 0.10:  ctype = CorrectionType.MINOR
            elif abs(delta) < 0.25:ctype = CorrectionType.MODERATE
            elif abs(delta) < MAX_CORRECTION: ctype = CorrectionType.SIGNIFICANT
            else:                  ctype = CorrectionType.OVERRIDE

        # Update climbing fiber signal
        self.climbing_fiber_signal = round(
            min(1.0, abs(raw_error) * (1.0 + model.accuracy)), 4
        )

        if ctype != CorrectionType.NONE:
            self.total_corrections  += 1
            self.total_magnitude    += abs(delta)

        event = CorrectionEvent(
            cycle            = cycle,
            module           = module,
            raw_value        = raw_value,
            corrected_value  = corrected,
            correction_delta = delta,
            correction_type  = ctype.value,
            pattern_key      = model.pattern_key,
            model_accuracy   = model.accuracy
        )
        return corrected, event

    def decay(self):
        """Climbing fiber signal decays each cycle."""
        self.climbing_fiber_signal = round(
            max(0.0, self.climbing_fiber_signal - ERROR_SIGNAL_DECAY), 4
        )

    def average_correction(self) -> float:
        if self.total_corrections == 0: return 0.0
        return round(self.total_magnitude / self.total_corrections, 4)


# ─── Timing Coordinator ───────────────────────────────────────────────────────

class TimingCoordinator:
    """
    Manages the timing of signals across modules.

    The cerebellum's timing function:
    It acts as a precise internal clock that coordinates
    when different brain regions fire relative to each other.

    Mistimed signals — even if each is individually correct —
    produce degraded output. The cerebellum ensures that
    amygdala fear, prefrontal reasoning, and hippocampal
    memory retrieval arrive in the right order.

    In FORGE: tracks the processing latency of each module
    and detects when modules are falling behind the rhythm.
    """

    def __init__(self, db: CerebellumDB, window=TIMING_WINDOW):
        self.db       = db
        self.window   = window
        self.module_times: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self.cycle_times: deque = deque(maxlen=window)
        self.jitter_history: deque = deque(maxlen=window)

    def record(self, module_latencies: dict[str, float], cycle: int) -> TimingEvent:
        """
        Record module latencies and assess timing state.
        module_latencies: {module_name: latency_ms}
        """
        for module, latency in module_latencies.items():
            self.module_times[module].append(latency)

        # Compute jitter — coefficient of variation across modules
        if module_latencies:
            lats  = list(module_latencies.values())
            mean  = sum(lats) / len(lats)
            if mean > 0:
                variance = sum((l - mean)**2 for l in lats) / len(lats)
                jitter   = round(math.sqrt(variance) / mean, 4)
            else:
                jitter = 0.0
        else:
            jitter = 0.0

        self.jitter_history.append(jitter)

        # Find late modules (> 2 standard deviations above mean)
        modules_late = []
        if len(module_latencies) >= 3:
            lats = list(module_latencies.values())
            mean = sum(lats) / len(lats)
            std  = math.sqrt(sum((l-mean)**2 for l in lats) / len(lats))
            modules_late = [
                m for m, l in module_latencies.items()
                if l > mean + 2 * std
            ]

        # Determine timing state
        state = self._assess_state(jitter, modules_late)

        event = TimingEvent(
            cycle              = cycle,
            state              = state.value,
            jitter             = jitter,
            modules_late       = modules_late,
            correction_applied = len(modules_late) > 0
        )
        self.db.save_timing(event)
        return event

    def _assess_state(self, jitter: float,
                      modules_late: list) -> TimingState:
        if len(modules_late) >= 3:    return TimingState.DESYNCHRONIZED
        if jitter > TIMING_JITTER_MAX:return TimingState.DYSRHYTHMIC
        if jitter > TIMING_JITTER_MAX * 0.5: return TimingState.DRIFTING
        return TimingState.SYNCHRONOUS

    def smoothness_score(self) -> float:
        """Overall timing smoothness (1.0 = perfect)."""
        if not self.jitter_history: return 1.0
        avg_jitter = sum(self.jitter_history) / len(self.jitter_history)
        return round(max(0.0, 1.0 - min(1.0, avg_jitter / TIMING_JITTER_MAX)), 4)


# ─── Skill Consolidator ───────────────────────────────────────────────────────

class SkillConsolidator:
    """
    Compresses repeated patterns into automatic routines.

    Biological analog: long-term depression (LTD) at parallel
    fiber → Purkinje cell synapses. Repeated activation of a
    pattern drives synaptic weakening in a precise way that
    encodes the skill into the cerebellar microcircuit.

    The result: what once required active attention becomes
    automatic. The cortex is freed to think about something else.

    In FORGE: tracks pattern repetition counts.
    Once a pattern crosses SKILL_THRESHOLD:
    - It gets a dedicated "routine" slot
    - Future processing is faster (simulated)
    - Error rate on that pattern drops further
    """

    def __init__(self):
        self.pattern_counts: dict[str, int] = defaultdict(int)
        self.routines: dict[str, dict]      = {}  # consolidated skills
        self.total_consolidated             = 0

    def observe(self, pattern: str, module: str,
                model: Optional[ForwardModel]) -> Optional[dict]:
        """
        Observe a pattern occurrence. Returns consolidation event
        if threshold reached.
        """
        self.pattern_counts[pattern] += 1
        count = self.pattern_counts[pattern]

        if count >= SKILL_THRESHOLD and pattern not in self.routines:
            # Consolidate into routine
            routine = {
                "pattern":       pattern,
                "module":        module,
                "consolidated_at": count,
                "speed_bonus":   round(min(0.40, (count - SKILL_THRESHOLD) * 0.02 + 0.10), 4),
                "error_reduction": round(min(0.50, count * 0.03), 4),
                "automaticity":  model.automaticity if model else 0.5,
                "created":       datetime.now().isoformat()
            }
            self.routines[pattern] = routine
            self.total_consolidated += 1
            return routine

        return None

    def get_routine(self, pattern: str) -> Optional[dict]:
        return self.routines.get(pattern)

    def speed_bonus(self, pattern: str) -> float:
        r = self.routines.get(pattern)
        return r["speed_bonus"] if r else 0.0

    def error_reduction(self, pattern: str) -> float:
        r = self.routines.get(pattern)
        return r["error_reduction"] if r else 0.0

    def top_routines(self, n=5) -> list[dict]:
        return sorted(
            self.routines.values(),
            key=lambda r: r["automaticity"], reverse=True
        )[:n]


# ─── Cerebellar Predictor ─────────────────────────────────────────────────────

class CerebellarPredictor:
    """
    Generates pre-movement predictions.

    The cerebellum's most remarkable property:
    it predicts the sensory consequences of an action
    BEFORE the action occurs.

    This allows it to generate a "corollary discharge" —
    a copy of the expected sensation — which is subtracted
    from the actual sensation. If they match: ignore.
    If they don't match: something unexpected happened.

    In FORGE: before a module generates output, the cerebellum
    generates a prediction of what that output will be.
    The difference between prediction and actual becomes
    the primary learning signal.
    """

    def __init__(self, library: ForwardModelLibrary):
        self.library       = library
        self.predictions:  dict[str, tuple[float, float]] = {}
        self.total_hits    = 0
        self.total_misses  = 0
        self.total_predictions = 0

    def predict_ahead(self, module: str,
                      pattern: str) -> Optional[tuple[float, float]]:
        """
        Generate a prediction before the module acts.
        Returns (predicted_value, confidence) or None.
        """
        result = self.library.predict(module, pattern)
        if result:
            self.predictions[f"{module}::{pattern}"] = result
            self.total_predictions += 1
        return result

    def evaluate(self, module: str, pattern: str,
                 actual: float) -> tuple[float, bool]:
        """
        Compare prediction to actual.
        Returns (prediction_error, was_predicted).
        """
        key = f"{module}::{pattern}"
        pred = self.predictions.pop(key, None)
        if pred is None:
            return 0.0, False

        predicted_val, _ = pred
        error = abs(actual - predicted_val)
        threshold = 0.15

        if error < threshold:
            self.total_hits += 1
        else:
            self.total_misses += 1

        return round(error, 4), True

    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        if total == 0: return 0.0
        return round(self.total_hits / total, 4)


# ─── FORGE Cerebellum ─────────────────────────────────────────────────────────

class ForgeCerebellum:
    def __init__(self):
        self.db          = CerebellumDB()
        self.library     = ForwardModelLibrary(self.db)
        self.corrector   = ErrorCorrectionEngine()
        self.timing      = TimingCoordinator(self.db)
        self.consolidator= SkillConsolidator()
        self.predictor   = CerebellarPredictor(self.library)
        self.cycle       = 0

        self.total_corrections    = 0
        self.total_consolidations = 0
        self.total_dysrhythmias   = 0

    def process(self, signal: dict,
                module_outputs: Optional[dict] = None) -> dict:
        """
        Full cerebellum processing pipeline.

        signal:         raw input (for context / timing data)
        module_outputs: dict of {module: {"value": float, "latency_ms": float, ...}}

        Returns cerebellar corrections, timing assessment,
        skill levels, and smoothed outputs.
        """
        t0             = time.time()
        self.cycle    += 1
        module_outputs = module_outputs or {}

        # Extract context pattern for this signal
        pattern = self._extract_pattern(signal)

        # 1. Decay climbing fiber signal
        self.corrector.decay()

        # 2. Process each module — update models and apply corrections
        corrections    = {}
        skill_updates  = []
        new_routines   = []
        module_latencies = {}

        for module, mout in module_outputs.items():
            raw_val  = mout.get("value", 0.5)
            latency  = mout.get("latency_ms", 20.0)
            module_latencies[module] = latency

            # Generate prediction before seeing actual
            self.predictor.predict_ahead(module, pattern)

            # Update forward model with actual value
            model = self.library.update(module, pattern, raw_val, self.cycle)

            # Evaluate prediction accuracy
            pred_error, was_predicted = self.predictor.evaluate(
                module, pattern, raw_val
            )

            # Apply correction
            corrected, corr_event = self.corrector.correct(
                module, raw_val, model, self.cycle
            )

            if corr_event.correction_type != CorrectionType.NONE.value:
                self.db.save_correction(corr_event)
                self.total_corrections += 1

            # Check skill consolidation
            routine = self.consolidator.observe(pattern, module, model)
            if routine:
                new_routines.append(routine)
                self.total_consolidations += 1

            corrections[module] = {
                "raw":          raw_val,
                "corrected":    corrected,
                "delta":        corr_event.correction_delta,
                "type":         corr_event.correction_type,
                "skill_level":  model.skill_level,
                "automaticity": model.automaticity,
                "accuracy":     model.accuracy,
                "samples":      model.samples,
                "pred_error":   pred_error,
                "speed_bonus":  self.consolidator.speed_bonus(pattern),
            }
            skill_updates.append({
                "module":  module,
                "skill":   model.skill_level,
                "auto":    model.automaticity,
                "samples": model.samples,
            })

        # 3. Timing coordination
        timing_event = self.timing.record(module_latencies, self.cycle)
        if timing_event.state in [
            TimingState.DYSRHYTHMIC.value,
            TimingState.DESYNCHRONIZED.value
        ]:
            self.total_dysrhythmias += 1

        # 4. Overall smoothness
        smoothness = self._compute_smoothness(corrections)

        elapsed = (time.time() - t0) * 1000

        return {
            "cycle":               self.cycle,
            "pattern":             pattern,
            "corrections":         corrections,
            "timing_state":        timing_event.state,
            "timing_jitter":       timing_event.jitter,
            "modules_late":        timing_event.modules_late,
            "smoothness":          smoothness,
            "climbing_fiber":      self.corrector.climbing_fiber_signal,
            "prediction_hit_rate": self.predictor.hit_rate(),
            "new_routines":        new_routines,
            "total_models":        len(self.library.models),
            "total_automatic":     self.library.total_automatic(),
            "total_corrections":   self.total_corrections,
            "total_consolidations":self.total_consolidations,
            "total_dysrhythmias":  self.total_dysrhythmias,
            "processing_ms":       round(elapsed, 2),
        }

    def _extract_pattern(self, signal: dict) -> str:
        """
        Extract a compact context key for the forward model.
        Groups similar contexts together for faster generalization.
        """
        threat  = signal.get("threat", 0)
        anomaly = int(signal.get("anomaly", False))
        social  = signal.get("social", {}) or {}
        intent  = social.get("inferred_intent", "neutral")[:8]
        # Bin continuous values for generalization
        t_bin   = threat
        return f"T{t_bin}A{anomaly}I{intent}"

    def _compute_smoothness(self, corrections: dict) -> float:
        """
        Overall smoothness score across all module corrections.
        1.0 = no corrections needed (fully automatic).
        """
        if not corrections: return 0.5
        auto_scores = [c["automaticity"] for c in corrections.values()]
        return round(sum(auto_scores) / len(auto_scores), 4)

    def get_status(self) -> dict:
        best = self.library.best_models(5)
        return {
            "version":             VERSION,
            "cycle":               self.cycle,
            "total_models":        len(self.library.models),
            "total_automatic":     self.library.total_automatic(),
            "total_corrections":   self.total_corrections,
            "total_consolidations":self.total_consolidations,
            "total_dysrhythmias":  self.total_dysrhythmias,
            "prediction_hit_rate": self.predictor.hit_rate(),
            "smoothness":          self.timing.smoothness_score(),
            "climbing_fiber":      self.corrector.climbing_fiber_signal,
            "avg_correction":      self.corrector.average_correction(),
            "top_skills":          [
                {
                    "module":      m.module,
                    "pattern":     m.pattern_key,
                    "skill":       m.skill_level,
                    "automaticity":m.automaticity,
                    "samples":     m.samples,
                    "accuracy":    m.accuracy,
                }
                for m in best
            ],
            "top_routines": self.consolidator.top_routines(3),
        }


# ─── Rich UI ──────────────────────────────────────────────────────────────────

SKILL_COLORS = {
    "NAIVE":      "dim",
    "LEARNING":   "blue",
    "COMPETENT":  "cyan",
    "PROFICIENT": "green",
    "AUTOMATIC":  "bright_green",
}

TIMING_COLORS = {
    "SYNCHRONOUS":    "green",
    "DRIFTING":       "yellow",
    "DYSRHYTHMIC":    "red",
    "DESYNCHRONIZED": "bright_red",
}

CORRECTION_COLORS = {
    "NONE":        "dim",
    "MINOR":       "blue",
    "MODERATE":    "yellow",
    "SIGNIFICANT": "red",
    "OVERRIDE":    "bright_red",
}

def render_cerebellum(result: dict, idx: int):
    if not HAS_RICH: return

    smoothness = result["smoothness"]
    sc         = "bright_green" if smoothness > 0.7 else "green" if smoothness > 0.4 else "yellow"
    timing     = result["timing_state"]
    tc         = TIMING_COLORS.get(timing, "white")
    hit_rate   = result["prediction_hit_rate"]
    cf         = result["climbing_fiber"]

    console.print(Rule(
        f"[bold cyan]⬡ CEREBELLUM[/bold cyan]  "
        f"[dim]#{idx}[/dim]  "
        f"[{sc}]smooth={smoothness:.3f}[/{sc}]  "
        f"[{tc}]{timing}[/{tc}]  "
        f"[dim]hit={hit_rate:.2f}  cf={cf:.3f}[/dim]"
    ))

    if timing in ["DYSRHYTHMIC", "DESYNCHRONIZED"]:
        console.print(Panel(
            f"[bold red]⚡ TIMING ALARM — {timing}[/bold red]\n"
            f"Jitter: {result['timing_jitter']:.3f}  "
            f"Late modules: {', '.join(result['modules_late']) or 'none'}\n"
            f"[dim]Timing correction applied.[/dim]",
            border_style="red"
        ))

    if result["new_routines"]:
        for r in result["new_routines"]:
            console.print(Panel(
                f"[bold bright_green]★ SKILL CONSOLIDATED — {r['module']}[/bold bright_green]\n"
                f"Pattern: {r['pattern']}  "
                f"Speed bonus: +{r['speed_bonus']:.2f}  "
                f"Error reduction: -{r['error_reduction']:.2f}",
                border_style="bright_green"
            ))

    def bar(v, w=10):
        return "█" * int(v * w) + "░" * (w - int(v * w))

    # Module corrections table
    corr_lines = []
    for mod, c in result["corrections"].items():
        skill = c["skill_level"]
        skc   = SKILL_COLORS.get(skill, "white")
        ct    = c["type"]
        ctc   = CORRECTION_COLORS.get(ct, "dim")
        auto  = c["automaticity"]
        delta = c["delta"]
        sign  = "+" if delta >= 0 else ""
        corr_lines.append(
            f"  [{skc}]{mod:<14}[/{skc}]  "
            f"[{ctc}]{ct:<11}[/{ctc}]  "
            f"[dim]{c['raw']:.3f}→{c['corrected']:.3f}[/dim]  "
            f"[dim]({sign}{delta:.3f})[/dim]  "
            f"[{skc}]{skill:<10}[/{skc}]  "
            f"[dim]auto={auto:.3f} acc={c['accuracy']:.3f}[/dim]"
        )

    left_lines = [
        f"[bold]Smoothness:[/bold]  [{sc}]{bar(smoothness)} {smoothness:.3f}[/{sc}]",
        f"[bold]Hit rate:  [/bold]  {bar(hit_rate)} {hit_rate:.3f}",
        f"[bold]Climb.fiber:[/bold] [magenta]{bar(cf)} {cf:.3f}[/magenta]",
        f"[bold]Timing:     [/bold] [{tc}]{timing}[/{tc}]  jitter={result['timing_jitter']:.3f}",
        f"",
        f"[bold]Total models:[/bold]     {result['total_models']}",
        f"[bold]Automatic:[/bold]        {result['total_automatic']}",
        f"[bold]Corrections:[/bold]      {result['total_corrections']}",
        f"[bold]Consolidations:[/bold]   {result['total_consolidations']}",
        f"[bold]Pattern:[/bold]          [dim]{result['pattern']}[/dim]",
    ]

    right_lines = ["[bold]Module Corrections:[/bold]"] + corr_lines \
        if corr_lines else ["[dim]No modules this cycle[/dim]"]

    console.print(Columns([
        Panel("\n".join(left_lines),  title="[bold cyan]Cerebellar State[/bold cyan]", border_style="cyan"),
        Panel("\n".join(right_lines), title="[bold]Module Skill + Corrections[/bold]", border_style="dim")
    ]))


def run_demo():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FORGE CEREBELLUM[/bold cyan]\n"
            "[dim]Forward Models · Error Correction · Timing · Skill Consolidation[/dim]\n"
            f"[dim]Version {VERSION}  |  Port {API_PORT}[/dim]",
            border_style="cyan"
        ))

    cb = ForgeCerebellum()

    # A learning arc: same context repeated many times,
    # watching the system go from NAIVE → AUTOMATIC
    base_mods = {
        "amygdala":    {"value": 0.20, "latency_ms": 8.0},
        "prefrontal":  {"value": 0.75, "latency_ms": 42.0},
        "hippocampus": {"value": 0.60, "latency_ms": 35.0},
        "thalamus":    {"value": 0.50, "latency_ms": 12.0},
    }

    scenarios = [
        # First encounters — NAIVE models, no corrections
        ({"threat":0,"anomaly":False,"social":{"inferred_intent":"cooperative"}},
         base_mods, "First encounter — building forward models"),

        ({"threat":0,"anomaly":False,"social":{"inferred_intent":"cooperative"}},
         {k:{**v,"value":v["value"]+0.03} for k,v in base_mods.items()},
         "Second encounter — models updating"),

        ({"threat":0,"anomaly":False,"social":{"inferred_intent":"cooperative"}},
         {k:{**v,"value":v["value"]+0.01} for k,v in base_mods.items()},
         "Third encounter — corrections begin"),

        # Repeated practice — models get accurate
        *[({"threat":0,"anomaly":False,"social":{"inferred_intent":"cooperative"}},
           {k:{**v,"value":v["value"]+ (i-4)*0.01} for k,v in base_mods.items()},
           f"Practice cycle {i-2} — automaticity growing")
          for i in range(4, 10)],

        # Timing disruption
        ({"threat":1,"anomaly":True,"social":{"inferred_intent":"ambiguous"}},
         {"amygdala":   {"value":0.65,"latency_ms":9.0},
          "prefrontal": {"value":0.55,"latency_ms":120.0},  # late!
          "hippocampus":{"value":0.50,"latency_ms":38.0},
          "thalamus":   {"value":0.45,"latency_ms":14.0}},
         "Timing disruption — prefrontal late"),

        # Novel threatening pattern — back to learning
        ({"threat":3,"anomaly":True,"social":{"inferred_intent":"intrusion"}},
         {"amygdala":   {"value":0.85,"latency_ms":8.0},
          "prefrontal": {"value":0.30,"latency_ms":55.0},
          "salience":   {"value":0.80,"latency_ms":18.0},
          "thalamus":   {"value":0.70,"latency_ms":11.0}},
         "Novel threat pattern — new models needed"),

        # Recovery to familiar pattern
        ({"threat":0,"anomaly":False,"social":{"inferred_intent":"cooperative"}},
         base_mods,
         "Back to familiar — automaticity kicks in"),
    ]

    for i, (sig, mods, label) in enumerate(scenarios):
        if HAS_RICH:
            console.print(f"\n[bold dim]━━━ {i+1}: {label.upper()} ━━━[/bold dim]")
        result = cb.process(sig, mods)
        render_cerebellum(result, i+1)
        time.sleep(0.05)

    # Final status
    if HAS_RICH:
        console.print(Rule("[bold cyan]⬡ CEREBELLUM FINAL STATUS[/bold cyan]"))
        status = cb.get_status()

        st = Table(box=box.DOUBLE_EDGE, border_style="cyan", title="Cerebellum Status")
        st.add_column("Metric",  style="cyan")
        st.add_column("Value",   style="white")
        st.add_row("Cycles",            str(status["cycle"]))
        st.add_row("Forward Models",    str(status["total_models"]))
        st.add_row("Automatic Skills",  str(status["total_automatic"]))
        st.add_row("Total Corrections", str(status["total_corrections"]))
        st.add_row("Consolidations",    str(status["total_consolidations"]))
        st.add_row("Dysrhythmias",      str(status["total_dysrhythmias"]))
        st.add_row("Prediction Hit Rate",f"{status['prediction_hit_rate']:.3f}")
        st.add_row("Timing Smoothness", f"{status['smoothness']:.3f}")
        st.add_row("Avg Correction",    f"{status['avg_correction']:.4f}")
        console.print(st)

        if status["top_skills"]:
            sk = Table(box=box.SIMPLE, title="Top Skill Models", title_style="green")
            sk.add_column("Module",   style="cyan")
            sk.add_column("Pattern",  style="dim")
            sk.add_column("Level",    style="white")
            sk.add_column("Auto",     justify="right")
            sk.add_column("Accuracy", justify="right")
            sk.add_column("Samples",  justify="right")
            for s in status["top_skills"]:
                sc2 = SKILL_COLORS.get(s["skill"], "white")
                sk.add_row(
                    s["module"], s["pattern"][:16],
                    f"[{sc2}]{s['skill']}[/{sc2}]",
                    f"{s['automaticity']:.3f}",
                    f"{s['accuracy']:.3f}",
                    str(s["samples"])
                )
            console.print(sk)

        if status["top_routines"]:
            console.print(Rule("[dim]Consolidated Routines[/dim]"))
            for r in status["top_routines"]:
                console.print(
                    f"  [bright_green]★[/bright_green]  "
                    f"[cyan]{r['module']:<14}[/cyan]  "
                    f"[dim]{r['pattern']:<20}[/dim]  "
                    f"speed+{r['speed_bonus']:.2f}  "
                    f"err-{r['error_reduction']:.2f}"
                )


# ─── HTTP API ─────────────────────────────────────────────────────────────────

def run_api(cb: ForgeCerebellum):
    if not HAS_FLASK: return
    app = Flask(__name__)

    @app.route("/process", methods=["POST"])
    def process():
        body = request.json or {}
        return jsonify(cb.process(
            body.get("signal", {}),
            body.get("module_outputs", {})
        ))

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(cb.get_status())

    @app.route("/models", methods=["GET"])
    def models():
        rows = cb.db.get_models(1)
        return jsonify([{
            "module": r[0], "pattern": r[1], "samples": r[2],
            "accuracy": r[3], "skill": r[4],
            "automaticity": r[5], "error_mean": r[6]
        } for r in rows])

    @app.route("/corrections", methods=["GET"])
    def corrections():
        rows = cb.db.get_recent_corrections(15)
        return jsonify([{
            "cycle": r[0], "module": r[1],
            "raw": r[2], "corrected": r[3],
            "delta": r[4], "type": r[5], "accuracy": r[6]
        } for r in rows])

    @app.route("/timing", methods=["GET"])
    def timing():
        rows = cb.db.get_timing_history(10)
        return jsonify([{
            "cycle": r[0], "state": r[1],
            "jitter": r[2], "corrected": bool(r[3])
        } for r in rows])

    app.run(host="0.0.0.0", port=API_PORT, debug=False)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cb = ForgeCerebellum()
    if "--api" in sys.argv:
        t = threading.Thread(target=run_api, args=(cb,), daemon=True)
        t.start()
    run_demo()
