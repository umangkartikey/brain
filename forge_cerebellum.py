"""
FORGE Cerebellum — forge_cerebellum.py
========================================
AI analog of the brain's cerebellum.

The cerebellum is the brain's PRECISION ENGINE.
It doesn't initiate — it perfects.

Key insight: The cerebellum receives a copy of every motor command
BEFORE it executes, predicts what the outcome should be, then
compares prediction to actual. The difference (error signal) is
used to refine the next attempt.

This is called the FORWARD MODEL — predict → act → compare → correct.

Four core functions:

  1. TIMING COORDINATION
     Ensures all modules fire in the correct temporal sequence.
     Detects when modules are out of sync (temporal dysmetria).
     Maintains cognitive rhythm — the brain's internal clock.

  2. ERROR CORRECTION (Forward Model)
     Before an action: predicts expected outcome
     After an action:  compares prediction vs actual
     Difference:       sends correction signal back
     Next time:        prediction is more accurate
     This is how FORGE gets SMOOTHER over time.

  3. MOTOR LEARNING (LTD — Long-Term Depression)
     Repeated errors weaken the synapses that caused them.
     Repeated successes strengthen the correct pathways.
     The cerebellum learns by SUBTRACTION — removing what's wrong.

  4. SEQUENCE OPTIMIZATION
     Breaks complex actions into precise timed sub-steps.
     Optimizes the inter-module timing for minimum latency.
     Detects when the pipeline has timing anomalies.

Architecture:
  ForwardModel         → predict outcome before acting
  ErrorCalculator      → actual - predicted = error signal
  TimingCoordinator    → synchronize module firing sequences
  SequenceOptimizer    → optimize multi-step action timing
  PurkinjeCells        → learn from error (LTD mechanism)
  GranuleCells         → pattern storage for timing sequences
  OliveInput           → error signal broadcaster
  AdaptiveFilter       → smooth out noisy pipeline behavior
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
    from rich.text import Text
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
API_PORT = 7793
VERSION  = "1.0.0"

# Timing thresholds (milliseconds)
TIMING_IDEAL     = {
    "reflex":        8.0,
    "fast":          50.0,
    "deliberate":    260.0,
    "pipeline_full": 300.0,
    "module_ping":   5.0,
}
TIMING_TOLERANCE = 0.25   # 25% tolerance before flagging

# Learning rates
LTD_RATE         = 0.08   # Long-Term Depression (error weakens)
LTP_RATE         = 0.05   # Long-Term Potentiation (success strengthens)
PREDICTION_ALPHA = 0.20   # Forward model update rate

# Dysmetria thresholds
DYSMETRIA_MILD   = 1.5    # 1.5× expected timing
DYSMETRIA_SEVERE = 3.0    # 3× expected timing

console = Console() if HAS_RICH else None

# ─── Enums ────────────────────────────────────────────────────────────────────

class TimingStatus(Enum):
    PRECISE   = "PRECISE"     # within tolerance
    EARLY     = "EARLY"       # faster than expected
    LATE      = "LATE"        # slower than expected
    DYSMETRIC = "DYSMETRIC"   # severely off timing

class ErrorType(Enum):
    NONE      = "NONE"
    TIMING    = "TIMING"      # wrong timing
    SEQUENCE  = "SEQUENCE"    # wrong order
    MAGNITUDE = "MAGNITUDE"   # wrong amplitude
    DIRECTION = "DIRECTION"   # wrong direction

class LearningPhase(Enum):
    ACQUISITION  = "ACQUISITION"  # learning new pattern
    CONSOLIDATION= "CONSOLIDATION"# strengthening
    REFINEMENT   = "REFINEMENT"   # fine-tuning
    EXPERT       = "EXPERT"       # fully optimized

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class TimingRecord:
    """Records actual vs predicted timing for a module."""
    module:       str   = ""
    operation:    str   = ""
    predicted_ms: float = 0.0
    actual_ms:    float = 0.0
    error_ms:     float = 0.0
    error_pct:    float = 0.0
    status:       str   = TimingStatus.PRECISE.value
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:        int   = 0

@dataclass
class ForwardPrediction:
    """A prediction made before an action executes."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    module:       str   = ""
    action:       str   = ""
    predicted_latency_ms: float = 0.0
    predicted_outcome:    str   = ""
    predicted_success_prob:float= 0.5
    context_hash: str   = ""
    cycle:        int   = 0

@dataclass
class ErrorSignal:
    """Error signal computed after action completes."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    prediction_id:str   = ""
    module:       str   = ""
    action:       str   = ""
    error_type:   str   = ErrorType.NONE.value
    timing_error: float = 0.0   # actual - predicted latency
    outcome_error:float = 0.0   # 0=perfect, 1=completely wrong
    magnitude:    float = 0.0   # how large the error was
    correction:   dict  = field(default_factory=dict)
    cycle:        int   = 0

@dataclass
class PurkinjeCell:
    """
    Models a Purkinje cell — the learning unit of the cerebellum.
    Each Purkinje cell is responsible for one module/action pattern.
    Receives climbing fiber input (error) and parallel fiber input (context).
    LTD weakens the synapses that caused errors.
    """
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    module:       str   = ""
    action:       str   = ""
    synaptic_weight: float = 0.5   # 0-1 strength of this cell
    error_history:   deque = field(default_factory=lambda: deque(maxlen=50))
    correction_count:int   = 0
    learning_phase:  str   = LearningPhase.ACQUISITION.value
    timing_bias:     float = 0.0   # learned timing offset

    @property
    def reliability(self) -> float:
        if not self.error_history: return 0.5
        recent = list(self.error_history)[-10:]
        return round(1.0 - sum(abs(e) for e in recent) / (len(recent) * 10), 3)

@dataclass
class SequencePattern:
    """A learned timing pattern for a multi-module sequence."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:         str   = ""
    modules:      list  = field(default_factory=list)   # ordered module sequence
    ideal_timings:list  = field(default_factory=list)   # ms between each step
    actual_timings:list = field(default_factory=list)   # observed timings
    success_count:int   = 0
    error_count:  int   = 0
    optimized:    bool  = False

@dataclass
class CerebellarOutput:
    """The cerebellum's output — corrections + timing adjustments."""
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:        int   = 0
    timing_corrections: dict = field(default_factory=dict)   # module → ms adjustment
    smoothing_weights:  dict = field(default_factory=dict)   # module → gain
    error_signals:      list = field(default_factory=list)
    dysmetric_modules:  list = field(default_factory=list)
    overall_smoothness: float = 1.0   # 0=chaotic, 1=perfectly smooth
    pipeline_rhythm:    str   = "STEADY"

# ─── Database ─────────────────────────────────────────────────────────────────

class CerebellumDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS timing_records (
                    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(4))),
                    module TEXT, operation TEXT,
                    predicted_ms REAL, actual_ms REAL,
                    error_ms REAL, error_pct REAL,
                    status TEXT, timestamp TEXT, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS forward_predictions (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    module TEXT, action TEXT,
                    predicted_latency_ms REAL,
                    predicted_outcome TEXT,
                    predicted_success_prob REAL,
                    context_hash TEXT, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS error_signals (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    prediction_id TEXT, module TEXT, action TEXT,
                    error_type TEXT, timing_error REAL,
                    outcome_error REAL, magnitude REAL,
                    correction TEXT, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS purkinje_cells (
                    id TEXT PRIMARY KEY, module TEXT, action TEXT,
                    synaptic_weight REAL, correction_count INTEGER,
                    learning_phase TEXT, timing_bias REAL
                );
                CREATE TABLE IF NOT EXISTS sequence_patterns (
                    id TEXT PRIMARY KEY, name TEXT,
                    modules TEXT, ideal_timings TEXT,
                    actual_timings TEXT, success_count INTEGER,
                    error_count INTEGER, optimized INTEGER
                );
            """)
            self.conn.commit()

    def save_timing(self, t: TimingRecord):
        with self.lock:
            self.conn.execute("""
                INSERT INTO timing_records
                (module, operation, predicted_ms, actual_ms,
                 error_ms, error_pct, status, timestamp, cycle)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (t.module, t.operation, t.predicted_ms, t.actual_ms,
                  t.error_ms, t.error_pct, t.status, t.timestamp, t.cycle))
            self.conn.commit()

    def save_prediction(self, p: ForwardPrediction):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO forward_predictions VALUES
                (?,?,?,?,?,?,?,?,?)
            """, (p.id, p.timestamp, p.module, p.action,
                  p.predicted_latency_ms, p.predicted_outcome,
                  p.predicted_success_prob, p.context_hash, p.cycle))
            self.conn.commit()

    def save_error(self, e: ErrorSignal):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO error_signals VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
            """, (e.id, e.timestamp, e.prediction_id, e.module,
                  e.action, e.error_type, e.timing_error,
                  e.outcome_error, e.magnitude,
                  json.dumps(e.correction), e.cycle))
            self.conn.commit()

    def save_purkinje(self, p: PurkinjeCell):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO purkinje_cells VALUES
                (?,?,?,?,?,?,?)
            """, (p.id, p.module, p.action, p.synaptic_weight,
                  p.correction_count, p.learning_phase, p.timing_bias))
            self.conn.commit()

    def save_sequence(self, s: SequencePattern):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO sequence_patterns VALUES
                (?,?,?,?,?,?,?,?)
            """, (s.id, s.name, json.dumps(s.modules),
                  json.dumps(s.ideal_timings), json.dumps(s.actual_timings),
                  s.success_count, s.error_count, int(s.optimized)))
            self.conn.commit()

    def get_timing_history(self, module: str, limit=20):
        with self.lock:
            return self.conn.execute("""
                SELECT module, operation, predicted_ms, actual_ms,
                       error_ms, status, timestamp
                FROM timing_records WHERE module=?
                ORDER BY timestamp DESC LIMIT ?
            """, (module, limit)).fetchall()

    def get_recent_errors(self, limit=20):
        with self.lock:
            return self.conn.execute("""
                SELECT module, action, error_type, timing_error,
                       outcome_error, magnitude, timestamp
                FROM error_signals ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_purkinje_cells(self):
        with self.lock:
            return self.conn.execute("""
                SELECT module, action, synaptic_weight,
                       correction_count, learning_phase, timing_bias
                FROM purkinje_cells ORDER BY synaptic_weight DESC
            """).fetchall()


# ─── Forward Model ────────────────────────────────────────────────────────────

class ForwardModel:
    """
    Predicts what SHOULD happen before it happens.
    The cerebellum's predictive engine.

    Before any module fires:
      → predict its latency
      → predict likely outcome
      → predict success probability

    After it fires:
      → compare prediction to reality
      → compute error
      → update model for next time
    """

    def __init__(self, db: CerebellumDB):
        self.db = db
        # Learned latency predictions per module
        self.latency_model: dict[str, float] = {
            "temporal":       25.0,
            "salience":       15.0,
            "visual":         20.0,
            "limbic":         30.0,
            "neuromodulator": 35.0,
            "amygdala":       12.0,
            "bridge":         40.0,
            "hippocampus":    120.0,
            "prefrontal":     25.0,
            "sensorimotor":   15.0,
            "basal_ganglia":  10.0,
            "thalamus":       10.0,
            "dmn":            5.0,
            "swarm":          12.0,
            "orchestrator":   300.0,
        }
        self.outcome_model: dict[str, float] = defaultdict(lambda: 0.7)
        self.predictions:   dict[str, ForwardPrediction] = {}

    def predict(self, module: str, action: str,
                context: dict, cycle: int) -> ForwardPrediction:
        """Make a prediction before the action executes."""
        import hashlib
        ctx_hash = hashlib.md5(
            json.dumps(sorted(context.items()), default=str).encode()
        ).hexdigest()[:8]

        predicted_ms = self.latency_model.get(module, 50.0)
        # Adjust for threat level
        threat = context.get("threat", 0)
        if threat >= 3: predicted_ms *= 0.85  # crisis = faster
        if threat == 0: predicted_ms *= 1.05  # calm = slightly slower

        success_prob = self.outcome_model[f"{module}:{action}"]

        pred = ForwardPrediction(
            module               = module,
            action               = action,
            predicted_latency_ms = round(predicted_ms, 2),
            predicted_outcome    = "success" if success_prob > 0.5 else "uncertain",
            predicted_success_prob = round(success_prob, 3),
            context_hash         = ctx_hash,
            cycle                = cycle
        )
        self.predictions[pred.id] = pred
        self.db.save_prediction(pred)
        return pred

    def update(self, pred: ForwardPrediction,
               actual_ms: float, actual_success: bool) -> float:
        """Update forward model based on actual outcome. Returns error."""
        # Timing error
        timing_error = actual_ms - pred.predicted_latency_ms

        # Update latency model (exponential moving average)
        old = self.latency_model.get(pred.module, actual_ms)
        self.latency_model[pred.module] = round(
            old * (1 - PREDICTION_ALPHA) + actual_ms * PREDICTION_ALPHA, 2
        )

        # Update success probability
        key = f"{pred.module}:{pred.action}"
        old_prob = self.outcome_model[key]
        actual_val = 1.0 if actual_success else 0.0
        self.outcome_model[key] = round(
            old_prob * (1 - PREDICTION_ALPHA) + actual_val * PREDICTION_ALPHA, 3
        )

        return timing_error

    def get_prediction(self, pred_id: str) -> Optional[ForwardPrediction]:
        return self.predictions.get(pred_id)


# ─── Error Calculator ─────────────────────────────────────────────────────────

class ErrorCalculator:
    """
    Computes the error signal — actual vs predicted.
    This is the climbing fiber input to the Purkinje cells.

    Error = actual - predicted

    Types of error:
      TIMING    → module was early or late
      MAGNITUDE → response was too strong or too weak
      SEQUENCE  → modules fired in wrong order
      NONE      → prediction was accurate
    """

    def compute(self, pred: ForwardPrediction,
                actual_ms: float, actual_success: bool,
                actual_outcome: str, cycle: int) -> ErrorSignal:

        timing_error   = round(actual_ms - pred.predicted_latency_ms, 2)
        outcome_error  = 0.0 if actual_success else 1.0
        if not actual_success and pred.predicted_success_prob > 0.7:
            outcome_error = pred.predicted_success_prob  # big surprise

        # Determine error type
        timing_pct = abs(timing_error) / max(pred.predicted_latency_ms, 1.0)
        if timing_pct > 0.5:        error_type = ErrorType.TIMING
        elif outcome_error > 0.5:   error_type = ErrorType.MAGNITUDE
        else:                       error_type = ErrorType.NONE

        magnitude = round(
            timing_pct * 0.6 + outcome_error * 0.4, 4
        )

        # Build correction signal
        correction = {}
        if abs(timing_error) > 10.0:
            correction["timing_adjust_ms"] = round(-timing_error * 0.3, 2)
        if outcome_error > 0.3:
            correction["confidence_penalty"] = round(-outcome_error * 0.2, 3)

        sig = ErrorSignal(
            prediction_id = pred.id,
            module        = pred.module,
            action        = pred.action,
            error_type    = error_type.value,
            timing_error  = timing_error,
            outcome_error = outcome_error,
            magnitude     = magnitude,
            correction    = correction,
            cycle         = cycle
        )
        return sig


# ─── Purkinje Cell Layer ──────────────────────────────────────────────────────

class PurkinjeCellLayer:
    """
    The learning layer of the cerebellum.

    Purkinje cells receive two inputs:
      1. Parallel fibers — context information (what's happening)
      2. Climbing fibers — error signals (what went wrong)

    When climbing fiber fires (error detected):
      → LTD weakens the parallel fiber synapses that were active
      → This is called Long-Term Depression
      → The cell "unlearns" the bad prediction

    When no error (success):
      → LTP slightly strengthens active synapses
      → Prediction gradually improves

    This is the ONLY learning mechanism in the cerebellum.
    Unlike the rest of the brain, it learns by subtraction.
    """

    def __init__(self, db: CerebellumDB):
        self.db    = db
        self.cells: dict[str, PurkinjeCell] = {}
        self._seed_cells()

    def _seed_cells(self):
        """Initialize Purkinje cells for each known module."""
        modules = ["temporal","salience","visual","limbic","neuromodulator",
                   "amygdala","bridge","hippocampus","prefrontal","sensorimotor",
                   "basal_ganglia","thalamus","dmn","swarm","orchestrator"]
        for mod in modules:
            cell_id = f"PC_{mod}"
            cell    = PurkinjeCell(id=cell_id, module=mod, action="general")
            self.cells[cell_id] = cell
            self.db.save_purkinje(cell)

    def apply_error(self, error: ErrorSignal) -> dict:
        """Apply error signal via LTD — weaken bad synapses."""
        cell_id = f"PC_{error.module}"
        if cell_id not in self.cells:
            cell = PurkinjeCell(id=cell_id, module=error.module, action=error.action)
            self.cells[cell_id] = cell

        cell = self.cells[cell_id]
        old_weight = cell.synaptic_weight

        if error.magnitude > 0.05:
            # LTD — weaken by error magnitude
            ltd_delta = LTD_RATE * error.magnitude
            cell.synaptic_weight = round(
                max(0.05, cell.synaptic_weight - ltd_delta), 4
            )
            # Apply timing bias correction
            cell.timing_bias = round(
                cell.timing_bias - error.timing_error * 0.1, 2
            )
            cell.correction_count += 1
            cell.error_history.append(error.magnitude)
        else:
            # LTP — small strengthening on success
            cell.synaptic_weight = round(
                min(0.98, cell.synaptic_weight + LTP_RATE * 0.1), 4
            )
            cell.error_history.append(0.0)

        # Update learning phase
        reliability = cell.reliability
        if reliability > 0.90:   cell.learning_phase = LearningPhase.EXPERT.value
        elif reliability > 0.75: cell.learning_phase = LearningPhase.REFINEMENT.value
        elif reliability > 0.55: cell.learning_phase = LearningPhase.CONSOLIDATION.value
        else:                    cell.learning_phase = LearningPhase.ACQUISITION.value

        self.db.save_purkinje(cell)

        return {
            "cell":          cell_id,
            "weight_before": round(old_weight, 4),
            "weight_after":  round(cell.synaptic_weight, 4),
            "delta":         round(cell.synaptic_weight - old_weight, 4),
            "phase":         cell.learning_phase,
            "reliability":   round(reliability, 3),
        }

    def get_smoothing_weights(self) -> dict[str, float]:
        """Get current synaptic weights for output smoothing."""
        return {
            cell.module: round(cell.synaptic_weight, 3)
            for cell in self.cells.values()
        }

    def get_timing_biases(self) -> dict[str, float]:
        """Get learned timing corrections per module."""
        return {
            cell.module: round(cell.timing_bias, 2)
            for cell in self.cells.values()
        }


# ─── Timing Coordinator ───────────────────────────────────────────────────────

class TimingCoordinator:
    """
    Maintains the cognitive rhythm — ensures modules fire
    in the correct temporal sequence.

    Detects timing anomalies:
      EARLY    — module fired before expected (may interrupt sequence)
      LATE     — module fired after window (may cause cascade delay)
      DYSMETRIC— severe timing violation (sequence coordination lost)

    Also maintains a rolling timing baseline per module.
    """

    def __init__(self, db: CerebellumDB):
        self.db      = db
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
        self.baseline:dict[str, float] = {}
        self.cycle   = 0

    def record(self, module: str, operation: str,
               actual_ms: float, predicted_ms: float) -> TimingRecord:
        self.cycle += 1
        error_ms  = round(actual_ms - predicted_ms, 2)
        error_pct = round(error_ms / max(predicted_ms, 1.0) * 100, 1)

        # Classify timing status
        ratio = actual_ms / max(predicted_ms, 1.0)
        if ratio > DYSMETRIA_SEVERE:
            status = TimingStatus.DYSMETRIC
        elif ratio > DYSMETRIA_MILD:
            status = TimingStatus.LATE
        elif ratio < (1.0 / DYSMETRIA_MILD):
            status = TimingStatus.EARLY
        else:
            status = TimingStatus.PRECISE

        record = TimingRecord(
            module      = module,
            operation   = operation,
            predicted_ms= round(predicted_ms, 2),
            actual_ms   = round(actual_ms, 2),
            error_ms    = error_ms,
            error_pct   = error_pct,
            status      = status.value,
            cycle       = self.cycle
        )
        self.db.save_timing(record)
        self.history[module].append(actual_ms)

        # Update rolling baseline
        hist = list(self.history[module])
        self.baseline[module] = round(sum(hist)/len(hist), 2)

        return record

    def get_dysmetric_modules(self) -> list[str]:
        """Modules with consistently poor timing."""
        dysmetric = []
        for module, hist in self.history.items():
            if len(hist) < 3: continue
            recent = list(hist)[-5:]
            baseline = self.baseline.get(module, 50.0)
            if any(ms > baseline * DYSMETRIA_MILD for ms in recent):
                dysmetric.append(module)
        return dysmetric

    def pipeline_smoothness(self) -> float:
        """0=chaotic, 1=perfectly smooth pipeline."""
        if not self.baseline: return 1.0
        variances = []
        for module, hist in self.history.items():
            if len(hist) < 3: continue
            mean = self.baseline[module]
            variance = sum((ms-mean)**2 for ms in hist) / len(hist)
            normalized = min(1.0, variance / (mean**2 + 1))
            variances.append(normalized)
        if not variances: return 1.0
        avg_variance = sum(variances) / len(variances)
        return round(max(0.0, 1.0 - avg_variance), 3)

    def rhythm_label(self, smoothness: float) -> str:
        if smoothness > 0.85: return "STEADY"
        if smoothness > 0.65: return "VARIABLE"
        if smoothness > 0.40: return "IRREGULAR"
        return "DYSRHYTHMIC"


# ─── Sequence Optimizer ───────────────────────────────────────────────────────

class SequenceOptimizer:
    """
    Learns and optimizes multi-module timing sequences.
    Like a conductor learning the orchestra's rhythms —
    over time the sequence becomes tighter and faster.
    """

    def __init__(self, db: CerebellumDB):
        self.db       = db
        self.sequences: dict[str, SequencePattern] = {}
        self._seed_pipeline_sequence()

    def _seed_pipeline_sequence(self):
        """Seed the main pipeline sequence."""
        pipeline = SequencePattern(
            id            = "SEQ_MAIN_PIPELINE",
            name          = "main_cognitive_pipeline",
            modules       = ["salience","temporal","bridge","limbic",
                             "prefrontal","hippocampus","swarm","dmn"],
            ideal_timings = [15.0, 25.0, 40.0, 30.0, 25.0, 120.0, 12.0, 5.0],
        )
        self.sequences[pipeline.id] = pipeline
        self.db.save_sequence(pipeline)

    def record_pipeline_run(self, module_timings: dict[str, float]):
        """Record actual timing for a pipeline run."""
        seq = self.sequences.get("SEQ_MAIN_PIPELINE")
        if not seq: return

        actual = [module_timings.get(m, 0.0) for m in seq.modules]
        seq.actual_timings = actual

        # Compare to ideal
        errors = [abs(a-i)/max(i,1) for a,i in zip(actual, seq.ideal_timings) if i>0]
        avg_error = sum(errors)/len(errors) if errors else 0

        if avg_error < 0.25:
            seq.success_count += 1
        else:
            seq.error_count += 1

        # Mark as optimized when consistently good
        if seq.success_count > 10 and seq.success_count > seq.error_count * 3:
            seq.optimized = True

        self.db.save_sequence(seq)

    def timing_recommendations(self) -> dict[str, float]:
        """Recommended timing adjustments based on learned sequences."""
        seq = self.sequences.get("SEQ_MAIN_PIPELINE")
        if not seq or not seq.actual_timings:
            return {}

        recommendations = {}
        for module, ideal, actual in zip(
            seq.modules, seq.ideal_timings, seq.actual_timings
        ):
            if actual > 0 and ideal > 0:
                delta = ideal - actual  # positive = need to be faster
                recommendations[module] = round(delta * 0.2, 2)  # gentle nudge

        return recommendations


# ─── Adaptive Filter ──────────────────────────────────────────────────────────

class AdaptiveFilter:
    """
    Smooths out noisy pipeline behavior.
    Uses Purkinje cell weights to weight module outputs —
    more reliable modules get higher gain.

    Also applies the timing corrections from the forward model
    to predict when each module should fire next.
    """

    def apply(self, module_outputs: dict,
              purkinje_weights: dict,
              timing_corrections: dict) -> dict:
        """
        Apply cerebellar smoothing to module outputs.
        Returns adjusted outputs.
        """
        adjusted = {}
        for module, output in module_outputs.items():
            weight     = purkinje_weights.get(module, 0.7)
            correction = timing_corrections.get(module, 0.0)

            # Scale confidence by Purkinje weight
            if isinstance(output, dict):
                adjusted_output = dict(output)
                # Apply weight to any confidence/score fields
                for key in ["confidence","score","strength","probability"]:
                    if key in adjusted_output and isinstance(adjusted_output[key], (int,float)):
                        adjusted_output[key] = round(
                            adjusted_output[key] * weight, 4
                        )
                adjusted_output["_cerebellum_weight"]     = weight
                adjusted_output["_timing_correction_ms"]  = correction
                adjusted[module] = adjusted_output
            else:
                adjusted[module] = output

        return adjusted


# ─── Olive Input (Error Broadcaster) ─────────────────────────────────────────

class OliveInput:
    """
    Models the inferior olive — the brain structure that
    broadcasts error signals to the entire cerebellum.

    In the real brain, olive firing is rare (~1Hz) and each
    spike represents a significant prediction error.

    In FORGE, the olive fires when:
      - A module's timing error exceeds threshold
      - An action's outcome was unexpected
      - The pipeline sequence breaks
    """

    def __init__(self):
        self.olive_fires:  deque = deque(maxlen=100)
        self.total_fires   = 0
        self.last_fire_ms  = 0.0

    def should_fire(self, error: ErrorSignal) -> bool:
        """Does this error warrant an olive signal?"""
        return error.magnitude > 0.15

    def fire(self, error: ErrorSignal) -> dict:
        """Broadcast error signal to cerebellum."""
        self.total_fires += 1
        self.last_fire_ms = error.timing_error
        event = {
            "fire_count": self.total_fires,
            "module":     error.module,
            "error_type": error.error_type,
            "magnitude":  error.magnitude,
            "timestamp":  datetime.now().isoformat()
        }
        self.olive_fires.append(event)
        return event


# ─── FORGE Cerebellum ─────────────────────────────────────────────────────────

class ForgeCerebellum:
    def __init__(self):
        self.db           = CerebellumDB()
        self.forward_model= ForwardModel(self.db)
        self.error_calc   = ErrorCalculator()
        self.purkinje     = PurkinjeCellLayer(self.db)
        self.timing       = TimingCoordinator(self.db)
        self.optimizer    = SequenceOptimizer(self.db)
        self.filter       = AdaptiveFilter()
        self.olive        = OliveInput()
        self.cycle        = 0
        self.total_errors = 0
        self.total_corrections = 0
        self.pending_predictions: dict[str, ForwardPrediction] = {}

    def before_action(self, module: str, action: str,
                      context: dict) -> dict:
        """
        Called BEFORE a module fires.
        Makes a forward prediction.
        """
        self.cycle += 1
        pred = self.forward_model.predict(module, action, context, self.cycle)
        self.pending_predictions[module] = pred

        return {
            "prediction_id":        pred.id,
            "predicted_latency_ms": pred.predicted_latency_ms,
            "predicted_success":    pred.predicted_outcome,
            "timing_bias":          self.purkinje.get_timing_biases().get(module, 0.0),
        }

    def after_action(self, module: str, actual_ms: float,
                     success: bool = True,
                     actual_outcome: str = "success") -> dict:
        """
        Called AFTER a module fires.
        Computes error, applies LTD, sends olive signal if needed.
        """
        pred = self.pending_predictions.pop(module, None)
        if not pred:
            # No prediction exists — record timing only
            rec = self.timing.record(module, "unknown", actual_ms,
                                     self.forward_model.latency_model.get(module, actual_ms))
            return {"module": module, "status": rec.status, "correction": {}}

        # Update forward model
        timing_error = self.forward_model.update(pred, actual_ms, success)

        # Record timing
        rec = self.timing.record(module, pred.action, actual_ms,
                                  pred.predicted_latency_ms)

        # Compute error signal
        error = self.error_calc.compute(pred, actual_ms, success,
                                         actual_outcome, self.cycle)
        self.db.save_error(error)

        # Apply Purkinje LTD/LTP
        purkinje_result = self.purkinje.apply_error(error)

        # Olive fires on significant error
        olive_event = None
        if self.olive.should_fire(error):
            olive_event = self.olive.fire(error)
            self.total_errors += 1

        if error.correction:
            self.total_corrections += 1

        return {
            "module":          module,
            "actual_ms":       actual_ms,
            "predicted_ms":    pred.predicted_latency_ms,
            "timing_error_ms": error.timing_error,
            "timing_status":   rec.status,
            "error_type":      error.error_type,
            "magnitude":       error.magnitude,
            "correction":      error.correction,
            "purkinje":        purkinje_result,
            "olive_fired":     olive_event is not None,
        }

    def observe_pipeline(self, module_timings: dict[str, float],
                         success: bool = True) -> CerebellarOutput:
        """
        Observe a full pipeline run.
        Records sequence timing, computes smoothness, emits corrections.
        """
        # Record sequence
        self.optimizer.record_pipeline_run(module_timings)

        # Get smoothness
        smoothness = self.timing.pipeline_smoothness()
        dysmetric  = self.timing.get_dysmetric_modules()

        # Get corrections from optimizer
        timing_corrections = self.optimizer.timing_recommendations()

        # Get smoothing weights from Purkinje cells
        smoothing_weights = self.purkinje.get_smoothing_weights()

        output = CerebellarOutput(
            cycle               = self.cycle,
            timing_corrections  = timing_corrections,
            smoothing_weights   = smoothing_weights,
            dysmetric_modules   = dysmetric,
            overall_smoothness  = smoothness,
            pipeline_rhythm     = self.timing.rhythm_label(smoothness)
        )
        return output

    def get_status(self) -> dict:
        cells = list(self.purkinje.cells.values())
        return {
            "version":           VERSION,
            "cycle":             self.cycle,
            "total_errors":      self.total_errors,
            "total_corrections": self.total_corrections,
            "olive_fires":       self.olive.total_fires,
            "pipeline_smoothness": self.timing.pipeline_smoothness(),
            "pipeline_rhythm":   self.timing.rhythm_label(
                self.timing.pipeline_smoothness()
            ),
            "dysmetric_modules": self.timing.get_dysmetric_modules(),
            "purkinje_summary": {
                "total_cells":   len(cells),
                "avg_weight":    round(sum(c.synaptic_weight for c in cells)/max(len(cells),1), 3),
                "expert_cells":  sum(1 for c in cells if c.learning_phase == "EXPERT"),
            },
            "forward_model": {
                "modules_tracked": len(self.forward_model.latency_model),
                "top_latencies":   dict(sorted(
                    self.forward_model.latency_model.items(),
                    key=lambda x: x[1], reverse=True
                )[:5])
            }
        }


# ─── Rich UI ──────────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "PRECISE":   "green",
    "EARLY":     "cyan",
    "LATE":      "yellow",
    "DYSMETRIC": "bright_red",
}

PHASE_COLORS = {
    "ACQUISITION":   "dim",
    "CONSOLIDATION": "yellow",
    "REFINEMENT":    "cyan",
    "EXPERT":        "bright_green",
}

def render_cerebellum(pipeline_result: dict,
                      output: CerebellarOutput,
                      idx: int):
    if not HAS_RICH: return

    smoothness = output.overall_smoothness
    rhythm     = output.pipeline_rhythm
    rc = {"STEADY":"green","VARIABLE":"yellow",
          "IRREGULAR":"orange3","DYSRHYTHMIC":"bright_red"}.get(rhythm,"white")

    console.print(Rule(
        f"[bold cyan]⬡ FORGE CEREBELLUM[/bold cyan]  "
        f"[dim]#{idx}[/dim]  "
        f"[{rc}]{rhythm}[/{rc}]  "
        f"smoothness={smoothness:.3f}"
    ))

    # Module timing table
    timing_table = Table(box=box.SIMPLE, show_header=True, expand=True)
    timing_table.add_column("Module",    style="dim", width=14)
    timing_table.add_column("Actual",    justify="right", width=8)
    timing_table.add_column("Predicted", justify="right", width=10)
    timing_table.add_column("Error",     justify="right", width=8)
    timing_table.add_column("Status",    width=12)
    timing_table.add_column("Purkinje",  justify="right", width=10)

    for mod, data in pipeline_result.items():
        if not isinstance(data, dict): continue
        act   = data.get("actual_ms", 0)
        pred  = data.get("predicted_ms", 0)
        err   = data.get("timing_error_ms", 0)
        stat  = data.get("timing_status", "PRECISE")
        sc    = STATUS_COLORS.get(stat, "white")
        pur   = data.get("purkinje",{}).get("weight_after", 0.5) if data.get("purkinje") else 0.5
        pc    = "bright_green" if pur>0.8 else "green" if pur>0.6 else "yellow" if pur>0.4 else "red"

        timing_table.add_row(
            mod,
            f"{act:.1f}ms",
            f"{pred:.1f}ms",
            f"[{sc}]{err:+.1f}[/{sc}]",
            f"[{sc}]{stat}[/{sc}]",
            f"[{pc}]{pur:.3f}[/{pc}]"
        )

    # Corrections panel
    corr_lines = []
    if output.timing_corrections:
        corr_lines.append("[bold]Timing recommendations:[/bold]")
        for mod, delta in output.timing_corrections.items():
            color = "green" if delta > 0 else "red"
            corr_lines.append(
                f"  {mod:<14} [{color}]{delta:+.1f}ms[/{color}]"
            )
    if output.dysmetric_modules:
        corr_lines.append(f"\n[bold red]Dysmetric:[/bold red]")
        for m in output.dysmetric_modules:
            corr_lines.append(f"  [red]⚠ {m}[/red]")
    if not corr_lines:
        corr_lines = ["[dim]All modules in rhythm ✓[/dim]"]

    console.print(Columns([
        Panel(timing_table, title="[bold]Module Timing[/bold]", border_style="dim"),
        Panel("\n".join(corr_lines), title="[bold]Corrections[/bold]", border_style=rc)
    ]))


def run_demo():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FORGE CEREBELLUM[/bold cyan]\n"
            "[dim]Timing · Error Correction · Forward Model · Purkinje Learning[/dim]\n"
            f"[dim]Version {VERSION}[/dim]",
            border_style="cyan"
        ))

    cerebellum = ForgeCerebellum()

    # Simulate several pipeline runs with realistic timing data
    # Watch the cerebellum learn and smooth the pipeline over time

    pipeline_scenarios = [
        # Run 1 — first time, timing is rough
        {
            "label": "First run — rough timing",
            "timings": {
                "salience":     18.0,   # a bit slow
                "temporal":     31.0,   # slow
                "bridge":       52.0,   # very slow
                "limbic":       28.0,   # ok
                "prefrontal":   89.0,   # way too slow (dysmetric)
                "hippocampus": 145.0,   # normal for hippocampus
                "swarm":        15.0,   # fast
                "dmn":           6.0,   # ok
            },
            "success": True
        },
        # Run 2 — cerebellum starts correcting
        {
            "label": "Second run — correction begins",
            "timings": {
                "salience":     16.0,
                "temporal":     28.0,
                "bridge":       45.0,
                "limbic":       27.0,
                "prefrontal":   60.0,   # improved
                "hippocampus": 138.0,
                "swarm":        13.0,
                "dmn":           5.5,
            },
            "success": True
        },
        # Run 3 — crisis signal, timing changes
        {
            "label": "Crisis — timing shifts under threat",
            "timings": {
                "salience":      9.0,   # faster (threat)
                "temporal":     22.0,
                "bridge":       38.0,
                "limbic":       18.0,   # emotion speeds up
                "prefrontal":   42.0,
                "hippocampus": 155.0,   # memory encoding stronger
                "swarm":         8.0,   # swarm on alert
                "dmn":           2.0,   # DMN suppressed
            },
            "success": True
        },
        # Run 4 — recovery, smoothing
        {
            "label": "Recovery — smoothing in",
            "timings": {
                "salience":     14.0,
                "temporal":     25.0,
                "bridge":       41.0,
                "limbic":       26.0,
                "prefrontal":   28.0,   # much better
                "hippocampus": 122.0,
                "swarm":        11.0,
                "dmn":           5.0,
            },
            "success": True
        },
        # Run 5 — approaching steady state
        {
            "label": "Steady state — cerebellum optimized",
            "timings": {
                "salience":     14.5,
                "temporal":     24.5,
                "bridge":       40.5,
                "limbic":       25.5,
                "prefrontal":   25.5,
                "hippocampus": 120.5,
                "swarm":        11.5,
                "dmn":           5.0,
            },
            "success": True
        },
    ]

    for i, scenario in enumerate(pipeline_scenarios):
        if HAS_RICH:
            console.print(f"\n[bold dim]━━━ RUN {i+1}: {scenario['label'].upper()} ━━━[/bold dim]")

        timings = scenario["timings"]
        pipeline_result = {}

        # Before each module
        for mod in timings:
            cerebellum.before_action(mod, "process", {"threat": 3 if i==2 else 0})

        # After each module
        for mod, actual_ms in timings.items():
            result = cerebellum.after_action(
                mod, actual_ms, scenario["success"]
            )
            pipeline_result[mod] = result

        # Observe full pipeline
        output = cerebellum.observe_pipeline(timings, scenario["success"])
        render_cerebellum(pipeline_result, output, i+1)
        time.sleep(0.1)

    # Final status
    if HAS_RICH:
        console.print(Rule("[bold cyan]⬡ CEREBELLUM FINAL STATUS[/bold cyan]"))
        status = cerebellum.get_status()

        st = Table(box=box.DOUBLE_EDGE, border_style="cyan", title="Cerebellum Status")
        st.add_column("Metric", style="cyan")
        st.add_column("Value",  style="white")
        st.add_row("Cycles",            str(status["cycle"]))
        st.add_row("Total Errors",      str(status["total_errors"]))
        st.add_row("Corrections Made",  str(status["total_corrections"]))
        st.add_row("Olive Fires",       str(status["olive_fires"]))
        st.add_row("Pipeline Smoothness",f"{status['pipeline_smoothness']:.3f}")
        st.add_row("Pipeline Rhythm",   status["pipeline_rhythm"])
        st.add_row("Dysmetric Modules", str(status["dysmetric_modules"] or "none"))
        st.add_row("Expert Cells",      str(status["purkinje_summary"]["expert_cells"]))
        st.add_row("Avg Purkinje Weight",f"{status['purkinje_summary']['avg_weight']:.3f}")
        console.print(st)

        # Forward model latencies
        console.print(Rule("[dim]Learned Latencies[/dim]"))
        lat_table = Table(box=box.SIMPLE, show_header=False)
        lat_table.add_column("module", style="dim", width=16)
        lat_table.add_column("predicted")
        for mod, ms in sorted(
            status["forward_model"]["top_latencies"].items(),
            key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(min(ms/10, 20)) + f" {ms:.1f}ms"
            lat_table.add_row(mod, bar)
        console.print(lat_table)


# ─── HTTP API ─────────────────────────────────────────────────────────────────

def run_api(cb: ForgeCerebellum):
    if not HAS_FLASK: return
    app = Flask(__name__)

    @app.route("/before", methods=["POST"])
    def before():
        data = request.json or {}
        return jsonify(cb.before_action(
            data.get("module",""),
            data.get("action",""),
            data.get("context",{})
        ))

    @app.route("/after", methods=["POST"])
    def after():
        data = request.json or {}
        return jsonify(cb.after_action(
            data.get("module",""),
            data.get("actual_ms", 0.0),
            data.get("success", True),
            data.get("outcome","success")
        ))

    @app.route("/observe", methods=["POST"])
    def observe():
        data  = request.json or {}
        out   = cb.observe_pipeline(
            data.get("timings",{}),
            data.get("success", True)
        )
        return jsonify({
            "smoothness":        out.overall_smoothness,
            "rhythm":            out.pipeline_rhythm,
            "dysmetric":         out.dysmetric_modules,
            "corrections":       out.timing_corrections,
            "smoothing_weights": out.smoothing_weights,
        })

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(cb.get_status())

    app.run(host="0.0.0.0", port=API_PORT, debug=False)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cb = ForgeCerebellum()
    if "--api" in sys.argv:
        t = threading.Thread(target=run_api, args=(cb,), daemon=True)
        t.start()
    run_demo()
