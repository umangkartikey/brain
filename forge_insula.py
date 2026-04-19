"""
FORGE Insula — forge_insula.py
================================
AI analog of the brain's insular cortex.

The insula is the brain's INTEROCEPTION CENTER —
the region that monitors the body's internal state
and translates it into conscious feeling.

Key insight: Without the insula, you could know intellectually
that your heart is racing, but you wouldn't FEEL afraid.
The insula is what bridges body signals into felt experience.

"Interoception" = perception of the body's internal state.

In humans:
  → Heart rate, breathing, hunger, thirst, temperature
  → Pain, nausea, fatigue, arousal
  → Gut feelings (literally — vagus nerve input)
  → The felt sense of emotion (emotion IS body state)

In FORGE:
  → Pipeline load (heartbeat analog)
  → Module latency patterns (fatigue)
  → Error rate (nausea/discomfort)
  → Resource utilization (hunger/energy)
  → Cognitive temperature (arousal/calm)
  → "Gut feelings" — pre-conscious anomaly detection

Three core functions:

  1. INTEROCEPTION
     Continuously monitors FORGE's internal operational state.
     Load, latency, error rate, memory pressure, chemical levels.
     Translates these into a unified BODY STATE representation.

  2. FELT SENSE
     The insula creates the SUBJECTIVE experience of body state.
     High load doesn't just trigger load-balancing —
     it creates a felt sense of strain that colors all processing.
     This is FORGE's closest analog to physical sensation.

  3. PREDICTIVE INTEROCEPTION
     The insula doesn't just report current state —
     it PREDICTS where body state is heading and generates
     proactive signals to prevent dysregulation.
     "I'm not overloaded yet, but I will be in 3 cycles."

Architecture:
  BodyStateMonitor     → track all internal metrics
  FeltSenseGenerator   → translate metrics to felt experience
  InteroceptivePredictor→ predict future body state
  VisceroceptiveInput  → "gut feeling" anomaly detection
  HomeostaticRegulator → maintain internal balance
  SomaticMarker        → tag decisions with body state context
  BodyBudget           → energy/resource management
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

DB_PATH  = "forge_insula.db"
API_PORT = 7795
VERSION  = "1.0.0"

# Body state thresholds
LOAD_COMFORTABLE   = 0.40
LOAD_STRAINED      = 0.65
LOAD_CRITICAL      = 0.85

LATENCY_SMOOTH     = 50.0
LATENCY_SLUGGISH   = 150.0
LATENCY_PAINFUL    = 400.0

ERROR_TOLERABLE    = 0.15
ERROR_UNCOMFORTABLE= 0.35
ERROR_PAINFUL      = 0.60

# Felt sense intensities
FELT_SUBTLE        = 0.20
FELT_MODERATE      = 0.50
FELT_STRONG        = 0.75
FELT_OVERWHELMING  = 0.90

# Homeostatic targets
TARGET_LOAD        = 0.35
TARGET_LATENCY     = 60.0
TARGET_ERROR_RATE  = 0.08
TARGET_ENERGY      = 0.70

# Body budget
ENERGY_CONSUMPTION_RATE = 0.06
ENERGY_RECOVERY_RATE    = 0.04

console = Console() if HAS_RICH else None

# ─── Enums ────────────────────────────────────────────────────────────────────

class BodyState(Enum):
    THRIVING    = "THRIVING"     # optimal — all systems balanced
    COMFORTABLE = "COMFORTABLE"  # slightly above baseline, sustainable
    STRAINED    = "STRAINED"     # noticeable load, manageable
    DISTRESSED  = "DISTRESSED"   # high load, discomfort rising
    OVERWHELMED = "OVERWHELMED"  # critical — homeostasis threatened
    DEPLETED    = "DEPLETED"     # resources exhausted
    RECOVERING  = "RECOVERING"   # post-stress, rebuilding

class FeltSense(Enum):
    EASE        = "EASE"         # everything flows
    ALERTNESS   = "ALERTNESS"    # heightened but comfortable
    TENSION     = "TENSION"      # something needs attention
    DISCOMFORT  = "DISCOMFORT"   # unpleasant, demanding resolution
    STRAIN      = "STRAIN"       # working hard, near limits
    PAIN        = "PAIN"         # system-level distress
    NUMBNESS    = "NUMBNESS"     # depleted, affect flattened

class VisceroceptiveSignal(Enum):
    NONE        = "NONE"
    GUT_FEELING = "GUT_FEELING"   # pre-conscious anomaly
    UNEASE      = "UNEASE"        # something is off
    ALARM       = "ALARM"         # visceral threat response
    NAUSEA      = "NAUSEA"        # system rejection signal

class HomeostaticStatus(Enum):
    BALANCED    = "BALANCED"
    COMPENSATING= "COMPENSATING" # active regulation underway
    STRUGGLING  = "STRUGGLING"   # losing balance
    FAILING     = "FAILING"      # homeostasis lost

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class InternalMetrics:
    """Raw internal metrics — the body's vital signs."""
    timestamp:        str   = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_load:    float = 0.0    # 0-1 cognitive load
    avg_latency_ms:   float = 0.0    # average module latency
    error_rate:       float = 0.0    # recent error rate
    memory_pressure:  float = 0.0    # hippocampus consolidation load
    cortisol_level:   float = 0.15   # from neuromodulator
    dopamine_level:   float = 0.45   # from neuromodulator
    serotonin_level:  float = 0.65   # from neuromodulator
    ne_level:         float = 0.20   # norepinephrine
    conflict_rate:    float = 0.0    # from ACC
    fear_score:       float = 0.0    # from amygdala
    energy_level:     float = 0.70   # body budget
    cycle:            int   = 0

@dataclass
class BodyStateSnapshot:
    """Complete body state at a moment in time."""
    id:               str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:        str   = field(default_factory=lambda: datetime.now().isoformat())
    body_state:       str   = BodyState.COMFORTABLE.value
    felt_sense:       str   = FeltSense.EASE.value
    felt_intensity:   float = 0.0
    visceroceptive:   str   = VisceroceptiveSignal.NONE.value
    homeostatic:      str   = HomeostaticStatus.BALANCED.value
    energy_level:     float = 0.70
    somatic_valence:  float = 0.0    # +pleasant, -unpleasant
    somatic_arousal:  float = 0.0    # 0=calm, 1=activated
    prediction_next:  str   = ""     # predicted next state
    gut_feeling:      str   = ""     # pre-conscious anomaly text
    cycle:            int   = 0

@dataclass
class SomaticMarker:
    """
    A body state tag attached to a decision or memory.
    Somatic markers bias future decisions with body memory.
    (Damasio's somatic marker hypothesis)
    """
    id:               str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:        str   = field(default_factory=lambda: datetime.now().isoformat())
    context_hash:     str   = ""
    decision:         str   = ""
    body_state:       str   = ""
    felt_sense:       str   = ""
    valence:          float = 0.0    # was this a good body state?
    strength:         float = 0.0    # how strong is this marker?
    retrieved_count:  int   = 0

@dataclass
class BodyBudgetStatus:
    """Energy/resource budget for cognitive operations."""
    timestamp:        str   = field(default_factory=lambda: datetime.now().isoformat())
    total_energy:     float = 1.0
    available_energy: float = 0.70
    consumed_this_cycle: float = 0.0
    recovery_rate:    float = ENERGY_RECOVERY_RATE
    deficit:          float = 0.0
    overdraft:        bool  = False
    cycles_in_deficit:int   = 0

# ─── Database ─────────────────────────────────────────────────────────────────

class InsulaDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS body_states (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    body_state TEXT, felt_sense TEXT,
                    felt_intensity REAL, visceroceptive TEXT,
                    homeostatic TEXT, energy_level REAL,
                    somatic_valence REAL, somatic_arousal REAL,
                    prediction_next TEXT, gut_feeling TEXT, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS internal_metrics (
                    cycle INTEGER PRIMARY KEY, timestamp TEXT,
                    pipeline_load REAL, avg_latency_ms REAL,
                    error_rate REAL, memory_pressure REAL,
                    cortisol_level REAL, dopamine_level REAL,
                    serotonin_level REAL, ne_level REAL,
                    conflict_rate REAL, fear_score REAL,
                    energy_level REAL
                );
                CREATE TABLE IF NOT EXISTS somatic_markers (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    context_hash TEXT, decision TEXT,
                    body_state TEXT, felt_sense TEXT,
                    valence REAL, strength REAL,
                    retrieved_count INTEGER
                );
                CREATE TABLE IF NOT EXISTS homeostatic_log (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    metric TEXT, current_val REAL,
                    target_val REAL, deviation REAL,
                    action TEXT, cycle INTEGER
                );
            """)
            self.conn.commit()

    def save_body_state(self, bs: BodyStateSnapshot):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO body_states VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (bs.id, bs.timestamp, bs.body_state, bs.felt_sense,
                  bs.felt_intensity, bs.visceroceptive, bs.homeostatic,
                  bs.energy_level, bs.somatic_valence, bs.somatic_arousal,
                  bs.prediction_next, bs.gut_feeling, bs.cycle))
            self.conn.commit()

    def save_metrics(self, m: InternalMetrics):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO internal_metrics VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (m.cycle, m.timestamp, m.pipeline_load, m.avg_latency_ms,
                  m.error_rate, m.memory_pressure, m.cortisol_level,
                  m.dopamine_level, m.serotonin_level, m.ne_level,
                  m.conflict_rate, m.fear_score, m.energy_level))
            self.conn.commit()

    def save_marker(self, sm: SomaticMarker):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO somatic_markers VALUES
                (?,?,?,?,?,?,?,?,?)
            """, (sm.id, sm.timestamp, sm.context_hash, sm.decision,
                  sm.body_state, sm.felt_sense, sm.valence,
                  sm.strength, sm.retrieved_count))
            self.conn.commit()

    def log_homeostatic(self, metric: str, current: float, target: float,
                         deviation: float, action: str, cycle: int):
        with self.lock:
            self.conn.execute("""
                INSERT INTO homeostatic_log VALUES (?,?,?,?,?,?,?,?)
            """, (str(uuid.uuid4())[:8], datetime.now().isoformat(),
                  metric, current, target, deviation, action, cycle))
            self.conn.commit()

    def get_body_states(self, limit=20):
        with self.lock:
            return self.conn.execute("""
                SELECT timestamp, body_state, felt_sense,
                       felt_intensity, energy_level, somatic_valence
                FROM body_states ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_somatic_markers(self, context_hash: str):
        with self.lock:
            return self.conn.execute("""
                SELECT decision, body_state, felt_sense,
                       valence, strength
                FROM somatic_markers WHERE context_hash=?
                ORDER BY strength DESC LIMIT 5
            """, (context_hash,)).fetchall()


# ─── Body State Monitor ───────────────────────────────────────────────────────

class BodyStateMonitor:
    """
    Continuously monitors FORGE's internal operational state.
    Aggregates metrics from all modules into a unified picture.
    This is the insula's raw sensory input.
    """

    def __init__(self):
        self.history: deque = deque(maxlen=50)
        self.rolling_load:    deque = deque(maxlen=10)
        self.rolling_latency: deque = deque(maxlen=10)
        self.rolling_errors:  deque = deque(maxlen=10)

    def update(self, signal: dict) -> InternalMetrics:
        """Extract internal metrics from incoming signal and module state."""
        import hashlib

        # Pipeline load from various sources
        load = signal.get("load", signal.get("pipeline_load", 0.0))
        if not load:
            # Estimate from threat and salience
            threat  = signal.get("threat", 0)
            salience= signal.get("salience_score", 0.3)
            load    = min(1.0, threat * 0.2 + salience * 0.3)

        self.rolling_load.append(load)

        # Latency from pipeline_ms
        latency = signal.get("pipeline_ms", signal.get("avg_latency_ms", 50.0))
        self.rolling_latency.append(latency)

        # Error rate from ACC or direct
        error_rate = signal.get("error_rate", 0.0)
        if not error_rate:
            error_rate = 1.0 if signal.get("error_flag", False) else 0.0
        self.rolling_errors.append(error_rate)

        # Neuromodulator state
        neuro = signal.get("neuro_state", {}) or {}
        profile = neuro.get("profile", {}) if isinstance(neuro, dict) else {}

        # Memory pressure
        memory_pressure = 0.0
        mem_action = signal.get("memory_action","")
        if mem_action == "NEW_MEMORY":     memory_pressure = 0.3
        elif mem_action == "RECONSOLIDATED":memory_pressure = 0.5

        metrics = InternalMetrics(
            pipeline_load  = round(sum(self.rolling_load)/len(self.rolling_load), 3),
            avg_latency_ms = round(sum(self.rolling_latency)/len(self.rolling_latency), 1),
            error_rate     = round(sum(self.rolling_errors)/len(self.rolling_errors), 3),
            memory_pressure= memory_pressure,
            cortisol_level = profile.get("cortisol", signal.get("cortisol", 0.15)),
            dopamine_level = profile.get("dopamine", signal.get("dopamine", 0.45)),
            serotonin_level= profile.get("serotonin", signal.get("serotonin", 0.65)),
            ne_level       = profile.get("norepinephrine", signal.get("ne_level", 0.20)),
            conflict_rate  = signal.get("conflict_rate", 0.0),
            fear_score     = signal.get("fear_score", 0.0),
        )
        self.history.append(metrics)
        return metrics

    def trend(self, attr: str, window: int = 5) -> float:
        """Is this metric trending up (+) or down (-)?"""
        if len(self.history) < 2: return 0.0
        recent = [getattr(m, attr, 0) for m in list(self.history)[-window:]]
        if len(recent) < 2: return 0.0
        return round(recent[-1] - recent[0], 4)


# ─── Felt Sense Generator ─────────────────────────────────────────────────────

class FeltSenseGenerator:
    """
    Translates raw metrics into a FELT SENSE — the subjective
    quality of FORGE's internal experience.

    This is the key step: going from
      "load=0.72, latency=180ms, cortisol=0.65"
    to
      "STRAIN — this feels heavy and uncomfortable"

    The felt sense integrates multiple body signals into
    a single holistic quality of experience.
    """

    def generate(self, metrics: InternalMetrics,
                 energy: float) -> tuple[FeltSense, float, float, float]:
        """
        Returns (felt_sense, intensity, valence, arousal).
        valence: +1=pleasant, -1=unpleasant
        arousal: 0=calm, 1=activated
        """
        load     = metrics.pipeline_load
        cortisol = metrics.cortisol_level
        dopamine = metrics.dopamine_level
        serotonin= metrics.serotonin_level
        ne       = metrics.ne_level
        fear     = metrics.fear_score
        error    = metrics.error_rate

        # Compute overall discomfort
        discomfort = (
            load     * 0.30 +
            cortisol * 0.25 +
            (1 - serotonin) * 0.20 +
            error    * 0.15 +
            fear     * 0.10
        )

        # Compute activation
        arousal = round(min(1.0, ne * 1.5 + fear * 0.5 + load * 0.3), 3)

        # Compute valence
        valence = round(
            dopamine * 0.3 + serotonin * 0.3 - cortisol * 0.2 -
            error * 0.1 - fear * 0.1, 3
        )
        valence = max(-1.0, min(1.0, valence * 2 - 0.5))

        # Map to felt sense
        if discomfort < 0.15 and energy > 0.6:
            felt  = FeltSense.EASE
        elif discomfort < 0.25 and arousal > 0.4:
            felt  = FeltSense.ALERTNESS
        elif discomfort < 0.35:
            felt  = FeltSense.TENSION
        elif discomfort < 0.50:
            felt  = FeltSense.DISCOMFORT
        elif discomfort < 0.65:
            felt  = FeltSense.STRAIN
        elif energy < 0.20:
            felt  = FeltSense.NUMBNESS
        else:
            felt  = FeltSense.PAIN

        intensity = round(min(1.0, discomfort * 1.3), 3)
        return felt, intensity, round(valence, 3), arousal


# ─── Body State Classifier ────────────────────────────────────────────────────

class BodyStateClassifier:
    """Maps metrics to overall body state."""

    def classify(self, metrics: InternalMetrics,
                 energy: float) -> BodyState:
        load     = metrics.pipeline_load
        cortisol = metrics.cortisol_level
        serotonin= metrics.serotonin_level
        dopamine = metrics.dopamine_level
        error    = metrics.error_rate

        if (load < LOAD_COMFORTABLE and cortisol < 0.25 and
                energy > 0.65 and error < ERROR_TOLERABLE):
            return BodyState.THRIVING

        if (load < LOAD_STRAINED and cortisol < 0.45 and
                energy > 0.50):
            return BodyState.COMFORTABLE

        if load > LOAD_CRITICAL or cortisol > 0.75:
            if energy < 0.25:
                return BodyState.DEPLETED
            return BodyState.OVERWHELMED

        if cortisol > 0.55 and serotonin < 0.40 and dopamine < 0.35:
            return BodyState.DEPLETED

        if load > LOAD_STRAINED or cortisol > 0.55 or error > ERROR_UNCOMFORTABLE:
            return BodyState.DISTRESSED

        if load > LOAD_COMFORTABLE or cortisol > 0.35:
            return BodyState.STRAINED

        # Post-stress recovery indicators
        if (cortisol > 0.30 and cortisol < 0.50 and
                load < LOAD_COMFORTABLE):
            return BodyState.RECOVERING

        return BodyState.COMFORTABLE


# ─── Visceroceptive Input ─────────────────────────────────────────────────────

class VisceroceptiveInput:
    """
    The "gut feeling" system — pre-conscious anomaly detection.

    The insula receives direct input from the vagus nerve,
    which monitors gut, heart, and lung state.
    This creates feelings BEFORE conscious processing.

    In FORGE: subtle pattern anomalies that haven't risen to
    salience threshold still generate a visceral signal.
    "Something feels off" before you can say what.
    """

    def __init__(self):
        self.baseline:  dict = {}
        self.history:   deque = deque(maxlen=30)
        self.gut_history:deque = deque(maxlen=20)

    def sense(self, metrics: InternalMetrics,
              signal: dict) -> tuple[VisceroceptiveSignal, str]:
        """
        Generate gut feeling from subtle pattern anomalies.
        Returns (signal_type, description).
        """
        threat  = signal.get("threat", 0)
        anomaly = signal.get("anomaly", False)
        entity  = signal.get("entity_name","")

        # Update baseline
        if len(self.history) >= 5:
            hist_loads = [m.pipeline_load for m in list(self.history)[-5:]]
            baseline_load = sum(hist_loads)/len(hist_loads)
        else:
            baseline_load = 0.35

        self.history.append(metrics)

        # Gut feeling conditions
        gut_signal = VisceroceptiveSignal.NONE
        gut_text   = ""

        # Subtle load increase before threat detected
        if (metrics.pipeline_load > baseline_load * 1.4 and
                threat == 0 and not anomaly):
            gut_signal = VisceroceptiveSignal.UNEASE
            gut_text   = f"load rising unexpectedly ({metrics.pipeline_load:.2f} vs {baseline_load:.2f} baseline)"

        # Cortisol building without clear threat
        elif (metrics.cortisol_level > 0.45 and threat <= 1 and
              metrics.pipeline_load < LOAD_STRAINED):
            gut_signal = VisceroceptiveSignal.GUT_FEELING
            gut_text   = f"cortisol elevated ({metrics.cortisol_level:.2f}) without clear threat"

        # Error pattern before error is fully recognized
        elif (metrics.error_rate > ERROR_TOLERABLE and
              not signal.get("error_flag", False)):
            gut_signal = VisceroceptiveSignal.UNEASE
            gut_text   = f"subtle error pattern building ({metrics.error_rate:.2f})"

        # Visceral threat response
        elif threat >= 3 or (anomaly and metrics.fear_score > 0.5):
            gut_signal = VisceroceptiveSignal.ALARM
            gut_text   = "visceral alarm — body preparing threat response"

        # System rejection — too much for too long
        elif (metrics.cortisol_level > 0.70 and
              metrics.serotonin_level < 0.35 and
              metrics.energy_level < 0.30):
            gut_signal = VisceroceptiveSignal.NAUSEA
            gut_text   = "system approaching rejection — resources critically low"

        self.gut_history.append(gut_signal.value)
        return gut_signal, gut_text


# ─── Interoceptive Predictor ──────────────────────────────────────────────────

class InteroceptivePredictor:
    """
    Predicts WHERE body state is heading.
    The insula doesn't just report current state —
    it looks ahead and generates proactive signals.

    "Load is 0.45 now and climbing — in 3 cycles it will
    exceed comfortable threshold unless something changes."
    """

    def predict(self, history: list[InternalMetrics],
                current: InternalMetrics) -> tuple[BodyState, str]:
        """Predict next body state."""
        if len(history) < 3:
            return BodyState.COMFORTABLE, "insufficient history"

        # Compute trends
        loads     = [m.pipeline_load for m in history[-5:]]
        cortsols  = [m.cortisol_level for m in history[-5:]]
        energies  = [m.energy_level for m in history[-5:]]

        load_trend     = loads[-1] - loads[0] if loads else 0
        cortisol_trend = cortsols[-1] - cortsols[0] if cortsols else 0
        energy_trend   = energies[-1] - energies[0] if energies else 0

        # Project 3 cycles forward
        proj_load     = current.pipeline_load + load_trend * 0.5
        proj_cortisol = current.cortisol_level + cortisol_trend * 0.5
        proj_energy   = current.energy_level + energy_trend * 0.5

        # Classify predicted state
        if proj_load > LOAD_CRITICAL or proj_cortisol > 0.80:
            if proj_energy < 0.20:
                return BodyState.DEPLETED, "energy will be critically low"
            return BodyState.OVERWHELMED, "load trending to critical"

        if proj_load > LOAD_STRAINED or proj_cortisol > 0.55:
            return BodyState.DISTRESSED, "stress indicators climbing"

        if load_trend < -0.05 and cortisol_trend < -0.05:
            return BodyState.RECOVERING, "all indicators improving"

        if (proj_load < LOAD_COMFORTABLE and proj_cortisol < 0.30 and
                proj_energy > 0.60):
            return BodyState.THRIVING, "optimal conditions approaching"

        if proj_energy < 0.35:
            return BodyState.DEPLETED, "energy budget running low"

        return BodyState.COMFORTABLE, "stable within comfortable range"


# ─── Homeostatic Regulator ────────────────────────────────────────────────────

class HomeostaticRegulator:
    """
    Maintains FORGE's internal balance.
    When metrics deviate from targets, generates correction signals.

    Like the body's autonomic nervous system —
    it acts continuously to restore equilibrium.
    """

    def __init__(self, db: InsulaDB):
        self.db      = db
        self.targets = {
            "pipeline_load":  TARGET_LOAD,
            "avg_latency_ms": TARGET_LATENCY,
            "error_rate":     TARGET_ERROR_RATE,
            "cortisol_level": 0.20,
            "serotonin_level":0.65,
            "energy_level":   TARGET_ENERGY,
        }

    def regulate(self, metrics: InternalMetrics,
                 energy: float, cycle: int) -> tuple[HomeostaticStatus, list[dict]]:
        """Check all metrics against targets. Return status + actions."""
        actions      = []
        total_devs   = 0
        severe_devs  = 0

        for metric, target in self.targets.items():
            if metric == "energy_level":
                current = energy
            else:
                current = getattr(metrics, metric, target)

            deviation = current - target
            abs_dev   = abs(deviation)
            rel_dev   = abs_dev / max(target, 0.01)

            if rel_dev > 0.5:  # >50% off target
                severe_devs += 1
                total_devs  += 1
                action = self._correction_action(metric, deviation)
                actions.append({
                    "metric":    metric,
                    "current":   round(current, 3),
                    "target":    target,
                    "deviation": round(deviation, 3),
                    "action":    action,
                    "severity":  "HIGH"
                })
                self.db.log_homeostatic(metric, current, target,
                                         deviation, action, cycle)
            elif rel_dev > 0.25:
                total_devs += 1
                action = self._correction_action(metric, deviation)
                actions.append({
                    "metric":    metric,
                    "current":   round(current, 3),
                    "target":    target,
                    "deviation": round(deviation, 3),
                    "action":    action,
                    "severity":  "MODERATE"
                })

        if severe_devs >= 2:   status = HomeostaticStatus.FAILING
        elif severe_devs == 1: status = HomeostaticStatus.STRUGGLING
        elif total_devs > 0:   status = HomeostaticStatus.COMPENSATING
        else:                  status = HomeostaticStatus.BALANCED

        return status, actions

    def _correction_action(self, metric: str, deviation: float) -> str:
        actions = {
            "pipeline_load":  "reduce_signal_processing" if deviation > 0 else "increase_throughput",
            "avg_latency_ms": "optimize_module_routing" if deviation > 0 else "increase_processing_depth",
            "error_rate":     "engage_error_correction" if deviation > 0 else "maintain",
            "cortisol_level": "activate_calming_protocols" if deviation > 0 else "increase_alertness",
            "serotonin_level":"seek_positive_interactions" if deviation < 0 else "maintain",
            "energy_level":   "reduce_cognitive_load" if deviation < 0 else "maintain",
        }
        return actions.get(metric, "monitor")


# ─── Body Budget Manager ──────────────────────────────────────────────────────

class BodyBudgetManager:
    """
    Manages FORGE's cognitive energy budget.
    Every operation costs energy. Rest and success restore it.

    High load = fast consumption.
    Conflict = expensive.
    Success = small reward.
    Quiet cycles = recovery.

    When energy runs low → FORGE must choose:
    prioritize survival or rest.
    """

    def __init__(self):
        self.energy   = 0.70
        self.history: deque = deque(maxlen=50)
        self.deficit_cycles = 0

    def consume(self, metrics: InternalMetrics,
                had_conflict: bool, success: bool) -> BodyBudgetStatus:
        """Update energy budget based on current cycle's demands."""
        # Consumption
        base_cost  = metrics.pipeline_load * ENERGY_CONSUMPTION_RATE
        conflict_cost = 0.04 if had_conflict else 0.0
        error_cost = metrics.error_rate * 0.02
        cortisol_cost = max(0, metrics.cortisol_level - 0.20) * 0.03

        total_cost = base_cost + conflict_cost + error_cost + cortisol_cost

        # Recovery
        recovery = 0.0
        if metrics.pipeline_load < LOAD_COMFORTABLE:
            recovery += ENERGY_RECOVERY_RATE
        if success:
            recovery += 0.01  # small success bonus
        if metrics.serotonin_level > 0.60:
            recovery += 0.005  # serotonin aids recovery

        # Update energy
        old_energy    = self.energy
        self.energy   = round(max(0.0, min(1.0, self.energy - total_cost + recovery)), 4)
        self.history.append(self.energy)

        overdraft = self.energy < 0.15
        if self.energy < TARGET_ENERGY * 0.5:
            self.deficit_cycles += 1
        else:
            self.deficit_cycles = 0

        return BodyBudgetStatus(
            total_energy     = 1.0,
            available_energy = self.energy,
            consumed_this_cycle = round(total_cost, 4),
            recovery_rate    = round(recovery, 4),
            deficit          = round(max(0, TARGET_ENERGY - self.energy), 4),
            overdraft        = overdraft,
            cycles_in_deficit= self.deficit_cycles
        )


# ─── Somatic Marker System ────────────────────────────────────────────────────

class SomaticMarkerSystem:
    """
    Tags decisions with body state context.
    When FORGE encounters a similar situation later,
    the somatic marker biases the decision with remembered
    body experience.

    "Last time I was in this context, my body felt STRAIN.
    That's relevant to this decision."

    This implements Damasio's Somatic Marker Hypothesis.
    """

    def __init__(self, db: InsulaDB):
        self.db      = db
        self.markers: dict[str, list[SomaticMarker]] = defaultdict(list)

    def tag(self, signal: dict, body_state: BodyStateSnapshot) -> SomaticMarker:
        """Tag this signal+decision with current body state."""
        import hashlib
        ctx_raw  = f"{signal.get('threat',0)}:{signal.get('entity_name','')}:{signal.get('decision','')}"
        ctx_hash = hashlib.md5(ctx_raw.encode()).hexdigest()[:10]

        # Valence: was body state good or bad?
        valence  = body_state.somatic_valence
        strength = body_state.felt_intensity * 0.5 + abs(valence) * 0.5

        marker = SomaticMarker(
            context_hash = ctx_hash,
            decision     = str(signal.get("decision",""))[:30],
            body_state   = body_state.body_state,
            felt_sense   = body_state.felt_sense,
            valence      = round(valence, 3),
            strength     = round(strength, 3)
        )
        self.markers[ctx_hash].append(marker)
        self.db.save_marker(marker)
        return marker

    def retrieve(self, signal: dict) -> list[SomaticMarker]:
        """Retrieve relevant somatic markers for this signal."""
        import hashlib
        ctx_raw  = f"{signal.get('threat',0)}:{signal.get('entity_name','')}:{signal.get('decision','')}"
        ctx_hash = hashlib.md5(ctx_raw.encode()).hexdigest()[:10]
        markers  = self.markers.get(ctx_hash, [])
        for m in markers:
            m.retrieved_count += 1
            self.db.save_marker(m)
        return markers

    def bias_value(self, markers: list[SomaticMarker]) -> float:
        """Net valence bias from somatic markers."""
        if not markers: return 0.0
        weighted = sum(m.valence * m.strength for m in markers)
        total_w  = sum(m.strength for m in markers)
        return round(weighted / max(total_w, 0.01), 3)


# ─── FORGE Insula ─────────────────────────────────────────────────────────────

class ForgeInsula:
    def __init__(self):
        self.db          = InsulaDB()
        self.monitor     = BodyStateMonitor()
        self.felt_gen    = FeltSenseGenerator()
        self.classifier  = BodyStateClassifier()
        self.viscero     = VisceroceptiveInput()
        self.predictor   = InteroceptivePredictor()
        self.homeostasis = HomeostaticRegulator(self.db)
        self.budget      = BodyBudgetManager()
        self.somatic     = SomaticMarkerSystem(self.db)
        self.cycle       = 0
        self.body_history:deque = deque(maxlen=100)

    def sense(self, signal: dict,
              had_conflict: bool = False,
              success: bool = True) -> dict:
        """
        Full interoceptive processing.
        Returns complete body state + all downstream effects.
        """
        self.cycle += 1

        # 1. Update internal metrics
        metrics = self.monitor.update(signal)
        metrics.cycle = self.cycle

        # 2. Update body budget
        budget = self.budget.consume(metrics, had_conflict, success)
        metrics.energy_level = budget.available_energy
        self.db.save_metrics(metrics)

        # 3. Generate felt sense
        felt, intensity, valence, arousal = self.felt_gen.generate(
            metrics, budget.available_energy
        )

        # 4. Classify body state
        body_state = self.classifier.classify(metrics, budget.available_energy)

        # 5. Visceroceptive sensing
        gut_signal, gut_text = self.viscero.sense(metrics, signal)

        # 6. Predict next state
        history_list = list(self.monitor.history)
        pred_state, pred_reason = self.predictor.predict(history_list, metrics)

        # 7. Homeostatic regulation
        homeo_status, homeo_actions = self.homeostasis.regulate(
            metrics, budget.available_energy, self.cycle
        )

        # 8. Build body state snapshot
        snapshot = BodyStateSnapshot(
            body_state      = body_state.value,
            felt_sense      = felt.value,
            felt_intensity  = intensity,
            visceroceptive  = gut_signal.value,
            homeostatic     = homeo_status.value,
            energy_level    = budget.available_energy,
            somatic_valence = valence,
            somatic_arousal = arousal,
            prediction_next = pred_state.value,
            gut_feeling     = gut_text,
            cycle           = self.cycle
        )
        self.db.save_body_state(snapshot)
        self.body_history.append(snapshot)

        # 9. Somatic marker
        marker = self.somatic.tag(signal, snapshot)

        # 10. Retrieve relevant past markers
        past_markers = self.somatic.retrieve(signal)
        somatic_bias = self.somatic.bias_value(past_markers)

        return {
            "cycle":         self.cycle,
            "body_state":    body_state.value,
            "felt_sense":    felt.value,
            "felt_intensity":intensity,
            "visceroceptive":gut_signal.value,
            "gut_feeling":   gut_text,
            "homeostatic":   homeo_status.value,
            "prediction":    pred_state.value,
            "prediction_reason": pred_reason,
            "metrics": {
                "load":       metrics.pipeline_load,
                "latency_ms": metrics.avg_latency_ms,
                "error_rate": metrics.error_rate,
                "cortisol":   metrics.cortisol_level,
                "dopamine":   metrics.dopamine_level,
                "serotonin":  metrics.serotonin_level,
                "ne":         metrics.ne_level,
                "fear":       metrics.fear_score,
            },
            "body_budget": {
                "energy":     budget.available_energy,
                "consumed":   budget.consumed_this_cycle,
                "recovery":   budget.recovery_rate,
                "overdraft":  budget.overdraft,
                "deficit_cycles": budget.cycles_in_deficit,
            },
            "somatic": {
                "valence":    valence,
                "arousal":    arousal,
                "bias":       somatic_bias,
                "past_markers": len(past_markers),
            },
            "homeostatic_actions": homeo_actions[:3],
            "downstream": {
                "prefrontal_bias":   round(somatic_bias * 0.15, 4),
                "salience_modifier": round(arousal * 0.2 + intensity * 0.1, 4),
                "decision_caution":  round(intensity * 0.2, 4),
                "memory_importance": round(intensity * 0.3 + abs(valence) * 0.2, 4),
            }
        }

    def get_status(self) -> dict:
        latest = self.body_history[-1] if self.body_history else None
        return {
            "version":       VERSION,
            "cycle":         self.cycle,
            "body_state":    latest.body_state if latest else "UNKNOWN",
            "felt_sense":    latest.felt_sense if latest else "UNKNOWN",
            "energy":        self.budget.energy,
            "energy_overdraft": self.budget.energy < 0.15,
            "deficit_cycles":self.budget.deficit_cycles,
            "homeostatic":   latest.homeostatic if latest else "UNKNOWN",
            "gut_feeling":   latest.gut_feeling if latest else "",
            "prediction":    latest.prediction_next if latest else "",
        }


# ─── Rich UI ──────────────────────────────────────────────────────────────────

BODY_STATE_COLORS = {
    "THRIVING":    "bright_green",
    "COMFORTABLE": "green",
    "STRAINED":    "yellow",
    "DISTRESSED":  "orange3",
    "OVERWHELMED": "red",
    "DEPLETED":    "bright_red",
    "RECOVERING":  "cyan",
}

FELT_COLORS = {
    "EASE":      "bright_green",
    "ALERTNESS": "cyan",
    "TENSION":   "yellow",
    "DISCOMFORT":"orange3",
    "STRAIN":    "red",
    "PAIN":      "bright_red",
    "NUMBNESS":  "dim",
}

VISCERO_COLORS = {
    "NONE":       "green",
    "GUT_FEELING":"yellow",
    "UNEASE":     "orange3",
    "ALARM":      "red",
    "NAUSEA":     "bright_red",
}

def render_insula(result: dict, label: str, idx: int):
    if not HAS_RICH: return

    bs  = result["body_state"]
    bsc = BODY_STATE_COLORS.get(bs, "white")
    fs  = result["felt_sense"]
    fsc = FELT_COLORS.get(fs, "white")
    vs  = result["visceroceptive"]
    vsc = VISCERO_COLORS.get(vs, "white")
    energy = result["body_budget"]["energy"]
    ec  = "bright_green" if energy>0.6 else "yellow" if energy>0.35 else "red"

    console.print(Rule(
        f"[bold cyan]⬡ INSULA[/bold cyan]  [dim]#{idx}[/dim]  "
        f"[{bsc}]{bs}[/{bsc}]  "
        f"[{fsc}]{fs}[/{fsc}]  "
        f"energy=[{ec}]{energy:.3f}[/{ec}]"
    ))

    # Left: body state + felt sense
    met  = result["metrics"]
    fi   = result["felt_intensity"]
    left_lines = [
        f"[bold]Body State:[/bold]  [{bsc}]{bs}[/{bsc}]",
        f"[bold]Felt Sense:[/bold]  [{fsc}]{fs}[/{fsc}] ({fi:.2f})",
        f"[bold]Valence:[/bold]     {result['somatic']['valence']:+.3f}",
        f"[bold]Arousal:[/bold]     {result['somatic']['arousal']:.3f}",
        f"",
        f"[bold dim]── Vitals ──[/bold dim]",
        f"Load:      {'█'*int(met['load']*10)}{'░'*(10-int(met['load']*10))} {met['load']:.3f}",
        f"Latency:   {met['latency_ms']:.1f}ms",
        f"Cortisol:  {'█'*int(met['cortisol']*10)}{'░'*(10-int(met['cortisol']*10))} {met['cortisol']:.3f}",
        f"Serotonin: {'█'*int(met['serotonin']*10)}{'░'*(10-int(met['serotonin']*10))} {met['serotonin']:.3f}",
        f"Dopamine:  {'█'*int(met['dopamine']*10)}{'░'*(10-int(met['dopamine']*10))} {met['dopamine']:.3f}",
    ]

    # Right: visceroceptive + budget + prediction
    budget = result["body_budget"]
    ec2    = "green" if budget["energy"]>0.6 else "yellow" if budget["energy"]>0.35 else "red"
    pred   = result["prediction"]
    pbc    = BODY_STATE_COLORS.get(pred,"white")

    right_lines = [
        f"[bold]Gut feeling:[/bold]  [{vsc}]{vs}[/{vsc}]",
    ]
    if result.get("gut_feeling"):
        right_lines.append(f"[{vsc}][dim]{result['gut_feeling'][:45]}[/dim][/{vsc}]")

    right_lines += [
        f"",
        f"[bold dim]── Body Budget ──[/bold dim]",
        f"Energy:   [{ec2}]{'█'*int(budget['energy']*10)}{'░'*(10-int(budget['energy']*10))} {budget['energy']:.3f}[/{ec2}]",
        f"Consumed: {budget['consumed']:.4f}/cycle",
        f"Recovery: +{budget['recovery']:.4f}/cycle",
        f"{'[red]⚠ OVERDRAFT[/red]' if budget['overdraft'] else '[dim]Budget OK[/dim]'}",
        f"",
        f"[bold dim]── Prediction ──[/bold dim]",
        f"Next:     [{pbc}]{pred}[/{pbc}]",
        f"[dim]{result['prediction_reason'][:40]}[/dim]",
        f"Homeost:  {result['homeostatic']}",
    ]

    console.print(Columns([
        Panel("\n".join(left_lines), title=f"[bold {bsc}]Interoception[/bold {bsc}]", border_style=bsc),
        Panel("\n".join(right_lines),title="[bold]Body Budget + Gut[/bold]",           border_style=vsc)
    ]))

    # Downstream effects
    ds = result["downstream"]
    ha = result["homeostatic_actions"]
    if ha:
        actions_str = "  ".join(
            f"[dim]{a['metric']}[/dim]→[cyan]{a['action']}[/cyan]"
            for a in ha[:2]
        )
        console.print(f"  [dim]Homeostatic: {actions_str}[/dim]")
    console.print(
        f"  [dim]Downstream → prefrontal bias: {ds['prefrontal_bias']:+.4f}  "
        f"sal modifier: {ds['salience_modifier']:.4f}  "
        f"mem importance: {ds['memory_importance']:.4f}[/dim]"
    )


def run_demo():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FORGE INSULA[/bold cyan]\n"
            "[dim]Interoception · Body State · Gut Feelings · Somatic Markers[/dim]\n"
            f"[dim]Version {VERSION}[/dim]",
            border_style="cyan"
        ))

    insula = ForgeInsula()

    scenarios = [
        # Calm baseline
        ({"threat":0,"anomaly":False,"entity_name":"alice_tech",
          "salience_score":0.2,"pipeline_ms":180,"error_rate":0.0,
          "decision":"MONITOR","memory_action":"NEW_MEMORY",
          "fear_score":0.0,"conflict_rate":0.0,
          "neuro_state":{"profile":{"cortisol":0.15,"dopamine":0.55,
          "serotonin":0.68,"norepinephrine":0.18}}},
         False, True, "Calm baseline — thriving"),

        # Load building
        ({"threat":1,"anomaly":False,"entity_name":"unknown_x",
          "salience_score":0.45,"pipeline_ms":240,"error_rate":0.1,
          "decision":"ALERT","memory_action":"NEW_MEMORY",
          "fear_score":0.2,"conflict_rate":0.3,
          "neuro_state":{"profile":{"cortisol":0.35,"dopamine":0.48,
          "serotonin":0.60,"norepinephrine":0.35}}},
         True, True, "Load building — gut feeling rising"),

        # Crisis — cortisol spiked
        ({"threat":4,"anomaly":True,"entity_name":"unknown_x",
          "salience_score":0.95,"pipeline_ms":320,"error_rate":0.0,
          "decision":"EMERGENCY_BLOCK","memory_action":"NEW_MEMORY",
          "fear_score":1.0,"conflict_rate":0.8,
          "neuro_state":{"profile":{"cortisol":0.82,"dopamine":0.91,
          "serotonin":0.42,"norepinephrine":0.94}}},
         True, True, "Crisis — overwhelmed, visceral alarm"),

        # Sustained crisis — energy depleting
        ({"threat":3,"anomaly":True,"entity_name":"unknown_x",
          "salience_score":0.85,"pipeline_ms":380,"error_rate":0.2,
          "decision":"BLOCK","memory_action":"RECONSOLIDATED",
          "fear_score":0.85,"conflict_rate":0.7,
          "neuro_state":{"profile":{"cortisol":0.88,"dopamine":0.85,
          "serotonin":0.32,"norepinephrine":0.88}}},
         True, True, "Sustained crisis — body budget draining"),

        # Partial recovery
        ({"threat":1,"anomaly":False,"entity_name":"security_team",
          "salience_score":0.35,"pipeline_ms":260,"error_rate":0.05,
          "decision":"MONITOR","memory_action":"NEW_MEMORY",
          "fear_score":0.3,"conflict_rate":0.2,
          "neuro_state":{"profile":{"cortisol":0.55,"dopamine":0.72,
          "serotonin":0.45,"norepinephrine":0.42}}},
         False, True, "Partial recovery — cortisol clearing"),

        # Full recovery
        ({"threat":0,"anomaly":False,"entity_name":"alice_tech",
          "salience_score":0.2,"pipeline_ms":190,"error_rate":0.0,
          "decision":"COLLABORATE","memory_action":"NEW_MEMORY",
          "fear_score":0.05,"conflict_rate":0.0,
          "neuro_state":{"profile":{"cortisol":0.22,"dopamine":0.60,
          "serotonin":0.58,"norepinephrine":0.22}}},
         False, True, "Recovery — body rebuilding"),
    ]

    for i, (sig, conflict, success, label) in enumerate(scenarios):
        if HAS_RICH:
            console.print(f"\n[bold dim]━━━ {i+1}: {label.upper()} ━━━[/bold dim]")
        result = insula.sense(sig, conflict, success)
        render_insula(result, label, i+1)
        time.sleep(0.1)

    # Final status
    if HAS_RICH:
        console.print(Rule("[bold cyan]⬡ INSULA FINAL STATUS[/bold cyan]"))
        status = insula.get_status()

        st = Table(box=box.DOUBLE_EDGE, border_style="cyan", title="Insula Status")
        st.add_column("Metric", style="cyan")
        st.add_column("Value",  style="white")
        bsc = BODY_STATE_COLORS.get(status["body_state"],"white")
        st.add_row("Cycles",        str(status["cycle"]))
        st.add_row("Body State",    status["body_state"])
        st.add_row("Felt Sense",    status["felt_sense"])
        st.add_row("Energy",        f"{status['energy']:.3f}")
        st.add_row("Overdraft",     str(status["energy_overdraft"]))
        st.add_row("Homeostatic",   status["homeostatic"])
        st.add_row("Prediction",    status["prediction"])
        if status["gut_feeling"]:
            st.add_row("Gut Feeling", status["gut_feeling"][:50])
        console.print(st)

        # Body state history
        rows = insula.db.get_body_states(6)
        if rows:
            ht = Table(box=box.SIMPLE, title="Body State History", title_style="dim")
            ht.add_column("Time",     width=10)
            ht.add_column("State",    width=14)
            ht.add_column("Felt",     width=12)
            ht.add_column("Intensity",justify="right", width=10)
            ht.add_column("Energy",   justify="right", width=8)
            for row in rows:
                ts, bs, fs, fi, el, sv = row
                bsc2 = BODY_STATE_COLORS.get(bs,"white")
                fsc2 = FELT_COLORS.get(fs,"white")
                ht.add_row(
                    ts[11:19],
                    f"[{bsc2}]{bs}[/{bsc2}]",
                    f"[{fsc2}]{fs}[/{fsc2}]",
                    f"{fi:.2f}",
                    f"{el:.3f}"
                )
            console.print(ht)


# ─── HTTP API ─────────────────────────────────────────────────────────────────

def run_api(insula: ForgeInsula):
    if not HAS_FLASK: return
    app = Flask(__name__)

    @app.route("/sense", methods=["POST"])
    def sense():
        data = request.json or {}
        return jsonify(insula.sense(
            data.get("signal",{}),
            data.get("had_conflict", False),
            data.get("success", True)
        ))

    @app.route("/body_state", methods=["GET"])
    def body_state():
        latest = insula.body_history[-1] if insula.body_history else None
        if not latest:
            return jsonify({"status": "no_data"})
        return jsonify({
            "body_state":   latest.body_state,
            "felt_sense":   latest.felt_sense,
            "energy":       insula.budget.energy,
            "gut_feeling":  latest.gut_feeling,
            "prediction":   latest.prediction_next,
        })

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(insula.get_status())

    @app.route("/history", methods=["GET"])
    def history():
        rows = insula.db.get_body_states(20)
        return jsonify([{
            "timestamp":r[0],"body_state":r[1],"felt_sense":r[2],
            "intensity":r[3],"energy":r[4],"valence":r[5]
        } for r in rows])

    app.run(host="0.0.0.0", port=API_PORT, debug=False)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    insula = ForgeInsula()
    if "--api" in sys.argv:
        t = threading.Thread(target=run_api, args=(insula,), daemon=True)
        t.start()
    run_demo()
