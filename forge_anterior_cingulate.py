"""
FORGE Anterior Cingulate Cortex — forge_anterior_cingulate.py
===============================================================
AI analog of the brain's anterior cingulate cortex (ACC).

The ACC is the brain's conflict detector and error monitor —
a watchdog sitting between emotion and cognition, constantly
asking: "Are the signals agreeing? Is something going wrong?"

It receives input from virtually every brain region and has
direct projections back to motor cortex, prefrontal, and
subcortical systems. It doesn't make decisions — it raises
flags when decisions are at risk of being wrong.

Key insight: The ACC doesn't know what the right answer is.
It only knows when the system is UNCERTAIN or CONFLICTED.
That signal alone is enough to redirect cognitive resources.

Four core functions:

  1. CONFLICT DETECTION (module disagreement monitor)
     When forge_amygdala says ALARM but forge_prefrontal says
     SAFE — that's conflict. The ACC fires, halts action,
     and routes the signal to deeper deliberation.
     The stronger the disagreement, the louder the flag.

  2. ERROR MONITORING (post-decision mismatch)
     Expected outcome vs actual outcome.
     If forge_conscious predicted X but got Y — ACC fires.
     This is the neural basis of the "oops" feeling.
     Generates an Error-Related Negativity (ERN) signal.

  3. PERFORMANCE MONITORING (degradation detection)
     Tracks rolling accuracy / coherence of the whole system.
     If recent cycles are producing low-confidence outputs,
     ACC escalates arousal to force recalibration.

  4. PAIN/SALIENCE GATING
     The ACC processes social pain and physical pain equally.
     Exclusion, rejection, loss — these activate the same
     ACC circuits as physical injury. This module uses that
     property to flag when inputs carry "social threat" even
     when explicit threat scores are low.

Architecture:
  ConflictDetector     → multi-module disagreement scanner
  ErrorMonitor         → expected vs actual outcome tracker
  PerformanceTracker   → rolling system health index
  SalienceGate         → social/pain signal amplifier
  ACCOutput            → routes flags to metacognition + prefrontal
  ResolutionEngine     → suggests how to resolve detected conflicts
"""

import json
import time
import uuid
import sqlite3
import threading
import math
from datetime import datetime
from collections import deque
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

DB_PATH  = "forge_anterior_cingulate.db"
API_PORT = 7798
VERSION  = "1.0.0"

# Conflict thresholds
CONFLICT_LOW       = 0.20   # minor disagreement — log only
CONFLICT_MODERATE  = 0.40   # moderate — flag to metacognition
CONFLICT_HIGH      = 0.65   # high — halt and re-evaluate
CONFLICT_CRITICAL  = 0.85   # critical — override and escalate

# Error monitoring
ERROR_DECAY        = 0.08   # how fast error signal fades
ERROR_SPIKE        = 0.45   # how much a mismatch spikes the ERN

# Performance tracking
PERF_WINDOW        = 20     # rolling window size (cycles)
PERF_ALARM_THRESH  = 0.45   # below this → performance alarm

# Social pain amplification
SOCIAL_PAIN_BOOST  = 0.30   # social threat adds to conflict score

console = Console() if HAS_RICH else None

# ─── Enums ────────────────────────────────────────────────────────────────────

class ConflictLevel(Enum):
    NONE     = "NONE"
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

class ErrorType(Enum):
    OUTCOME_MISMATCH  = "OUTCOME_MISMATCH"   # predicted X, got Y
    CONFIDENCE_CRASH  = "CONFIDENCE_CRASH"   # module confidence dropped suddenly
    MODULE_DROPOUT    = "MODULE_DROPOUT"     # expected module went silent
    LOOP_DETECTED     = "LOOP_DETECTED"      # same conflict recurring
    SOCIAL_VIOLATION  = "SOCIAL_VIOLATION"   # social norm unexpectedly broken

class ResolutionStrategy(Enum):
    DEFER_TO_PREFRONTAL  = "DEFER_TO_PREFRONTAL"   # let reason decide
    DEFER_TO_AMYGDALA    = "DEFER_TO_AMYGDALA"     # let instinct decide
    PAUSE_AND_GATHER     = "PAUSE_AND_GATHER"       # collect more signal
    ESCALATE_AROUSAL     = "ESCALATE_AROUSAL"       # boost attention/NE
    INHIBIT_ACTION       = "INHIBIT_ACTION"         # freeze until resolved
    NOTIFY_METACOGNITION = "NOTIFY_METACOGNITION"   # surface to self-awareness

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class ConflictEvent:
    """A detected conflict between modules or signals."""
    id:             str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:      str   = field(default_factory=lambda: datetime.now().isoformat())
    modules:        list  = field(default_factory=list)   # conflicting modules
    conflict_score: float = 0.0
    level:          str   = ConflictLevel.NONE.value
    description:    str   = ""
    resolution:     str   = ResolutionStrategy.PAUSE_AND_GATHER.value
    resolved:       bool  = False
    cycles_open:    int   = 0

@dataclass
class ErrorSignal:
    """An error-related negativity event."""
    id:             str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:      str   = field(default_factory=lambda: datetime.now().isoformat())
    error_type:     str   = ErrorType.OUTCOME_MISMATCH.value
    predicted:      str   = ""
    actual:         str   = ""
    ern_magnitude:  float = 0.0   # Error-Related Negativity strength
    source_module:  str   = ""
    corrected:      bool  = False

@dataclass
class PerformanceSample:
    """A single system performance snapshot."""
    timestamp:      str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:          int   = 0
    confidence:     float = 0.0
    conflict_score: float = 0.0
    error_rate:     float = 0.0
    module_count:   int   = 0
    health_score:   float = 0.0

# ─── Database ─────────────────────────────────────────────────────────────────

class ACCDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS conflicts (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    modules TEXT, conflict_score REAL,
                    level TEXT, description TEXT,
                    resolution TEXT, resolved INTEGER,
                    cycles_open INTEGER
                );
                CREATE TABLE IF NOT EXISTS error_signals (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    error_type TEXT, predicted TEXT,
                    actual TEXT, ern_magnitude REAL,
                    source_module TEXT, corrected INTEGER
                );
                CREATE TABLE IF NOT EXISTS performance_log (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    cycle INTEGER, confidence REAL,
                    conflict_score REAL, error_rate REAL,
                    module_count INTEGER, health_score REAL
                );
                CREATE TABLE IF NOT EXISTS resolution_log (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    conflict_id TEXT, strategy TEXT,
                    outcome TEXT, cycles_to_resolve INTEGER
                );
            """)
            self.conn.commit()

    def save_conflict(self, c: ConflictEvent):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO conflicts VALUES (?,?,?,?,?,?,?,?,?)
            """, (c.id, c.timestamp, json.dumps(c.modules),
                  c.conflict_score, c.level, c.description,
                  c.resolution, int(c.resolved), c.cycles_open))
            self.conn.commit()

    def save_error(self, e: ErrorSignal):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO error_signals VALUES (?,?,?,?,?,?,?,?)
            """, (e.id, e.timestamp, e.error_type, e.predicted,
                  e.actual, e.ern_magnitude, e.source_module,
                  int(e.corrected)))
            self.conn.commit()

    def save_performance(self, p: PerformanceSample):
        with self.lock:
            self.conn.execute("""
                INSERT INTO performance_log VALUES (?,?,?,?,?,?,?,?)
            """, (str(uuid.uuid4())[:8], p.timestamp, p.cycle,
                  p.confidence, p.conflict_score, p.error_rate,
                  p.module_count, p.health_score))
            self.conn.commit()

    def log_resolution(self, conflict_id: str, strategy: str,
                       outcome: str, cycles: int):
        with self.lock:
            self.conn.execute("""
                INSERT INTO resolution_log VALUES (?,?,?,?,?,?)
            """, (str(uuid.uuid4())[:8], datetime.now().isoformat(),
                  conflict_id, strategy, outcome, cycles))
            self.conn.commit()

    def get_recent_conflicts(self, limit=15):
        with self.lock:
            return self.conn.execute("""
                SELECT timestamp, modules, conflict_score, level,
                       description, resolution, resolved
                FROM conflicts ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_recent_errors(self, limit=10):
        with self.lock:
            return self.conn.execute("""
                SELECT timestamp, error_type, predicted, actual,
                       ern_magnitude, source_module
                FROM error_signals ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_health_trend(self, limit=20):
        with self.lock:
            return self.conn.execute("""
                SELECT cycle, health_score, conflict_score, error_rate
                FROM performance_log ORDER BY cycle DESC LIMIT ?
            """, (limit,)).fetchall()


# ─── Conflict Detector ────────────────────────────────────────────────────────

class ConflictDetector:
    """
    Scans module outputs for disagreement.

    The ACC doesn't know which module is right.
    It only knows they disagree — and that disagreement
    itself is a signal requiring attention.

    Conflict types detected:
      VALENCE CONFLICT   — one module says good, another says bad
      CONFIDENCE SPREAD  — modules have wildly different certainty
      PRIORITY CONFLICT  — multiple modules claim urgent attention
      TEMPORAL CONFLICT  — current signal contradicts recent history
    """

    def detect(self, module_outputs: dict) -> tuple[float, str, list]:
        """
        Returns (conflict_score, description, conflicting_modules).
        module_outputs: {module_name: {"valence": float, "confidence": float, ...}}
        """
        conflict_score = 0.0
        conflicts_found = []
        description_parts = []

        if len(module_outputs) < 2:
            return 0.0, "insufficient modules", []

        # ── VALENCE CONFLICT ─────────────────────────────
        valences = {
            k: v.get("valence", 0.0)
            for k, v in module_outputs.items()
            if "valence" in v
        }
        if len(valences) >= 2:
            vals = list(valences.values())
            valence_spread = max(vals) - min(vals)
            if valence_spread > 0.3:
                conflict_score = max(conflict_score, valence_spread * 0.8)
                conflicting_modules = [
                    k for k, v in valences.items()
                    if v == max(vals) or v == min(vals)
                ]
                conflicts_found.extend(conflicting_modules)
                description_parts.append(
                    f"valence spread={valence_spread:.2f} "
                    f"({'+'.join(conflicting_modules[:2])})"
                )

        # ── CONFIDENCE SPREAD ────────────────────────────
        confidences = {
            k: v.get("confidence", 0.5)
            for k, v in module_outputs.items()
            if "confidence" in v
        }
        if len(confidences) >= 2:
            c_vals = list(confidences.values())
            conf_spread = max(c_vals) - min(c_vals)
            if conf_spread > 0.5:
                conflict_score = max(conflict_score, conf_spread * 0.5)
                low_conf = [k for k, v in confidences.items() if v == min(c_vals)]
                conflicts_found.extend(low_conf)
                description_parts.append(f"confidence spread={conf_spread:.2f}")

        # ── PRIORITY CONFLICT ────────────────────────────
        priorities = {
            k: v.get("priority", 0)
            for k, v in module_outputs.items()
            if "priority" in v
        }
        high_priority = [k for k, v in priorities.items() if v >= 3]
        if len(high_priority) >= 3:
            conflict_score = max(conflict_score, min(1.0, len(high_priority) * 0.15))
            conflicts_found.extend(high_priority[:3])
            description_parts.append(f"{len(high_priority)} modules claiming high priority")

        # ── THREAT VS SAFETY CONFLICT ────────────────────
        threat_score  = module_outputs.get("amygdala", {}).get("fear_score", 0.0)
        safety_score  = module_outputs.get("amygdala", {}).get("safety_score", 0.0)
        prefrontal_ok = module_outputs.get("prefrontal", {}).get("valence", 0.5)

        if threat_score > 0.5 and prefrontal_ok > 0.6:
            clash = (threat_score + prefrontal_ok - 1.0)
            conflict_score = max(conflict_score, min(1.0, clash * 1.2))
            conflicts_found.extend(["amygdala", "prefrontal"])
            description_parts.append(
                f"amygdala fear={threat_score:.2f} vs prefrontal safe={prefrontal_ok:.2f}"
            )

        description = " | ".join(description_parts) if description_parts else "none"
        return round(conflict_score, 4), description, list(set(conflicts_found))

    def score_to_level(self, score: float) -> ConflictLevel:
        if score >= CONFLICT_CRITICAL: return ConflictLevel.CRITICAL
        if score >= CONFLICT_HIGH:     return ConflictLevel.HIGH
        if score >= CONFLICT_MODERATE: return ConflictLevel.MODERATE
        if score >= CONFLICT_LOW:      return ConflictLevel.LOW
        return ConflictLevel.NONE


# ─── Error Monitor ────────────────────────────────────────────────────────────

class ErrorMonitor:
    """
    Tracks prediction errors — when the system expected X and got Y.

    The brain's ACC generates a distinctive EEG signal ~100ms after
    an error: the Error-Related Negativity (ERN). It's involuntary,
    automatic, and present even when the person doesn't consciously
    notice the mistake.

    This module models that mechanism:
    - Store predictions before action
    - Compare with actual outcomes after action
    - Generate ERN signal proportional to mismatch
    - Feed ERN to metacognition for self-correction
    """

    def __init__(self, db: ACCDB):
        self.db         = db
        self.pending:   dict[str, str] = {}    # prediction_id → expected outcome
        self.ern_signal: float         = 0.0   # current ERN level (decays)
        self.total_errors = 0
        self.total_corrections = 0

    def register_prediction(self, prediction_id: str, expected: str):
        """Store a prediction before the outcome is known."""
        self.pending[prediction_id] = expected

    def evaluate(self, prediction_id: str, actual: str,
                 source_module: str) -> Optional[ErrorSignal]:
        """
        Compare prediction with actual outcome.
        Returns ErrorSignal if mismatch detected.
        """
        expected = self.pending.pop(prediction_id, None)
        if expected is None:
            return None

        # Measure mismatch
        if expected == actual:
            return None  # correct prediction — no ERN

        # Mismatch detected
        mismatch_severity = self._severity(expected, actual)
        self.ern_signal = min(1.0, self.ern_signal + ERROR_SPIKE * mismatch_severity)
        self.total_errors += 1

        err = ErrorSignal(
            error_type    = ErrorType.OUTCOME_MISMATCH.value,
            predicted     = expected,
            actual        = actual,
            ern_magnitude = round(self.ern_signal, 4),
            source_module = source_module
        )
        self.db.save_error(err)
        return err

    def flag_confidence_crash(self, module: str,
                               conf_before: float, conf_after: float) -> Optional[ErrorSignal]:
        """Flag when a module's confidence suddenly drops."""
        if conf_before - conf_after < 0.40:
            return None

        self.ern_signal = min(1.0, self.ern_signal + ERROR_SPIKE * 0.5)
        self.total_errors += 1

        err = ErrorSignal(
            error_type    = ErrorType.CONFIDENCE_CRASH.value,
            predicted     = f"conf≈{conf_before:.2f}",
            actual        = f"conf={conf_after:.2f}",
            ern_magnitude = round(self.ern_signal, 4),
            source_module = module
        )
        self.db.save_error(err)
        return err

    def decay(self):
        """ERN signal decays over time (called each cycle)."""
        self.ern_signal = round(max(0.0, self.ern_signal - ERROR_DECAY), 4)

    def _severity(self, expected: str, actual: str) -> float:
        """Estimate how severe the mismatch is."""
        opposites = {
            ("SAFE","ALARM"),("ALARM","SAFE"),
            ("POSITIVE","NEGATIVE"),("NEGATIVE","POSITIVE"),
            ("LOW","HIGH"),("HIGH","LOW"),
            ("ACCEPT","REJECT"),("REJECT","ACCEPT"),
        }
        pair = (expected.upper(), actual.upper())
        if pair in opposites or tuple(reversed(pair)) in opposites:
            return 1.0   # complete reversal
        if expected.upper() == actual.upper():
            return 0.0
        return 0.5       # partial mismatch


# ─── Performance Tracker ─────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Monitors rolling system health.

    The ACC maintains a constant background estimate of:
    "Is the system performing well overall?"

    When rolling performance drops, ACC escalates arousal
    (via NE projections) to force the rest of the brain
    to work harder / more carefully.

    This is why difficult tasks feel effortful —
    the ACC is continuously signaling "not good enough yet."
    """

    def __init__(self, db: ACCDB, window=PERF_WINDOW):
        self.db      = db
        self.window  = window
        self.history: deque = deque(maxlen=window)
        self.cycle   = 0

    def record(self, confidence: float, conflict_score: float,
               error_rate: float, module_count: int) -> PerformanceSample:
        self.cycle += 1

        health = self._compute_health(confidence, conflict_score, error_rate)
        sample = PerformanceSample(
            cycle          = self.cycle,
            confidence     = round(confidence, 4),
            conflict_score = round(conflict_score, 4),
            error_rate     = round(error_rate, 4),
            module_count   = module_count,
            health_score   = round(health, 4)
        )
        self.history.append(sample)
        self.db.save_performance(sample)
        return sample

    def _compute_health(self, confidence: float,
                         conflict: float, error_rate: float) -> float:
        """
        Health = weighted combination of:
         - confidence (positive)
         - conflict (negative)
         - error rate (negative)
        """
        health = (
            confidence * 0.50
            - conflict * 0.30
            - error_rate * 0.20
        )
        return max(0.0, min(1.0, health))

    def rolling_health(self) -> float:
        if not self.history: return 0.5
        return round(sum(s.health_score for s in self.history) / len(self.history), 4)

    def trend(self) -> str:
        """Is system health improving, stable, or declining?"""
        if len(self.history) < 4: return "UNKNOWN"
        recent = [s.health_score for s in list(self.history)[-4:]]
        delta  = recent[-1] - recent[0]
        if delta > 0.08:  return "IMPROVING"
        if delta < -0.08: return "DECLINING"
        return "STABLE"

    def alarm_active(self) -> bool:
        return self.rolling_health() < PERF_ALARM_THRESH


# ─── Salience Gate ────────────────────────────────────────────────────────────

class SalienceGate:
    """
    The ACC processes social pain identically to physical pain.
    Social exclusion, rejection, betrayal — all activate
    the dorsal ACC (dACC) the same way a burn does.

    This gate amplifies conflict scores when social threat
    signals are present, even when explicit threat levels are low.

    Why this matters in FORGE:
    Social signals that "feel wrong" — a sudden change in
    communication style, unexpected silence, anomalous trust
    changes — deserve elevated conflict flags even when
    numerical threat scores remain low.
    """

    SOCIAL_PAIN_CUES = [
        "exclusion", "rejection", "betrayal", "silence",
        "withdrawal", "dismissal", "isolation", "ignored"
    ]

    def amplify(self, signal: dict, base_conflict: float) -> tuple[float, str]:
        """
        Returns (amplified_conflict, reason).
        Boosts conflict if social pain cues detected.
        """
        social = signal.get("social", {}) or {}
        reason = ""

        # Trust collapse
        trust   = social.get("trust_score", 1.0)
        if trust < 0.2:
            boost  = SOCIAL_PAIN_BOOST * (1.0 - trust)
            base_conflict = min(1.0, base_conflict + boost)
            reason = f"trust_collapse={trust:.2f}"

        # Intent check for social pain cues
        intent = social.get("inferred_intent", "").lower()
        for cue in self.SOCIAL_PAIN_CUES:
            if cue in intent:
                base_conflict = min(1.0, base_conflict + SOCIAL_PAIN_BOOST * 0.5)
                reason = f"social_pain_cue={cue}"
                break

        # Sudden communication change (anomaly + social)
        if signal.get("anomaly", False) and social:
            base_conflict = min(1.0, base_conflict + 0.10)
            reason += " + anomaly"

        return round(base_conflict, 4), reason.strip()


# ─── Resolution Engine ────────────────────────────────────────────────────────

class ResolutionEngine:
    """
    Given a conflict, the ACC suggests how to resolve it.

    The ACC doesn't resolve conflicts itself — it routes them
    to the appropriate downstream system. This is its primary
    output function.

    Resolution choice depends on:
      - Conflict level (how bad)
      - Time pressure (how urgent)
      - Which modules are in conflict (emotion vs reason)
      - System health (can we afford to pause?)
    """

    def suggest(self, conflict: ConflictEvent,
                error_signal: Optional[ErrorSignal],
                health: float,
                hijack_active: bool) -> ResolutionStrategy:
        """Select the best resolution strategy."""

        level = ConflictLevel(conflict.level)

        # Hijack active — everything defers to amygdala
        if hijack_active:
            return ResolutionStrategy.DEFER_TO_AMYGDALA

        # Critical conflict — inhibit action entirely
        if level == ConflictLevel.CRITICAL:
            return ResolutionStrategy.INHIBIT_ACTION

        # ERN spike — surface to metacognition
        if error_signal and error_signal.ern_magnitude > 0.7:
            return ResolutionStrategy.NOTIFY_METACOGNITION

        # System health poor — boost arousal
        if health < PERF_ALARM_THRESH:
            return ResolutionStrategy.ESCALATE_AROUSAL

        # Amygdala vs prefrontal conflict — defer to reason unless urgent
        modules = conflict.modules
        if "amygdala" in modules and "prefrontal" in modules:
            if conflict.conflict_score < CONFLICT_HIGH:
                return ResolutionStrategy.DEFER_TO_PREFRONTAL
            else:
                return ResolutionStrategy.PAUSE_AND_GATHER

        # High conflict, enough health — gather more signal
        if level in [ConflictLevel.HIGH, ConflictLevel.MODERATE]:
            return ResolutionStrategy.PAUSE_AND_GATHER

        # Low conflict — just notify metacognition quietly
        return ResolutionStrategy.NOTIFY_METACOGNITION

    def urgency(self, level: ConflictLevel, health: float) -> float:
        """
        Compute ACC urgency signal (0-1).
        Urgency drives NE release in neuromodulator.
        """
        base = {
            ConflictLevel.NONE:     0.0,
            ConflictLevel.LOW:      0.15,
            ConflictLevel.MODERATE: 0.40,
            ConflictLevel.HIGH:     0.70,
            ConflictLevel.CRITICAL: 1.0,
        }.get(level, 0.0)

        # Low health amplifies urgency
        health_factor = 1.0 + (1.0 - health) * 0.3
        return round(min(1.0, base * health_factor), 4)


# ─── ACC Output ──────────────────────────────────────────────────────────────

class ACCOutput:
    """
    Translates ACC activation into downstream module signals.

    ACC projects to:
      - forge_metacognition  → surface conflict to awareness
      - forge_prefrontal     → request deliberate reasoning
      - forge_neuromodulator → modulate NE / arousal
      - forge_thalamus       → adjust attention gating
      - forge_conscious      → interrupt current processing
    """

    def compute(self, conflict: ConflictEvent,
                error: Optional[ErrorSignal],
                health: float,
                strategy: ResolutionStrategy,
                urgency: float) -> dict:

        # NE signal to neuromodulator
        ne_signal = round(urgency * 0.6, 4)

        # Attention redirect signal to thalamus
        attn_redirect = round(urgency * 0.5, 4) if conflict.level != ConflictLevel.NONE.value else 0.0

        # Interrupt signal to conscious
        interrupt = conflict.level in [ConflictLevel.HIGH.value, ConflictLevel.CRITICAL.value]

        # ERN magnitude for metacognition
        ern = error.ern_magnitude if error else 0.0

        # Prefrontal request flag
        needs_reasoning = strategy in [
            ResolutionStrategy.DEFER_TO_PREFRONTAL,
            ResolutionStrategy.PAUSE_AND_GATHER
        ]

        return {
            "ne_signal":          ne_signal,
            "attention_redirect": attn_redirect,
            "interrupt_conscious":interrupt,
            "ern_magnitude":      round(ern, 4),
            "needs_reasoning":    needs_reasoning,
            "strategy":           strategy.value,
            "urgency":            urgency,
            "health_index":       round(health, 4),
            "action_suppressed":  strategy == ResolutionStrategy.INHIBIT_ACTION,
        }


# ─── FORGE Anterior Cingulate ─────────────────────────────────────────────────

class ForgeAnteriorCingulate:
    def __init__(self):
        self.db          = ACCDB()
        self.detector    = ConflictDetector()
        self.error_mon   = ErrorMonitor(self.db)
        self.perf        = PerformanceTracker(self.db)
        self.salience    = SalienceGate()
        self.resolver    = ResolutionEngine()
        self.output_calc = ACCOutput()
        self.cycle       = 0

        # Open conflicts (not yet resolved)
        self.open_conflicts: list[ConflictEvent] = []

        # Running counters
        self.total_conflicts  = 0
        self.total_errors     = 0
        self.total_inhibitions= 0
        self.total_escalations= 0

    def process(self, signal: dict,
                module_outputs: Optional[dict] = None,
                hijack_active: bool = False) -> dict:
        """
        Full ACC processing pipeline.

        signal:         raw input signal (same format as other FORGE modules)
        module_outputs: dict of {module_name: {valence, confidence, priority, ...}}
        hijack_active:  whether forge_amygdala has triggered hijack

        Returns full ACC activation result.
        """
        t0         = time.time()
        self.cycle += 1
        module_outputs = module_outputs or {}

        # 1. Decay ERN signal each cycle
        self.error_mon.decay()

        # 2. Detect conflicts between modules
        conflict_score, description, conflicting_modules = self.detector.detect(
            module_outputs
        )

        # 3. Amplify via social pain / salience gating
        conflict_score, salience_reason = self.salience.amplify(signal, conflict_score)
        if salience_reason:
            description = f"{description} | salience: {salience_reason}".strip(" |")

        # 4. Classify conflict level
        level = self.detector.score_to_level(conflict_score)

        # 5. Build conflict event
        conflict = ConflictEvent(
            modules        = conflicting_modules,
            conflict_score = conflict_score,
            level          = level.value,
            description    = description,
        )

        if level != ConflictLevel.NONE:
            self.total_conflicts += 1
            self.db.save_conflict(conflict)
            self.open_conflicts.append(conflict)

        # 6. Age open conflicts
        still_open = []
        for oc in self.open_conflicts[-10:]:
            oc.cycles_open += 1
            if oc.cycles_open > 5:
                oc.resolved = True
                self.db.save_conflict(oc)
                self.db.log_resolution(oc.id, oc.resolution, "timeout", oc.cycles_open)
            else:
                still_open.append(oc)
        self.open_conflicts = still_open

        # 7. Error monitoring — check for expected outcome mismatches
        error_signal = None
        expected = signal.get("expected_outcome","")
        actual   = signal.get("actual_outcome","")
        if expected and actual and expected != actual:
            pred_id = str(uuid.uuid4())[:8]
            self.error_mon.register_prediction(pred_id, expected)
            error_signal = self.error_mon.evaluate(
                pred_id, actual,
                source_module=signal.get("source_module","unknown")
            )
            if error_signal:
                self.total_errors += 1

        # 8. Performance tracking
        confidence  = signal.get("confidence", 0.6)
        error_rate  = self.error_mon.total_errors / max(self.cycle, 1)
        sample      = self.perf.record(
            confidence, conflict_score, error_rate, len(module_outputs)
        )

        # 9. Choose resolution strategy
        strategy = self.resolver.suggest(
            conflict, error_signal,
            self.perf.rolling_health(), hijack_active
        )
        urgency  = self.resolver.urgency(level, self.perf.rolling_health())

        conflict.resolution = strategy.value
        self.db.save_conflict(conflict)

        # Track special events
        if strategy == ResolutionStrategy.INHIBIT_ACTION:
            self.total_inhibitions += 1
        if strategy == ResolutionStrategy.ESCALATE_AROUSAL:
            self.total_escalations += 1

        # 10. Compute output signals
        output = self.output_calc.compute(
            conflict, error_signal,
            self.perf.rolling_health(), strategy, urgency
        )

        elapsed = (time.time() - t0) * 1000

        return {
            "cycle":                self.cycle,
            "conflict_score":       conflict_score,
            "conflict_level":       level.value,
            "conflicting_modules":  conflicting_modules,
            "conflict_description": description,
            "resolution_strategy":  strategy.value,
            "urgency":              urgency,
            "ern_signal":           round(self.error_mon.ern_signal, 4),
            "error_detected":       error_signal is not None,
            "error_type":           error_signal.error_type if error_signal else "",
            "rolling_health":       self.perf.rolling_health(),
            "health_trend":         self.perf.trend(),
            "performance_alarm":    self.perf.alarm_active(),
            "open_conflicts":       len(self.open_conflicts),
            "output":               output,
            "processing_ms":        round(elapsed, 2),
            "total_conflicts":      self.total_conflicts,
            "total_errors":         self.total_errors,
            "total_inhibitions":    self.total_inhibitions,
        }

    def register_prediction(self, prediction_id: str, expected: str):
        """Pre-register a prediction before outcome is known."""
        self.error_mon.register_prediction(prediction_id, expected)

    def evaluate_prediction(self, prediction_id: str, actual: str,
                             source: str = "unknown") -> Optional[dict]:
        """Evaluate a previously registered prediction."""
        err = self.error_mon.evaluate(prediction_id, actual, source)
        if err:
            return {
                "error_type":    err.error_type,
                "predicted":     err.predicted,
                "actual":        err.actual,
                "ern_magnitude": err.ern_magnitude,
            }
        return None

    def get_status(self) -> dict:
        return {
            "version":           VERSION,
            "cycle":             self.cycle,
            "total_conflicts":   self.total_conflicts,
            "total_errors":      self.total_errors,
            "total_inhibitions": self.total_inhibitions,
            "total_escalations": self.total_escalations,
            "open_conflicts":    len(self.open_conflicts),
            "rolling_health":    self.perf.rolling_health(),
            "health_trend":      self.perf.trend(),
            "performance_alarm": self.perf.alarm_active(),
            "ern_signal":        round(self.error_mon.ern_signal, 4),
            "open_conflict_details": [
                {
                    "id":          c.id,
                    "level":       c.level,
                    "score":       c.conflict_score,
                    "modules":     c.modules,
                    "cycles_open": c.cycles_open,
                    "resolution":  c.resolution,
                }
                for c in self.open_conflicts[-5:]
            ],
        }


# ─── Rich UI ──────────────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "NONE":     "dim",
    "LOW":      "blue",
    "MODERATE": "yellow",
    "HIGH":     "red",
    "CRITICAL": "bright_red",
}

STRATEGY_COLORS = {
    "DEFER_TO_PREFRONTAL":  "cyan",
    "DEFER_TO_AMYGDALA":    "red",
    "PAUSE_AND_GATHER":     "yellow",
    "ESCALATE_AROUSAL":     "orange3",
    "INHIBIT_ACTION":       "bright_red",
    "NOTIFY_METACOGNITION": "blue",
}

def render_acc(result: dict, signal: dict, idx: int):
    if not HAS_RICH: return

    level    = result["conflict_level"]
    lc       = LEVEL_COLORS.get(level, "white")
    strategy = result["resolution_strategy"]
    sc       = STRATEGY_COLORS.get(strategy, "white")
    health   = result["rolling_health"]
    hc       = "green" if health > 0.6 else "yellow" if health > 0.4 else "red"
    urgency  = result["urgency"]

    console.print(Rule(
        f"[bold cyan]⬡ ANTERIOR CINGULATE[/bold cyan]  "
        f"[dim]#{idx}[/dim]  "
        f"[{lc}]{level}[/{lc}]  "
        f"[dim]conflict={result['conflict_score']:.3f}  "
        f"urgency={urgency:.2f}  "
        f"health={health:.2f}[/dim]"
    ))

    # Critical banner
    if level == "CRITICAL":
        console.print(Panel(
            f"[bold bright_red]⚡ ACC CRITICAL CONFLICT — ACTION INHIBITED[/bold bright_red]\n"
            f"Modules: {', '.join(result['conflicting_modules'])}\n"
            f"[red]{result['conflict_description']}[/red]\n"
            f"[dim]System paused pending conflict resolution.[/dim]",
            border_style="bright_red"
        ))

    # Performance alarm banner
    if result["performance_alarm"]:
        console.print(Panel(
            f"[bold yellow]⚠  PERFORMANCE ALARM — rolling health={health:.2f}[/bold yellow]\n"
            f"Trend: {result['health_trend']}  |  Strategy: [{sc}]{strategy}[/{sc}]",
            border_style="yellow"
        ))

    # Main columns
    cs = result["conflict_score"]
    c_bar = "█" * int(cs * 14) + "░" * (14 - int(cs * 14))
    h_bar = "█" * int(health * 14) + "░" * (14 - int(health * 14))
    ern   = result["ern_signal"]
    e_bar = "█" * int(ern * 14) + "░" * (14 - int(ern * 14))

    left_lines = [
        f"[bold]Conflict:[/bold]  [{lc}]{c_bar} {cs:.3f}[/{lc}]",
        f"[bold]Health:  [/bold]  [{hc}]{h_bar} {health:.3f}[/{hc}]",
        f"[bold]ERN:     [/bold]  [magenta]{e_bar} {ern:.3f}[/magenta]",
        f"",
        f"[bold]Resolution:[/bold]  [{sc}]{strategy}[/{sc}]",
        f"[bold]Urgency:[/bold]     [{lc}]{urgency:.3f}[/{lc}]",
        f"[bold]Trend:[/bold]       {result['health_trend']}",
        f"[bold]Open conflicts:[/bold] {result['open_conflicts']}",
    ]

    right_lines = []
    if result["conflicting_modules"]:
        right_lines.append("[bold]Conflicting modules:[/bold]")
        for m in result["conflicting_modules"]:
            right_lines.append(f"  [red]⚡ {m}[/red]")
        right_lines.append("")

    right_lines.append(f"[bold]Description:[/bold]")
    desc = result["conflict_description"]
    for part in desc.split("|"):
        if part.strip():
            right_lines.append(f"  [dim]{part.strip()}[/dim]")

    if result["error_detected"]:
        right_lines.append(f"\n[bold magenta]ERN ERROR:[/bold magenta]")
        right_lines.append(f"  type: {result['error_type']}")

    out = result["output"]
    right_lines.append(f"\n[bold]Output signals:[/bold]")
    right_lines.append(f"  NE signal:    {out['ne_signal']:.3f}")
    right_lines.append(f"  Attn redir:   {out['attention_redirect']:.3f}")
    right_lines.append(f"  Interrupt:    {out['interrupt_conscious']}")
    right_lines.append(f"  Action supp:  {out['action_suppressed']}")

    if not right_lines:
        right_lines = ["[dim]All modules in agreement[/dim]"]

    console.print(Columns([
        Panel("\n".join(left_lines),  title=f"[bold {lc}]ACC Signal[/bold {lc}]",  border_style=lc),
        Panel("\n".join(right_lines), title="[bold]Conflict + Output[/bold]",       border_style="dim")
    ]))


def run_demo():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FORGE ANTERIOR CINGULATE CORTEX[/bold cyan]\n"
            "[dim]Conflict Detection · Error Monitoring · Performance Tracking · Resolution[/dim]\n"
            f"[dim]Version {VERSION}  |  Port {API_PORT}[/dim]",
            border_style="cyan"
        ))

    acc = ForgeAnteriorCingulate()

    scenarios = [
        # All modules in agreement — no conflict
        (
            {"threat": 0, "anomaly": False, "confidence": 0.85,
             "social": {"trust_score": 0.9, "inferred_intent": "cooperative"},
             "expected_outcome": "", "actual_outcome": ""},
            {
                "amygdala":    {"valence": 0.7,  "confidence": 0.85, "fear_score": 0.1, "safety_score": 0.8, "priority": 1},
                "prefrontal":  {"valence": 0.75, "confidence": 0.80, "priority": 1},
                "hippocampus": {"valence": 0.6,  "confidence": 0.75, "priority": 1},
            },
            False,
            "All modules agree — no conflict"
        ),

        # Amygdala vs prefrontal disagreement
        (
            {"threat": 2, "anomaly": False, "confidence": 0.5,
             "social": {"trust_score": 0.5, "inferred_intent": "ambiguous"},
             "expected_outcome": "", "actual_outcome": ""},
            {
                "amygdala":   {"valence": 0.2,  "confidence": 0.8, "fear_score": 0.65, "safety_score": 0.1, "priority": 3},
                "prefrontal": {"valence": 0.75, "confidence": 0.7, "priority": 2},
                "thalamus":   {"valence": 0.5,  "confidence": 0.6, "priority": 1},
            },
            False,
            "Amygdala fear vs prefrontal calm — moderate conflict"
        ),

        # Outcome mismatch — error signal
        (
            {"threat": 1, "anomaly": False, "confidence": 0.6,
             "social": {"trust_score": 0.7, "inferred_intent": "neutral"},
             "expected_outcome": "SAFE", "actual_outcome": "ALARM",
             "source_module": "prefrontal"},
            {
                "amygdala":   {"valence": 0.3, "confidence": 0.7, "fear_score": 0.4, "safety_score": 0.3, "priority": 2},
                "prefrontal": {"valence": 0.4, "confidence": 0.5, "priority": 2},
            },
            False,
            "Outcome mismatch: expected SAFE, got ALARM — ERN spike"
        ),

        # Social pain — trust collapse
        (
            {"threat": 1, "anomaly": True, "confidence": 0.4,
             "social": {"trust_score": 0.05, "inferred_intent": "withdrawal"},
             "expected_outcome": "", "actual_outcome": ""},
            {
                "amygdala":   {"valence": 0.25, "confidence": 0.75, "fear_score": 0.5, "safety_score": 0.1, "priority": 3},
                "prefrontal": {"valence": 0.45, "confidence": 0.55, "priority": 2},
                "limbic":     {"valence": 0.2,  "confidence": 0.65, "priority": 3},
            },
            False,
            "Social pain — trust collapse + withdrawal"
        ),

        # Multiple modules competing for priority
        (
            {"threat": 2, "anomaly": True, "confidence": 0.35,
             "social": {"trust_score": 0.3, "inferred_intent": "ambiguous"},
             "expected_outcome": "", "actual_outcome": ""},
            {
                "amygdala":    {"valence": 0.2, "confidence": 0.8, "fear_score": 0.6, "safety_score": 0.15, "priority": 4},
                "prefrontal":  {"valence": 0.5, "confidence": 0.45, "priority": 3},
                "salience":    {"valence": 0.3, "confidence": 0.7,  "priority": 4},
                "thalamus":    {"valence": 0.4, "confidence": 0.5,  "priority": 3},
                "hippocampus": {"valence": 0.6, "confidence": 0.4,  "priority": 3},
            },
            False,
            "Priority storm — 5 modules competing"
        ),

        # Amygdala hijack active — everything defers
        (
            {"threat": 4, "anomaly": True, "confidence": 0.2,
             "social": {"trust_score": 0.05, "inferred_intent": "intrusion_attempt"},
             "expected_outcome": "", "actual_outcome": ""},
            {
                "amygdala":   {"valence": 0.0, "confidence": 0.95, "fear_score": 0.95, "safety_score": 0.0, "priority": 5},
                "prefrontal": {"valence": 0.3, "confidence": 0.2, "priority": 1},
            },
            True,
            "Amygdala hijack active — ACC defers to survival"
        ),

        # System recovering — performance improving
        (
            {"threat": 0, "anomaly": False, "confidence": 0.75,
             "social": {"trust_score": 0.8, "inferred_intent": "cooperative"},
             "expected_outcome": "", "actual_outcome": ""},
            {
                "amygdala":    {"valence": 0.7, "confidence": 0.8, "fear_score": 0.15, "safety_score": 0.7, "priority": 1},
                "prefrontal":  {"valence": 0.8, "confidence": 0.85, "priority": 1},
                "hippocampus": {"valence": 0.75,"confidence": 0.8,  "priority": 1},
            },
            False,
            "System recovering — agreement restored"
        ),
    ]

    for i, (sig, mods, hijack, label) in enumerate(scenarios):
        if HAS_RICH:
            console.print(f"\n[bold dim]━━━ {i+1}: {label.upper()} ━━━[/bold dim]")
        result = acc.process(sig, mods, hijack)
        render_acc(result, sig, i+1)
        time.sleep(0.08)

    # Final status
    if HAS_RICH:
        console.print(Rule("[bold cyan]⬡ ACC FINAL STATUS[/bold cyan]"))
        status = acc.get_status()

        st = Table(box=box.DOUBLE_EDGE, border_style="cyan", title="ACC Status")
        st.add_column("Metric", style="cyan")
        st.add_column("Value",  style="white")
        st.add_row("Cycles",          str(status["cycle"]))
        st.add_row("Total Conflicts", str(status["total_conflicts"]))
        st.add_row("Total Errors",    str(status["total_errors"]))
        st.add_row("Inhibitions",     str(status["total_inhibitions"]))
        st.add_row("Escalations",     str(status["total_escalations"]))
        st.add_row("Open Conflicts",  str(status["open_conflicts"]))
        st.add_row("Rolling Health",  f"{status['rolling_health']:.3f}")
        st.add_row("Health Trend",    status["health_trend"])
        st.add_row("ERN Signal",      f"{status['ern_signal']:.3f}")
        console.print(st)

        if status["open_conflict_details"]:
            ot = Table(box=box.SIMPLE, title="Open Conflicts", title_style="yellow")
            ot.add_column("ID",       style="dim")
            ot.add_column("Level",    style="red")
            ot.add_column("Score",    justify="right")
            ot.add_column("Modules")
            ot.add_column("Strategy", style="cyan")
            ot.add_column("Cycles",   justify="right")
            for c in status["open_conflict_details"]:
                lc2 = LEVEL_COLORS.get(c["level"], "white")
                ot.add_row(
                    c["id"],
                    f"[{lc2}]{c['level']}[/{lc2}]",
                    f"{c['score']:.3f}",
                    ", ".join(c["modules"][:3]),
                    c["resolution"],
                    str(c["cycles_open"])
                )
            console.print(ot)


# ─── HTTP API ─────────────────────────────────────────────────────────────────

def run_api(acc: ForgeAnteriorCingulate):
    if not HAS_FLASK: return
    app = Flask(__name__)

    @app.route("/process", methods=["POST"])
    def process():
        body   = request.json or {}
        signal = body.get("signal", {})
        mods   = body.get("module_outputs", {})
        hijack = body.get("hijack_active", False)
        return jsonify(acc.process(signal, mods, hijack))

    @app.route("/predict", methods=["POST"])
    def predict():
        body = request.json or {}
        acc.register_prediction(body["id"], body["expected"])
        return jsonify({"registered": body["id"]})

    @app.route("/evaluate", methods=["POST"])
    def evaluate():
        body   = request.json or {}
        result = acc.evaluate_prediction(
            body["id"], body["actual"],
            body.get("source", "unknown")
        )
        return jsonify(result or {"match": True})

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(acc.get_status())

    @app.route("/conflicts", methods=["GET"])
    def conflicts():
        rows = acc.db.get_recent_conflicts(15)
        return jsonify([{
            "timestamp":  r[0], "modules": json.loads(r[1]),
            "score":      r[2], "level":   r[3],
            "description":r[4], "resolution": r[5],
            "resolved":   bool(r[6])
        } for r in rows])

    @app.route("/errors", methods=["GET"])
    def errors():
        rows = acc.db.get_recent_errors(10)
        return jsonify([{
            "timestamp": r[0], "type":      r[1],
            "predicted": r[2], "actual":    r[3],
            "ern":       r[4], "source":    r[5]
        } for r in rows])

    @app.route("/health", methods=["GET"])
    def health():
        rows = acc.db.get_health_trend(20)
        return jsonify([{
            "cycle": r[0], "health": r[1],
            "conflict": r[2], "error_rate": r[3]
        } for r in rows])

    app.run(host="0.0.0.0", port=API_PORT, debug=False)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    acc = ForgeAnteriorCingulate()
    if "--api" in sys.argv:
        t = threading.Thread(target=run_api, args=(acc,), daemon=True)
        t.start()
    run_demo()
