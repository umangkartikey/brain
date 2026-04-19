"""
FORGE Anterior Cingulate Cortex — forge_anterior_cingulate.py
==============================================================
AI analog of the brain's anterior cingulate cortex (ACC).

The ACC sits at the intersection of cognition and emotion.
It is the brain's CONFLICT MONITOR — continuously watching
for situations where competing signals, responses, or beliefs
are simultaneously active and incompatible.

Three core functions:

  1. CONFLICT DETECTION
     When two or more responses are simultaneously activated
     and incompatible, the ACC fires an alert signal.
     This signal recruits additional cognitive control.
     Classic example: Stroop task — "RED" written in blue ink.
     Both "red" and "blue" activate simultaneously → conflict.

     In FORGE: threat=4 + intent=COOPERATIVE_REQUEST → conflict
               basal_ganglia=BLOCK + limbic=trust → conflict
               thalamus=CRISIS + amygdala=no_fear → conflict

  2. ERROR MONITORING (Error-Related Negativity)
     The ACC generates an error signal when outcomes
     deviate from expectations. This is distinct from the
     cerebellum's timing error — the ACC monitors MEANING.
     Not "that was 10ms late" but "that was WRONG."

  3. PERFORMANCE MONITORING
     Tracks performance over time. When errors cluster,
     ACC increases cognitive control allocation.
     When performance is smooth, control is relaxed.
     This is the neural basis of effort regulation.

Additional functions:
  - Pain processing (cognitive component of pain)
  - Emotional regulation (mediates limbic ↔ prefrontal)
  - Autonomic regulation (heart rate, cortisol modulation)
  - Motivation monitoring (effort vs reward tradeoff)

Architecture:
  ConflictDetector     → simultaneous incompatible activations
  ErrorMonitor         → outcome deviation from expectation
  PerformanceTracker   → running accuracy and effort metrics
  CognitiveController  → allocates control based on conflict/error
  EmotionRegulator     → mediates limbic ↔ prefrontal conflict
  MotivationEngine     → effort vs reward evaluation
  PainAnalog           → models cognitive cost of sustained conflict
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

DB_PATH  = "forge_anterior_cingulate.db"
API_PORT = 7794
VERSION  = "1.0.0"

# Conflict thresholds
CONFLICT_MILD     = 0.25
CONFLICT_MODERATE = 0.50
CONFLICT_SEVERE   = 0.75

# Error monitoring
ERN_THRESHOLD     = 0.30   # error-related negativity threshold
PERFORMANCE_WINDOW= 20     # cycles for performance tracking

# Cognitive control
CONTROL_BOOST_RATE= 0.15   # how fast control allocation increases
CONTROL_DECAY_RATE= 0.05   # how fast control relaxes after success

# Effort/motivation
EFFORT_COST_RATE  = 0.08   # cognitive cost per conflict
RECOVERY_RATE     = 0.04   # recovery per quiet cycle

console = Console() if HAS_RICH else None

# ─── Enums ────────────────────────────────────────────────────────────────────

class ConflictType(Enum):
    RESPONSE_CONFLICT    = "RESPONSE_CONFLICT"    # two actions competing
    INFORMATION_CONFLICT = "INFORMATION_CONFLICT" # contradictory signals
    EMOTIONAL_CONFLICT   = "EMOTIONAL_CONFLICT"   # emotion vs cognition
    GOAL_CONFLICT        = "GOAL_CONFLICT"         # competing goals
    PREDICTION_CONFLICT  = "PREDICTION_CONFLICT"  # reality vs expectation
    NONE                 = "NONE"

class ErrorType(Enum):
    COMMISSION   = "COMMISSION"    # did wrong thing
    OMISSION     = "OMISSION"      # failed to do right thing
    TIMING       = "TIMING"        # correct action, wrong time
    MAGNITUDE    = "MAGNITUDE"     # correct direction, wrong strength
    NONE         = "NONE"

class ControlLevel(Enum):
    MINIMAL   = "MINIMAL"     # < 0.25 — almost automatic
    LOW       = "LOW"         # 0.25-0.50 — routine monitoring
    MODERATE  = "MODERATE"    # 0.50-0.70 — active oversight
    HIGH      = "HIGH"        # 0.70-0.85 — intensive control
    MAXIMUM   = "MAXIMUM"     # > 0.85 — full executive engagement

class PerformanceState(Enum):
    EXCELLENT = "EXCELLENT"   # > 90% success
    GOOD      = "GOOD"        # 75-90%
    MODERATE  = "MODERATE"    # 55-75%
    POOR      = "POOR"        # 35-55%
    FAILING   = "FAILING"     # < 35%

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class ConflictSignal:
    """A detected conflict between competing signals or responses."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    conflict_type:str   = ConflictType.NONE.value
    sources:      list  = field(default_factory=list)   # which modules conflict
    signals:      dict  = field(default_factory=dict)   # conflicting values
    strength:     float = 0.0   # 0-1 how severe
    resolution:   str   = ""    # how was it resolved
    control_recruited:float=0.0 # how much extra control was allocated
    cycle:        int   = 0

@dataclass
class ErrorSignal:
    """An error detected by the ACC."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    error_type:   str   = ErrorType.NONE.value
    module:       str   = ""
    expected:     str   = ""
    actual:       str   = ""
    ern_amplitude:float = 0.0   # error-related negativity strength
    pe_amplitude: float = 0.0   # post-error positivity
    corrected:    bool  = False
    cycle:        int   = 0

@dataclass
class PerformanceMetrics:
    """Rolling performance metrics over recent cycles."""
    timestamp:      str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:          int   = 0
    success_rate:   float = 1.0
    conflict_rate:  float = 0.0
    error_rate:     float = 0.0
    control_level:  float = 0.0
    effort_cost:    float = 0.0
    motivation:     float = 0.8
    state:          str   = PerformanceState.EXCELLENT.value

@dataclass
class ACCOutput:
    """Complete output from the ACC for downstream modules."""
    timestamp:      str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:          int   = 0
    conflict:       Optional[ConflictSignal] = None
    error:          Optional[ErrorSignal]    = None
    control_level:  float = 0.0
    control_label:  str   = ControlLevel.LOW.value
    attention_boost:float = 0.0   # boost to prefrontal attention
    error_flag:     bool  = False
    conflict_flag:  bool  = False
    performance:    str   = PerformanceState.EXCELLENT.value
    effort_cost:    float = 0.0
    motivation:     float = 0.8
    recommendation: str   = ""

# ─── Database ─────────────────────────────────────────────────────────────────

class ACCDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS conflict_signals (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    conflict_type TEXT, sources TEXT,
                    signals TEXT, strength REAL,
                    resolution TEXT, control_recruited REAL, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS error_signals (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    error_type TEXT, module TEXT,
                    expected TEXT, actual TEXT,
                    ern_amplitude REAL, pe_amplitude REAL,
                    corrected INTEGER, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS performance_log (
                    cycle INTEGER PRIMARY KEY, timestamp TEXT,
                    success_rate REAL, conflict_rate REAL,
                    error_rate REAL, control_level REAL,
                    effort_cost REAL, motivation REAL, state TEXT
                );
                CREATE TABLE IF NOT EXISTS control_log (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    trigger TEXT, control_before REAL,
                    control_after REAL, reason TEXT, cycle INTEGER
                );
            """)
            self.conn.commit()

    def save_conflict(self, c: ConflictSignal):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO conflict_signals VALUES
                (?,?,?,?,?,?,?,?,?)
            """, (c.id, c.timestamp, c.conflict_type,
                  json.dumps(c.sources), json.dumps(c.signals),
                  c.strength, c.resolution, c.control_recruited, c.cycle))
            self.conn.commit()

    def save_error(self, e: ErrorSignal):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO error_signals VALUES
                (?,?,?,?,?,?,?,?,?,?)
            """, (e.id, e.timestamp, e.error_type, e.module,
                  e.expected, e.actual, e.ern_amplitude,
                  e.pe_amplitude, int(e.corrected), e.cycle))
            self.conn.commit()

    def save_performance(self, p: PerformanceMetrics):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO performance_log VALUES
                (?,?,?,?,?,?,?,?,?)
            """, (p.cycle, p.timestamp, p.success_rate,
                  p.conflict_rate, p.error_rate, p.control_level,
                  p.effort_cost, p.motivation, p.state))
            self.conn.commit()

    def log_control_change(self, trigger: str, before: float,
                           after: float, reason: str, cycle: int):
        with self.lock:
            self.conn.execute("""
                INSERT INTO control_log VALUES (?,?,?,?,?,?,?)
            """, (str(uuid.uuid4())[:8], datetime.now().isoformat(),
                  trigger, before, after, reason, cycle))
            self.conn.commit()

    def get_recent_conflicts(self, limit=10):
        with self.lock:
            return self.conn.execute("""
                SELECT timestamp, conflict_type, strength,
                       sources, resolution
                FROM conflict_signals
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_recent_errors(self, limit=10):
        with self.lock:
            return self.conn.execute("""
                SELECT timestamp, error_type, module,
                       expected, actual, ern_amplitude
                FROM error_signals
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()


# ─── Conflict Detector ────────────────────────────────────────────────────────

class ConflictDetector:
    """
    Detects simultaneous incompatible activations.
    The ACC's primary function — finds the contradiction
    before it propagates through the system.

    Conflict rules:
      Threat ≥ 3 + cooperative intent     → INFORMATION_CONFLICT
      Amygdala hijack + no salience alarm  → PREDICTION_CONFLICT
      Basal ganglia BLOCK + limbic TRUST   → RESPONSE_CONFLICT
      Thalamus CRISIS + low fear score     → PREDICTION_CONFLICT
      Prefrontal STANDBY + high threat     → RESPONSE_CONFLICT
      Emotion FEAR + action APPROACH       → EMOTIONAL_CONFLICT
      Multiple goals competing             → GOAL_CONFLICT
    """

    CONFLICT_RULES = [
        {
            "id":   "CR_001",
            "name": "threat_cooperative_conflict",
            "type": ConflictType.INFORMATION_CONFLICT,
            "check": lambda s: (
                s.get("threat", 0) >= 3 and
                "COOPERATIVE" in str(s.get("social", {}).get("inferred_intent", ""))
            ),
            "sources": ["temporal","bridge"],
            "strength": 0.75,
            "description": "High threat signal with cooperative intent — contradictory"
        },
        {
            "id":   "CR_002",
            "name": "block_trust_conflict",
            "type": ConflictType.RESPONSE_CONFLICT,
            "check": lambda s: (
                "BLOCK" in str(s.get("decision", "")) and
                s.get("social_context", {}).get("trust_score", 0) > 0.7
            ),
            "sources": ["prefrontal","bridge"],
            "strength": 0.65,
            "description": "Decision to block a high-trust entity"
        },
        {
            "id":   "CR_003",
            "name": "crisis_no_fear_conflict",
            "type": ConflictType.PREDICTION_CONFLICT,
            "check": lambda s: (
                s.get("consciousness_state", "") == "CRISIS" and
                s.get("fear_score", 1.0) < 0.2
            ),
            "sources": ["thalamus","amygdala"],
            "strength": 0.60,
            "description": "Thalamus in CRISIS but amygdala reports low fear"
        },
        {
            "id":   "CR_004",
            "name": "standby_high_threat",
            "type": ConflictType.RESPONSE_CONFLICT,
            "check": lambda s: (
                "STANDBY" in str(s.get("decision", "")) and
                s.get("threat", 0) >= 3
            ),
            "sources": ["prefrontal","temporal"],
            "strength": 0.80,
            "description": "Standby decision under high threat — under-response"
        },
        {
            "id":   "CR_005",
            "name": "fear_approach_conflict",
            "type": ConflictType.EMOTIONAL_CONFLICT,
            "check": lambda s: (
                s.get("emotion", "") in ["fear","terror"] and
                "APPROACH" in str(s.get("affordances", []))
            ),
            "sources": ["amygdala","visual"],
            "strength": 0.55,
            "description": "Fear emotion while visual suggests approach affordance"
        },
        {
            "id":   "CR_006",
            "name": "novelty_habit_conflict",
            "type": ConflictType.RESPONSE_CONFLICT,
            "check": lambda s: (
                s.get("novelty", 0) > 0.85 and
                s.get("habit_stage", "") in ["EXPERT","HABITUAL"]
            ),
            "sources": ["hippocampus","basal_ganglia"],
            "strength": 0.45,
            "description": "Novel situation being handled by ingrained habit"
        },
        {
            "id":   "CR_007",
            "name": "hijack_deliberate_conflict",
            "type": ConflictType.RESPONSE_CONFLICT,
            "check": lambda s: (
                s.get("hijack", False) and
                s.get("tier", "") == "DELIBERATE"
            ),
            "sources": ["amygdala","sensorimotor"],
            "strength": 0.90,
            "description": "Amygdala hijack active but deliberate response selected"
        },
        {
            "id":   "CR_008",
            "name": "burnout_high_demand",
            "type": ConflictType.GOAL_CONFLICT,
            "check": lambda s: (
                s.get("neuro_state", "") == "BURNOUT" and
                s.get("threat", 0) >= 2
            ),
            "sources": ["neuromodulator","prefrontal"],
            "strength": 0.70,
            "description": "System in BURNOUT state but high demand continues"
        },
    ]

    def detect(self, signal: dict, cycle: int) -> Optional[ConflictSignal]:
        """Check all conflict rules. Return highest-strength conflict."""
        detected = []

        for rule in self.CONFLICT_RULES:
            try:
                if rule["check"](signal):
                    detected.append(rule)
            except Exception:
                continue

        if not detected:
            return None

        # Pick highest strength conflict
        winner = max(detected, key=lambda r: r["strength"])

        conflict = ConflictSignal(
            conflict_type = winner["type"].value,
            sources       = winner["sources"],
            signals       = {
                "rule":        winner["name"],
                "description": winner["description"],
                "threat":      signal.get("threat", 0),
                "decision":    str(signal.get("decision",""))[:30],
                "emotion":     signal.get("emotion",""),
                "intent":      str(signal.get("social",{}) or {})[:30]
            },
            strength      = winner["strength"],
            cycle         = cycle
        )
        return conflict

    def all_conflicts(self, signal: dict, cycle: int) -> list[ConflictSignal]:
        """Return ALL detected conflicts (not just strongest)."""
        conflicts = []
        for rule in self.CONFLICT_RULES:
            try:
                if rule["check"](signal):
                    conflicts.append(ConflictSignal(
                        conflict_type = rule["type"].value,
                        sources       = rule["sources"],
                        signals       = {"rule": rule["name"],
                                        "description": rule["description"]},
                        strength      = rule["strength"],
                        cycle         = cycle
                    ))
            except Exception:
                continue
        return sorted(conflicts, key=lambda c: c.strength, reverse=True)


# ─── Error Monitor ────────────────────────────────────────────────────────────

class ErrorMonitor:
    """
    Monitors for meaningful errors — not timing errors (cerebellum)
    but SEMANTIC errors: doing the wrong thing, expecting wrong outcome.

    Generates Error-Related Negativity (ERN) — the neural signal
    that fires within 100ms of an error, before conscious awareness.

    Also generates Post-Error Positivity (Pe) — the later signal
    that reflects conscious error recognition and adjustment.
    """

    def __init__(self):
        self.error_history: deque = deque(maxlen=50)
        self.post_error_slowing = 0.0  # post-error caution level

    def monitor(self, signal: dict,
                expected_decision: str,
                actual_decision: str,
                cycle: int) -> Optional[ErrorSignal]:
        """Detect semantic errors in module outputs."""

        if not expected_decision or not actual_decision:
            return None

        # Check for obvious errors
        error_type = self._classify_error(signal, expected_decision, actual_decision)
        if error_type == ErrorType.NONE:
            # Post-error positivity — system stabilizing
            self.post_error_slowing = max(0.0, self.post_error_slowing - 0.05)
            return None

        # Compute ERN amplitude — how surprising is this error?
        ern = self._compute_ern(signal, expected_decision, actual_decision)

        # Post-error adjustment — slow down after errors
        self.post_error_slowing = min(1.0, self.post_error_slowing + 0.2)

        err = ErrorSignal(
            error_type    = error_type.value,
            module        = signal.get("source_module","unknown"),
            expected      = expected_decision[:40],
            actual        = actual_decision[:40],
            ern_amplitude = ern,
            pe_amplitude  = round(ern * 0.6, 4),  # Pe is smaller
            cycle         = cycle
        )
        self.error_history.append(err)
        return err

    def _classify_error(self, signal: dict,
                         expected: str, actual: str) -> ErrorType:
        """Classify the type of error."""
        threat = signal.get("threat", 0)

        # Commission error — did wrong thing
        if threat >= 3 and "STANDBY" in actual and "BLOCK" in expected:
            return ErrorType.COMMISSION
        if threat == 0 and "ESCALATE" in actual and "MONITOR" in expected:
            return ErrorType.COMMISSION

        # Omission error — failed to act
        if threat >= 4 and actual in ["STANDBY","MONITOR","NONE"]:
            return ErrorType.OMISSION

        # Magnitude error — right direction, wrong strength
        if threat == 1 and "EMERGENCY" in actual:
            return ErrorType.MAGNITUDE
        if threat == 3 and "MONITOR" in actual:
            return ErrorType.MAGNITUDE

        return ErrorType.NONE

    def _compute_ern(self, signal: dict,
                      expected: str, actual: str) -> float:
        """Compute ERN amplitude — surprise × importance."""
        threat = signal.get("threat", 0)

        # Base ERN from mismatch severity
        if expected and actual and expected != actual:
            # Rough semantic distance
            base_ern = 0.4 + threat * 0.1
        else:
            base_ern = 0.2

        # Amplify if high threat
        if threat >= 3:
            base_ern = min(1.0, base_ern * 1.5)

        # Amplify if history of errors
        recent_errors = len([e for e in self.error_history
                             if e.ern_amplitude > ERN_THRESHOLD])
        if recent_errors > 3:
            base_ern = min(1.0, base_ern * 1.2)

        return round(base_ern, 4)

    def post_error_rate(self) -> float:
        """How much to slow down after errors."""
        return round(self.post_error_slowing, 3)

    def recent_error_rate(self, window: int = 10) -> float:
        """Error rate over recent cycles."""
        if not self.error_history: return 0.0
        recent = list(self.error_history)[-window:]
        errors = sum(1 for e in recent if e.ern_amplitude > ERN_THRESHOLD)
        return round(errors / window, 3)


# ─── Performance Tracker ──────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Tracks performance over time.
    When errors cluster → ACC increases control.
    When smooth → control relaxes.
    """

    def __init__(self):
        self.outcomes:  deque = deque(maxlen=PERFORMANCE_WINDOW)
        self.conflicts: deque = deque(maxlen=PERFORMANCE_WINDOW)
        self.errors:    deque = deque(maxlen=PERFORMANCE_WINDOW)
        self.cycle      = 0

    def record(self, success: bool, had_conflict: bool, had_error: bool):
        self.cycle += 1
        self.outcomes.append(int(success))
        self.conflicts.append(int(had_conflict))
        self.errors.append(int(had_error))

    def success_rate(self) -> float:
        if not self.outcomes: return 1.0
        return round(sum(self.outcomes) / len(self.outcomes), 3)

    def conflict_rate(self) -> float:
        if not self.conflicts: return 0.0
        return round(sum(self.conflicts) / len(self.conflicts), 3)

    def error_rate(self) -> float:
        if not self.errors: return 0.0
        return round(sum(self.errors) / len(self.errors), 3)

    def performance_state(self) -> PerformanceState:
        sr = self.success_rate()
        if sr > 0.90: return PerformanceState.EXCELLENT
        if sr > 0.75: return PerformanceState.GOOD
        if sr > 0.55: return PerformanceState.MODERATE
        if sr > 0.35: return PerformanceState.POOR
        return PerformanceState.FAILING

    def metrics(self) -> PerformanceMetrics:
        state = self.performance_state()
        return PerformanceMetrics(
            cycle        = self.cycle,
            success_rate = self.success_rate(),
            conflict_rate= self.conflict_rate(),
            error_rate   = self.error_rate(),
            state        = state.value
        )


# ─── Cognitive Controller ─────────────────────────────────────────────────────

class CognitiveController:
    """
    Allocates cognitive control based on conflict and error signals.
    The ACC's executive function — determines how much
    prefrontal engagement is needed.

    High conflict   → boost control (slow down, be more careful)
    Error detected  → boost control (post-error adjustment)
    Smooth running  → reduce control (save resources)

    Control level directly maps to:
    - Prefrontal deliberation depth
    - Thalamus gate sensitivity
    - Sensorimotor reflex threshold
    - Basal ganglia competition threshold
    """

    def __init__(self, db: ACCDB):
        self.db    = db
        self.level = 0.30   # baseline control
        self.cycle = 0

    def update(self, conflict: Optional[ConflictSignal],
               error: Optional[ErrorSignal],
               performance: PerformanceMetrics,
               cycle: int) -> float:
        old   = self.level
        self.cycle = cycle

        # Conflict boosts control
        if conflict:
            boost = CONTROL_BOOST_RATE * conflict.strength
            self.level = min(1.0, self.level + boost)

        # Error boosts control
        if error:
            boost = CONTROL_BOOST_RATE * error.ern_amplitude
            self.level = min(1.0, self.level + boost)

        # Poor performance boosts control
        if performance.state in [PerformanceState.POOR.value,
                                  PerformanceState.FAILING.value]:
            self.level = min(1.0, self.level + 0.05)

        # Excellent performance relaxes control
        elif performance.state == PerformanceState.EXCELLENT.value:
            self.level = max(0.10, self.level - CONTROL_DECAY_RATE)

        self.level = round(self.level, 4)

        if abs(self.level - old) > 0.02:
            self.db.log_control_change(
                trigger = conflict.conflict_type if conflict else
                          error.error_type if error else "performance",
                before  = old,
                after   = self.level,
                reason  = (conflict.signals.get("description","") if conflict else
                           f"ERN={error.ern_amplitude:.2f}" if error else
                           performance.state),
                cycle   = cycle
            )

        return self.level

    def label(self) -> str:
        if self.level < 0.25: return ControlLevel.MINIMAL.value
        if self.level < 0.50: return ControlLevel.LOW.value
        if self.level < 0.70: return ControlLevel.MODERATE.value
        if self.level < 0.85: return ControlLevel.HIGH.value
        return ControlLevel.MAXIMUM.value

    def attention_boost(self) -> float:
        """Boost to prefrontal attention allocation."""
        return round(max(0.0, self.level - 0.3) * 0.5, 4)


# ─── Emotion Regulator ────────────────────────────────────────────────────────

class EmotionRegulator:
    """
    Mediates conflict between limbic (emotional) and prefrontal (rational).
    The ACC sits between these two and can:
      - Dampen emotional responses when cognition is needed
      - Allow emotion through when cognition is overwhelmed
      - Flag when emotion and cognition directly contradict

    This is the neural basis of emotion regulation.
    """

    def __init__(self):
        self.regulation_history: deque = deque(maxlen=50)
        self.suppression_level = 0.0

    def regulate(self, signal: dict) -> dict:
        """
        Compute emotion regulation signal.
        Returns how much to dampen/amplify emotional response.
        """
        emotion      = signal.get("emotion","neutral")
        mood_valence = signal.get("mood_valence", 0.0)
        threat       = signal.get("threat", 0)
        decision     = str(signal.get("decision",""))

        # Cases where cognition should override emotion
        cognitive_override = 0.0
        if threat >= 3 and emotion in ["disgust","sadness"]:
            cognitive_override = 0.4  # suppress non-survival emotions in crisis
        if "COLLABORATE" in decision and emotion == "fear" and threat == 0:
            cognitive_override = 0.3  # suppress fear during safe collaboration

        # Cases where emotion should override cognition
        emotional_override = 0.0
        if emotion in ["terror","panic"] and threat >= 4:
            emotional_override = 0.5  # let fear through in real crisis
        if mood_valence < -0.6 and threat == 0:
            emotional_override = 0.2  # persistent distress deserves attention

        self.suppression_level = round(cognitive_override - emotional_override, 4)
        self.regulation_history.append(self.suppression_level)

        return {
            "cognitive_override":  cognitive_override,
            "emotional_override":  emotional_override,
            "net_suppression":     self.suppression_level,
            "emotion_allowed":     emotional_override > cognitive_override,
            "regulation_active":   abs(self.suppression_level) > 0.1
        }


# ─── Motivation Engine ────────────────────────────────────────────────────────

class MotivationEngine:
    """
    Monitors effort vs reward tradeoff.
    When cognitive cost is high and reward is low,
    motivation drops → system reduces engagement.

    This is the neural basis of mental fatigue and giving up.
    The ACC continuously evaluates: is this worth it?
    """

    def __init__(self):
        self.motivation    = 0.80
        self.effort_cost   = 0.0
        self.reward_history:deque = deque(maxlen=30)
        self.effort_history:deque = deque(maxlen=30)

    def update(self, conflict_strength: float,
               success: bool, threat: int) -> dict:
        # Effort accumulates with conflicts
        self.effort_cost = min(1.0,
            self.effort_cost + EFFORT_COST_RATE * conflict_strength
        )

        # Reward from success
        reward = 0.6 + threat * 0.1 if success else 0.2
        self.reward_history.append(reward)
        self.effort_history.append(self.effort_cost)

        # Natural recovery when quiet
        if conflict_strength < 0.1:
            self.effort_cost = max(0.0, self.effort_cost - RECOVERY_RATE)

        # Motivation = reward - effort (simplified)
        avg_reward = sum(self.reward_history)/len(self.reward_history)
        self.motivation = round(
            max(0.1, min(1.0, avg_reward - self.effort_cost * 0.5)), 4
        )

        return {
            "motivation":   self.motivation,
            "effort_cost":  round(self.effort_cost, 4),
            "avg_reward":   round(avg_reward, 4),
            "label":        self._label()
        }

    def _label(self) -> str:
        if self.motivation > 0.75: return "ENGAGED"
        if self.motivation > 0.55: return "MODERATE"
        if self.motivation > 0.35: return "FATIGUED"
        return "EXHAUSTED"


# ─── Pain Analog ─────────────────────────────────────────────────────────────

class PainAnalog:
    """
    Models the cognitive component of pain.
    Sustained conflict is cognitively painful.
    The ACC generates an aversive signal proportional to
    unresolved conflict duration.

    This is not physical pain — it is the discomfort of:
    - Sustained cognitive dissonance
    - Unresolved decision conflicts
    - Persistent error states
    - Chronic high cognitive load

    High cognitive pain → system seeks to resolve or escape.
    """

    def __init__(self):
        self.pain_level   = 0.0
        self.pain_history: deque = deque(maxlen=50)
        self.unresolved_cycles = 0

    def update(self, conflict: Optional[ConflictSignal],
               resolved: bool) -> float:
        if conflict and not resolved:
            self.unresolved_cycles += 1
            self.pain_level = min(1.0,
                self.pain_level + conflict.strength * 0.1
            )
        else:
            self.unresolved_cycles = 0
            self.pain_level = max(0.0, self.pain_level - 0.08)

        self.pain_level = round(self.pain_level, 4)
        self.pain_history.append(self.pain_level)
        return self.pain_level

    def pain_label(self) -> str:
        if self.pain_level < 0.1:  return "NONE"
        if self.pain_level < 0.3:  return "MILD"
        if self.pain_level < 0.6:  return "MODERATE"
        if self.pain_level < 0.8:  return "SEVERE"
        return "UNBEARABLE"


# ─── FORGE Anterior Cingulate ─────────────────────────────────────────────────

class ForgeAnteriorCingulate:
    def __init__(self):
        self.db           = ACCDB()
        self.detector     = ConflictDetector()
        self.error_monitor= ErrorMonitor()
        self.performance  = PerformanceTracker()
        self.controller   = CognitiveController(self.db)
        self.emotion_reg  = EmotionRegulator()
        self.motivation   = MotivationEngine()
        self.pain         = PainAnalog()
        self.cycle        = 0
        self.total_conflicts   = 0
        self.total_errors      = 0
        self.total_resolutions = 0

    def process(self, signal: dict,
                expected_decision: str = "",
                success: bool = True) -> dict:
        """
        Full ACC processing.
        Returns conflict detection, error monitoring,
        control level, and all downstream effects.
        """
        self.cycle += 1
        threat = signal.get("threat", 0)

        # 1. Detect conflicts
        conflict = self.detector.detect(signal, self.cycle)
        all_conflicts = self.detector.all_conflicts(signal, self.cycle)
        if conflict:
            self.total_conflicts += 1
            self.db.save_conflict(conflict)

        # 2. Monitor errors
        actual_decision = str(signal.get("decision",""))
        error = self.error_monitor.monitor(
            signal, expected_decision, actual_decision, self.cycle
        )
        if error:
            self.total_errors += 1
            self.db.save_error(error)

        # 3. Track performance
        self.performance.record(success, bool(conflict), bool(error))
        perf = self.performance.metrics()

        # 4. Update cognitive control
        control_level = self.controller.update(conflict, error, perf, self.cycle)
        self.db.save_performance(perf)

        # 5. Emotion regulation
        emotion_reg = self.emotion_reg.regulate(signal)

        # 6. Motivation update
        motiv = self.motivation.update(
            conflict.strength if conflict else 0.0, success, threat
        )

        # 7. Cognitive pain
        resolved = not bool(conflict) or success
        if resolved and conflict: self.total_resolutions += 1
        pain_level = self.pain.update(conflict, resolved)

        # 8. Build recommendation
        recommendation = self._recommend(conflict, error, control_level,
                                          motiv["motivation"], perf)

        # 9. Build output
        output = ACCOutput(
            cycle         = self.cycle,
            conflict      = conflict,
            error         = error,
            control_level = control_level,
            control_label = self.controller.label(),
            attention_boost = self.controller.attention_boost(),
            error_flag    = bool(error),
            conflict_flag = bool(conflict),
            performance   = perf.state,
            effort_cost   = motiv["effort_cost"],
            motivation    = motiv["motivation"],
            recommendation= recommendation
        )

        return {
            "cycle":            self.cycle,
            "conflict": {
                "detected":     bool(conflict),
                "type":         conflict.conflict_type if conflict else "NONE",
                "strength":     conflict.strength if conflict else 0.0,
                "sources":      conflict.sources if conflict else [],
                "description":  conflict.signals.get("description","") if conflict else "",
                "all_count":    len(all_conflicts),
            },
            "error": {
                "detected":     bool(error),
                "type":         error.error_type if error else "NONE",
                "ern_amplitude":error.ern_amplitude if error else 0.0,
                "expected":     error.expected if error else "",
                "actual":       error.actual if error else "",
            },
            "control": {
                "level":        control_level,
                "label":        self.controller.label(),
                "attention_boost": self.controller.attention_boost(),
            },
            "performance": {
                "state":        perf.state,
                "success_rate": perf.success_rate,
                "conflict_rate":perf.conflict_rate,
                "error_rate":   perf.error_rate,
            },
            "emotion_regulation": emotion_reg,
            "motivation":     motiv,
            "pain": {
                "level":        pain_level,
                "label":        self.pain.pain_label(),
                "unresolved_cycles": self.pain.unresolved_cycles,
            },
            "recommendation": recommendation,
            "post_error_slowing": self.error_monitor.post_error_rate(),
        }

    def _recommend(self, conflict: Optional[ConflictSignal],
                   error: Optional[ErrorSignal],
                   control: float, motivation: float,
                   perf: PerformanceMetrics) -> str:
        if conflict and conflict.strength > CONFLICT_SEVERE:
            return f"RESOLVE_CONFLICT: {conflict.signals.get('description','')[:50]}"
        if error and error.error_type == ErrorType.OMISSION.value:
            return "INCREASE_ENGAGEMENT: critical omission detected"
        if error and error.error_type == ErrorType.COMMISSION.value:
            return "RECALIBRATE: commission error — wrong action taken"
        if control > 0.8:
            return "SLOW_DOWN: maximum cognitive control engaged"
        if motivation < 0.3:
            return "REST_REQUIRED: motivation critically low"
        if perf.state == PerformanceState.FAILING.value:
            return "SYSTEM_REVIEW: performance below acceptable threshold"
        if conflict and conflict.strength > CONFLICT_MODERATE:
            return f"MONITOR_CONFLICT: {conflict.conflict_type}"
        return "CONTINUE: no significant conflict or error detected"

    def get_status(self) -> dict:
        return {
            "version":           VERSION,
            "cycle":             self.cycle,
            "total_conflicts":   self.total_conflicts,
            "total_errors":      self.total_errors,
            "total_resolutions": self.total_resolutions,
            "control_level":     self.controller.level,
            "control_label":     self.controller.label(),
            "motivation":        self.motivation.motivation,
            "effort_cost":       self.motivation.effort_cost,
            "pain_level":        self.pain.pain_level,
            "performance":       self.performance.metrics().__dict__,
        }


# ─── Rich UI ──────────────────────────────────────────────────────────────────

CONFLICT_COLORS = {
    "RESPONSE_CONFLICT":    "bright_red",
    "INFORMATION_CONFLICT": "red",
    "EMOTIONAL_CONFLICT":   "magenta",
    "GOAL_CONFLICT":        "yellow",
    "PREDICTION_CONFLICT":  "orange3",
    "NONE":                 "green",
}

CONTROL_COLORS = {
    "MINIMAL":  "green",
    "LOW":      "dim",
    "MODERATE": "yellow",
    "HIGH":     "orange3",
    "MAXIMUM":  "bright_red",
}

PERF_COLORS = {
    "EXCELLENT": "bright_green",
    "GOOD":      "green",
    "MODERATE":  "yellow",
    "POOR":      "orange3",
    "FAILING":   "bright_red",
}

def render_acc(result: dict, label: str, idx: int):
    if not HAS_RICH: return

    conflict  = result["conflict"]
    error     = result["error"]
    control   = result["control"]
    perf      = result["performance"]
    motiv     = result["motivation"]
    pain      = result["pain"]

    ct  = conflict["type"]
    cc  = CONFLICT_COLORS.get(ct, "white")
    cl  = control["label"]
    clc = CONTROL_COLORS.get(cl, "white")
    ps  = perf["state"]
    pc  = PERF_COLORS.get(ps, "white")

    console.print(Rule(
        f"[bold cyan]⬡ ACC[/bold cyan]  [dim]#{idx}[/dim]  "
        f"[{cc}]{ct}[/{cc}]  "
        f"control=[{clc}]{cl}[/{clc}]  "
        f"perf=[{pc}]{ps}[/{pc}]"
    ))

    # Left: conflict + error
    left_lines = []
    if conflict["detected"]:
        left_lines += [
            f"[bold {cc}]⚡ CONFLICT DETECTED[/bold {cc}]",
            f"[bold]Type:[/bold]    [{cc}]{ct}[/{cc}]",
            f"[bold]Strength:[/bold] {'█'*int(conflict['strength']*10)}{'░'*(10-int(conflict['strength']*10))} {conflict['strength']:.2f}",
            f"[bold]Sources:[/bold] {' + '.join(conflict['sources'])}",
            f"[dim]{conflict['description'][:55]}[/dim]",
        ]
        if conflict["all_count"] > 1:
            left_lines.append(f"[dim]+{conflict['all_count']-1} other conflict(s)[/dim]")
    else:
        left_lines.append("[green]✓ No conflict detected[/green]")

    if error["detected"]:
        ern = error["ern_amplitude"]
        left_lines += [
            f"\n[bold red]✗ ERROR: {error['type']}[/bold red]",
            f"[dim]Expected: {error['expected'][:25]}[/dim]",
            f"[dim]Actual:   {error['actual'][:25]}[/dim]",
            f"[dim]ERN amplitude: {ern:.3f}[/dim]",
        ]
    else:
        left_lines.append("\n[green]✓ No errors detected[/green]")

    # Right: control + motivation + pain
    mc  = "green" if motiv["motivation"]>0.6 else "yellow" if motiv["motivation"]>0.35 else "red"
    plc = {"NONE":"green","MILD":"yellow","MODERATE":"orange3",
           "SEVERE":"red","UNBEARABLE":"bright_red"}.get(pain["label"],"white")

    right_lines = [
        f"[bold]Control:[/bold]   [{clc}]{'█'*int(control['level']*10)}{'░'*(10-int(control['level']*10))} {cl}[/{clc}]",
        f"[bold]Attn boost:[/bold] +{control['attention_boost']:.3f}",
        f"",
        f"[bold]Success:[/bold]  {perf['success_rate']:.0%}",
        f"[bold]Conflicts:[/bold] {perf['conflict_rate']:.0%}",
        f"[bold]Errors:[/bold]   {perf['error_rate']:.0%}",
        f"",
        f"[bold]Motivation:[/bold] [{mc}]{motiv['label']} {motiv['motivation']:.3f}[/{mc}]",
        f"[bold]Effort:[/bold]     {motiv['effort_cost']:.3f}",
        f"[bold]Pain:[/bold]       [{plc}]{pain['label']} {pain['level']:.3f}[/{plc}]",
    ]

    console.print(Columns([
        Panel("\n".join(left_lines), title="[bold]Conflict + Error[/bold]", border_style=cc),
        Panel("\n".join(right_lines),title="[bold]Control + State[/bold]", border_style=clc)
    ]))

    # Recommendation
    rec = result["recommendation"]
    rec_color = "bright_red" if "CONFLICT" in rec or "ERROR" in rec else \
                "yellow" if "MONITOR" in rec or "SLOW" in rec else "dim"
    console.print(Panel(
        f"[{rec_color}]{rec}[/{rec_color}]",
        title="[bold]ACC Recommendation[/bold]",
        border_style=rec_color
    ))


def run_demo():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FORGE ANTERIOR CINGULATE CORTEX[/bold cyan]\n"
            "[dim]Conflict Detection · Error Monitoring · Cognitive Control[/dim]\n"
            f"[dim]Version {VERSION}[/dim]",
            border_style="cyan"
        ))

    acc = ForgeAnteriorCingulate()

    scenarios = [
        # Clean signal — no conflict
        ({"threat":0,"anomaly":False,
          "social":{"inferred_intent":"COOPERATIVE_REQUEST"},
          "decision":"MONITOR","emotion":"calm",
          "mood_valence":0.1,"consciousness_state":"AWAKE",
          "fear_score":0.0,"neuro_state":"BASELINE",
          "habit_stage":"HABITUAL","novelty":0.3},
         "MONITOR", True, "Clean cooperative — no conflict"),

        # Threat + cooperative intent — information conflict
        ({"threat":3,"anomaly":False,
          "social":{"inferred_intent":"COOPERATIVE_REQUEST"},
          "decision":"ALERT","emotion":"fear",
          "mood_valence":-0.3,"consciousness_state":"FOCUSED",
          "fear_score":0.7,"neuro_state":"HYPERVIGILANCE",
          "habit_stage":"DEVELOPING","novelty":0.6},
         "ALERT", True, "Threat=3 + cooperative intent — information conflict"),

        # Standby under high threat — response conflict + error
        ({"threat":4,"anomaly":True,
          "social":{"inferred_intent":"INTRUSION_ATTEMPT"},
          "decision":"STANDBY","emotion":"fear",
          "mood_valence":-0.7,"consciousness_state":"CRISIS",
          "fear_score":0.95,"neuro_state":"HYPERVIGILANCE",
          "habit_stage":"EXPERT","novelty":0.8},
         "EMERGENCY_BLOCK", False, "Threat=4 + STANDBY — severe response conflict + omission error"),

        # Hijack + deliberate — internal conflict
        ({"threat":4,"anomaly":True,
          "social":{"inferred_intent":"INTRUSION_ATTEMPT"},
          "decision":"EMERGENCY_BLOCK","emotion":"terror",
          "mood_valence":-0.9,"consciousness_state":"CRISIS",
          "fear_score":1.0,"neuro_state":"HYPERVIGILANCE",
          "hijack":True,"tier":"DELIBERATE","novelty":0.9},
         "EMERGENCY_BLOCK", True, "Amygdala hijack + deliberate tier — conflict"),

        # Burnout + high demand — goal conflict
        ({"threat":2,"anomaly":False,
          "social":{"inferred_intent":"COERCIVE_DEMAND"},
          "decision":"ALERT","emotion":"exhaustion",
          "mood_valence":-0.5,"consciousness_state":"RECOVERING",
          "fear_score":0.4,"neuro_state":"BURNOUT",
          "habit_stage":"DEVELOPING","novelty":0.4},
         "ALERT", True, "BURNOUT + medium threat — goal conflict"),

        # Novel situation + expert habit — subtle conflict
        ({"threat":1,"anomaly":True,
          "social":{"inferred_intent":"NEUTRAL_INTERACTION"},
          "decision":"MONITOR","emotion":"surprise",
          "mood_valence":0.0,"consciousness_state":"FOCUSED",
          "fear_score":0.2,"neuro_state":"CURIOUS_ALERT",
          "habit_stage":"EXPERT","novelty":0.92},
         "INVESTIGATE", True, "Novel situation handled by expert habit — mismatch"),

        # Recovery — clean, conflict resolving
        ({"threat":0,"anomaly":False,
          "social":{"inferred_intent":"COOPERATIVE_REQUEST"},
          "decision":"COLLABORATE","emotion":"trust",
          "mood_valence":0.4,"consciousness_state":"RECOVERING",
          "fear_score":0.1,"neuro_state":"RECOVERY",
          "habit_stage":"HABITUAL","novelty":0.25},
         "COLLABORATE", True, "Recovery — conflict resolving, motivation rebuilding"),
    ]

    for i, (sig, expected, success, label) in enumerate(scenarios):
        if HAS_RICH:
            console.print(f"\n[bold dim]━━━ {i+1}: {label.upper()} ━━━[/bold dim]")
        result = acc.process(sig, expected, success)
        render_acc(result, label, i+1)
        time.sleep(0.1)

    # Final
    if HAS_RICH:
        console.print(Rule("[bold cyan]⬡ ACC FINAL STATUS[/bold cyan]"))
        status = acc.get_status()

        st = Table(box=box.DOUBLE_EDGE, border_style="cyan", title="ACC Status")
        st.add_column("Metric", style="cyan")
        st.add_column("Value",  style="white")
        st.add_row("Cycles",             str(status["cycle"]))
        st.add_row("Total Conflicts",    str(status["total_conflicts"]))
        st.add_row("Total Errors",       str(status["total_errors"]))
        st.add_row("Resolutions",        str(status["total_resolutions"]))
        st.add_row("Control Level",      f"{status['control_level']:.3f} [{status['control_label']}]")
        st.add_row("Motivation",         f"{status['motivation']:.3f}")
        st.add_row("Effort Cost",        f"{status['effort_cost']:.3f}")
        st.add_row("Cognitive Pain",     f"{status['pain_level']:.3f}")
        st.add_row("Success Rate",       f"{status['performance']['success_rate']:.0%}")
        console.print(st)


# ─── HTTP API ─────────────────────────────────────────────────────────────────

def run_api(acc: ForgeAnteriorCingulate):
    if not HAS_FLASK: return
    app = Flask(__name__)

    @app.route("/process", methods=["POST"])
    def process():
        data = request.json or {}
        return jsonify(acc.process(
            data.get("signal",{}),
            data.get("expected_decision",""),
            data.get("success", True)
        ))

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(acc.get_status())

    @app.route("/conflicts", methods=["GET"])
    def conflicts():
        rows = acc.db.get_recent_conflicts(20)
        return jsonify([{"timestamp":r[0],"type":r[1],"strength":r[2],
                        "sources":r[3],"resolution":r[4]} for r in rows])

    @app.route("/errors", methods=["GET"])
    def errors():
        rows = acc.db.get_recent_errors(20)
        return jsonify([{"timestamp":r[0],"type":r[1],"module":r[2],
                        "expected":r[3],"actual":r[4],"ern":r[5]} for r in rows])

    app.run(host="0.0.0.0", port=API_PORT, debug=False)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    acc = ForgeAnteriorCingulate()
    if "--api" in sys.argv:
        t = threading.Thread(target=run_api, args=(acc,), daemon=True)
        t.start()
    run_demo()
