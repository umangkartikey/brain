"""
FORGE Insula — forge_insula.py
================================
AI analog of the brain's insular cortex.

The insula is the brain's interoceptive hub — the region that
monitors the body's internal state and translates raw physiological
signals into felt experience. It answers the question the rest of
the brain keeps asking: "How are we doing in here?"

It doesn't process the outside world. It processes the INSIDE.

Key insight: The insula is why decisions feel different when you're
tired vs rested, hungry vs fed, overwhelmed vs calm. It's the
biological substrate of "gut feeling" — not metaphor, but literal
visceral state feeding into cognition.

In FORGE, the insula monitors the system's internal computational
state: processing load, memory pressure, cycle latency, module
health, resource saturation. It translates these into an
interoceptive signal that biases every other module.

Four core functions:

  1. INTEROCEPTION (internal state sensing)
     Continuously polls system-level signals:
     CPU-analog load, memory pressure, latency drift,
     error accumulation, thermal analog (sustained high load).
     Maps these to a felt internal state.

  2. BODY BUDGET TRACKING (allostatic load)
     The brain doesn't just react to current state —
     it maintains a running prediction of resource needs.
     "Body budget" = predicted cost vs available resources.
     When budget is deficit: fatigue, reduced risk tolerance.
     When budget is surplus: confidence, expanded action space.

  3. VISCERAL SIGNAL GENERATION
     Translates internal state into signals other modules use:
     - Cognitive load → attention narrowing
     - Resource fatigue → conservative bias
     - Thermal load → error rate increase
     - Recovery state → expanded creativity window

  4. DISGUST / WRONGNESS DETECTION
     The insula processes more than physical state.
     It generates the feeling that something is "off" —
     moral disgust, aesthetic wrongness, conceptual mismatch.
     In FORGE: flags inputs that are internally inconsistent
     even when surface features appear normal.

Architecture:
  InteroceptiveSensor  → polls internal system metrics
  BodyBudget           → allostatic load tracker
  ViscerlaSignalMapper → state → felt signal translation
  DiscomfortDetector   → wrongness / disgust flagging
  InsulaClock          → tracks rhythmic internal cycles
  InsulaOutput         → projects to ACC, amygdala, prefrontal
"""

import json
import time
import uuid
import sqlite3
import threading
import math
import random
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
API_PORT = 7799
VERSION  = "1.0.0"

# Body budget
BUDGET_MAX          = 1.0
BUDGET_MIN          = 0.0
BUDGET_RECOVERY     = 0.04   # per cycle when load is low
BUDGET_BURN_RATE    = 0.06   # per cycle under high load
BUDGET_ALARM        = 0.25   # deficit threshold

# Thermal analog (sustained high load)
THERMAL_RISE        = 0.08
THERMAL_COOL        = 0.05
THERMAL_MAX         = 1.0
THERMAL_ALARM       = 0.70

# Load thresholds
LOAD_LOW            = 0.30
LOAD_MODERATE       = 0.55
LOAD_HIGH           = 0.75
LOAD_CRITICAL       = 0.90

# Disgust / wrongness
WRONGNESS_THRESHOLD = 0.50

# Rolling window
HISTORY_WINDOW      = 25

console = Console() if HAS_RICH else None

# ─── Enums ────────────────────────────────────────────────────────────────────

class InteroceptiveState(Enum):
    VITAL       = "VITAL"       # budget full, load low — optimal
    COMFORTABLE = "COMFORTABLE" # budget good, moderate load
    TAXED       = "TAXED"       # budget draining, high load
    FATIGUED    = "FATIGUED"    # budget low — reduced capacity
    DEPLETED    = "DEPLETED"    # budget critical — conservative mode
    OVERHEATED  = "OVERHEATED"  # thermal alarm — error risk high
    RECOVERING  = "RECOVERING"  # post-stress recovery window

class ViscerlaSignal(Enum):
    EXPAND     = "EXPAND"       # surplus — broaden action space
    MAINTAIN   = "MAINTAIN"     # balanced — continue current
    CONSERVE   = "CONSERVE"     # mild deficit — reduce load
    CONTRACT   = "CONTRACT"     # significant deficit — narrow focus
    EMERGENCY  = "EMERGENCY"    # critical — survival functions only

class WrongnessType(Enum):
    NONE          = "NONE"
    INCONSISTENCY = "INCONSISTENCY"   # signal internally contradicts itself
    OVERLOAD      = "OVERLOAD"        # too many simultaneous demands
    MORAL_VALENCE = "MORAL_VALENCE"   # ethics flag from pattern analysis
    AESTHETIC     = "AESTHETIC"       # structural wrongness in input
    VISCERAL      = "VISCERAL"        # gut-level signal, source unclear

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class InteroceptiveSample:
    """One snapshot of the system's internal state."""
    id:             str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:      str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:          int   = 0
    load:           float = 0.0    # computational load analog 0-1
    memory_pressure:float = 0.0    # memory usage analog 0-1
    latency:        float = 0.0    # processing latency analog 0-1
    error_density:  float = 0.0    # recent error rate 0-1
    thermal:        float = 0.0    # thermal / sustained-load analog 0-1
    body_budget:    float = 1.0    # allostatic reserve 0-1
    state:          str   = InteroceptiveState.COMFORTABLE.value
    visceral_signal:str   = ViscerlaSignal.MAINTAIN.value
    wrongness:      float = 0.0

@dataclass
class BodyBudgetEvent:
    """A significant change in the body budget."""
    id:          str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:   str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:       int   = 0
    budget_before: float = 0.0
    budget_after:  float = 0.0
    delta:         float = 0.0
    cause:         str   = ""
    alarm:         bool  = False

@dataclass
class WrongnessEvent:
    """A detected wrongness / disgust signal."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    wrongness_type: str = WrongnessType.NONE.value
    magnitude:    float = 0.0
    source:       str   = ""
    description:  str   = ""

# ─── Database ─────────────────────────────────────────────────────────────────

class InsulaDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS interoceptive_samples (
                    id TEXT PRIMARY KEY, timestamp TEXT, cycle INTEGER,
                    load REAL, memory_pressure REAL, latency REAL,
                    error_density REAL, thermal REAL, body_budget REAL,
                    state TEXT, visceral_signal TEXT, wrongness REAL
                );
                CREATE TABLE IF NOT EXISTS budget_events (
                    id TEXT PRIMARY KEY, timestamp TEXT, cycle INTEGER,
                    budget_before REAL, budget_after REAL, delta REAL,
                    cause TEXT, alarm INTEGER
                );
                CREATE TABLE IF NOT EXISTS wrongness_events (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    wrongness_type TEXT, magnitude REAL,
                    source TEXT, description TEXT
                );
                CREATE TABLE IF NOT EXISTS state_transitions (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    from_state TEXT, to_state TEXT,
                    trigger TEXT, cycle INTEGER
                );
            """)
            self.conn.commit()

    def save_sample(self, s: InteroceptiveSample):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO interoceptive_samples VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (s.id, s.timestamp, s.cycle, s.load, s.memory_pressure,
                  s.latency, s.error_density, s.thermal, s.body_budget,
                  s.state, s.visceral_signal, s.wrongness))
            self.conn.commit()

    def save_budget_event(self, e: BodyBudgetEvent):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO budget_events VALUES (?,?,?,?,?,?,?,?)
            """, (e.id, e.timestamp, e.cycle, e.budget_before,
                  e.budget_after, e.delta, e.cause, int(e.alarm)))
            self.conn.commit()

    def save_wrongness(self, w: WrongnessEvent):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO wrongness_events VALUES (?,?,?,?,?,?)
            """, (w.id, w.timestamp, w.wrongness_type, w.magnitude,
                  w.source, w.description))
            self.conn.commit()

    def log_transition(self, from_state: str, to_state: str,
                       trigger: str, cycle: int):
        with self.lock:
            self.conn.execute("""
                INSERT INTO state_transitions VALUES (?,?,?,?,?,?)
            """, (str(uuid.uuid4())[:8], datetime.now().isoformat(),
                  from_state, to_state, trigger, cycle))
            self.conn.commit()

    def get_recent_samples(self, limit=20):
        with self.lock:
            return self.conn.execute("""
                SELECT cycle, load, thermal, body_budget, state,
                       visceral_signal, wrongness
                FROM interoceptive_samples
                ORDER BY cycle DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_budget_history(self, limit=20):
        with self.lock:
            return self.conn.execute("""
                SELECT cycle, budget_before, budget_after, delta, cause, alarm
                FROM budget_events ORDER BY cycle DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_wrongness_events(self, limit=10):
        with self.lock:
            return self.conn.execute("""
                SELECT timestamp, wrongness_type, magnitude, source, description
                FROM wrongness_events ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()


# ─── Interoceptive Sensor ─────────────────────────────────────────────────────

class InteroceptiveSensor:
    """
    Polls internal system metrics and maps them to
    a unified load signal (0-1).

    In the biological insula, this is afferent signals
    from visceral organs — heart rate, gut activity,
    muscle tension, respiratory rate.

    In FORGE, these are computational analogs:
    module response times, error rates, queue depths,
    memory usage, cycle frequency.
    """

    def sense(self, signal: dict, acc_health: float,
              amygdala_fear: float, cycle: int) -> tuple[float, float, float, float]:
        """
        Returns (load, memory_pressure, latency, error_density).
        All values 0-1.
        """
        # Load — derived from threat level, anomaly, module count
        threat = signal.get("threat", 0)
        anomaly = signal.get("anomaly", False)
        module_count = signal.get("active_modules", 10)
        processing_ms = signal.get("processing_ms", 20.0)

        load = (
            (threat / 4.0) * 0.35
            + (0.3 if anomaly else 0.0)
            + min(0.15, module_count / 100.0)
            + (amygdala_fear * 0.20)
        )
        load = round(min(1.0, load), 4)

        # Memory pressure — from confidence inversely
        confidence = signal.get("confidence", 0.7)
        memory_pressure = round(min(1.0, (1.0 - confidence) * 0.6
                                    + (1.0 - acc_health) * 0.4), 4)

        # Latency — from processing time analog
        latency = round(min(1.0, processing_ms / 100.0), 4)

        # Error density — from ACC health and ERN signal
        ern = signal.get("ern_signal", 0.0)
        error_density = round(min(1.0, (1.0 - acc_health) * 0.5 + ern * 0.5), 4)

        return load, memory_pressure, latency, error_density


# ─── Body Budget ─────────────────────────────────────────────────────────────

class BodyBudget:
    """
    Tracks the system's allostatic reserve.

    Allostasis = maintaining stability through change.
    The body budget is the brain's running ledger of
    "how much resource do I have vs how much am I spending?"

    High budget → confidence, risk tolerance, creativity
    Low budget  → conservatism, narrow focus, fatigue
    Critical    → emergency conservation mode

    Budget recovers slowly when load is low.
    Budget burns faster under sustained high load.
    Sudden threat causes a sharp budget draw.
    """

    def __init__(self, db: InsulaDB):
        self.db     = db
        self.budget = 0.75   # start at 75% — not fresh, not depleted
        self.cycle  = 0

    def update(self, load: float, thermal: float,
               threat: int, cycle: int) -> tuple[float, Optional[BodyBudgetEvent]]:
        """
        Update budget based on current load and threat.
        Returns (new_budget, event_if_significant).
        """
        prev = self.budget
        self.cycle = cycle

        if load < LOAD_LOW and thermal < 0.3:
            # Recovery — low load, cool system
            delta = BUDGET_RECOVERY * (1.0 - self.budget)
            self.budget = round(min(BUDGET_MAX, self.budget + delta), 4)
            cause = "recovery"
        elif load > LOAD_HIGH or thermal > THERMAL_ALARM:
            # High expenditure
            burn = BUDGET_BURN_RATE * load
            if thermal > THERMAL_ALARM:
                burn *= 1.3   # thermal stress burns faster
            self.budget = round(max(BUDGET_MIN, self.budget - burn), 4)
            cause = f"high_load={load:.2f}"
        elif threat >= 3:
            # Acute threat — sharp draw
            draw = 0.08 + (threat - 2) * 0.04
            self.budget = round(max(BUDGET_MIN, self.budget - draw), 4)
            cause = f"threat={threat}"
        else:
            # Moderate load — slow burn
            burn = BUDGET_BURN_RATE * 0.3 * load
            self.budget = round(max(BUDGET_MIN, self.budget - burn), 4)
            cause = "moderate_load"

        delta_val = round(self.budget - prev, 4)
        alarm     = self.budget < BUDGET_ALARM

        # Log significant changes
        event = None
        if abs(delta_val) > 0.03 or alarm:
            event = BodyBudgetEvent(
                cycle        = cycle,
                budget_before= prev,
                budget_after = self.budget,
                delta        = delta_val,
                cause        = cause,
                alarm        = alarm
            )
            self.db.save_budget_event(event)

        return self.budget, event

    def surplus(self) -> float:
        """How far above the alarm threshold."""
        return round(max(0.0, self.budget - BUDGET_ALARM), 4)

    def deficit(self) -> float:
        """How far below 0.5 (neutral)."""
        return round(max(0.0, 0.5 - self.budget), 4)


# ─── Thermal Tracker ─────────────────────────────────────────────────────────

class ThermalTracker:
    """
    Models sustained high-load heating.

    Biological analog: cortisol buildup, neuroinflammation,
    synaptic fatigue under prolonged stress.

    In FORGE: thermal = accumulated cost of sustained high load.
    It rises faster than it falls.
    Once hot, errors increase and budget burns faster.
    Cooling requires genuine low-load periods.
    """

    def __init__(self):
        self.thermal   = 0.0
        self.peak      = 0.0
        self.hot_cycles= 0

    def update(self, load: float) -> float:
        if load > LOAD_HIGH:
            self.thermal = round(min(THERMAL_MAX,
                                     self.thermal + THERMAL_RISE * load), 4)
            self.hot_cycles += 1
        else:
            cool = THERMAL_COOL * (1.0 + (1.0 - load))
            self.thermal = round(max(0.0, self.thermal - cool), 4)

        self.peak = max(self.peak, self.thermal)
        return self.thermal

    def alarm(self) -> bool:
        return self.thermal >= THERMAL_ALARM

    def error_amplifier(self) -> float:
        """Thermal stress increases error probability."""
        return round(1.0 + self.thermal * 0.4, 3)


# ─── Visceral Signal Mapper ───────────────────────────────────────────────────

class VisceralSignalMapper:
    """
    Maps internal state metrics to a felt visceral signal.

    This is the translation layer between raw metrics
    and meaningful internal experience.

    The insula doesn't report "load=0.82" to the rest of the brain.
    It reports "CONTRACT" — a signal that compresses
    into the entire cognitive architecture as a felt pressure
    to narrow, slow down, be careful.
    """

    def map_state(self, load: float, budget: float,
                  thermal: float, error_density: float) -> tuple[InteroceptiveState, ViscerlaSignal]:

        # Determine interoceptive state
        if thermal >= THERMAL_ALARM:
            state = InteroceptiveState.OVERHEATED
        elif budget < 0.10:
            state = InteroceptiveState.DEPLETED
        elif budget < BUDGET_ALARM:
            state = InteroceptiveState.FATIGUED
        elif load > LOAD_HIGH and budget < 0.5:
            state = InteroceptiveState.TAXED
        elif budget > 0.7 and load < LOAD_LOW:
            state = InteroceptiveState.VITAL
        elif load < LOAD_MODERATE and budget > 0.5:
            state = InteroceptiveState.COMFORTABLE
        else:
            # Check if recovering (budget rising from low point)
            state = InteroceptiveState.RECOVERING

        # Map state to visceral signal
        signal_map = {
            InteroceptiveState.VITAL:       ViscerlaSignal.EXPAND,
            InteroceptiveState.COMFORTABLE: ViscerlaSignal.MAINTAIN,
            InteroceptiveState.RECOVERING:  ViscerlaSignal.MAINTAIN,
            InteroceptiveState.TAXED:       ViscerlaSignal.CONSERVE,
            InteroceptiveState.FATIGUED:    ViscerlaSignal.CONTRACT,
            InteroceptiveState.DEPLETED:    ViscerlaSignal.EMERGENCY,
            InteroceptiveState.OVERHEATED:  ViscerlaSignal.CONTRACT,
        }
        signal = signal_map.get(state, ViscerlaSignal.MAINTAIN)
        return state, signal

    def confidence_bias(self, state: InteroceptiveState) -> float:
        """
        How much does this internal state bias confidence?
        Positive = confidence boost, Negative = confidence reduction.
        """
        biases = {
            InteroceptiveState.VITAL:       +0.15,
            InteroceptiveState.COMFORTABLE: +0.05,
            InteroceptiveState.RECOVERING:   0.00,
            InteroceptiveState.TAXED:       -0.10,
            InteroceptiveState.FATIGUED:    -0.20,
            InteroceptiveState.DEPLETED:    -0.35,
            InteroceptiveState.OVERHEATED:  -0.25,
        }
        return biases.get(state, 0.0)

    def risk_tolerance(self, state: InteroceptiveState) -> float:
        """
        Risk tolerance biased by internal state. 0-1.
        High budget → willing to explore.
        Low budget  → conservative, avoid risk.
        """
        tolerances = {
            InteroceptiveState.VITAL:       0.80,
            InteroceptiveState.COMFORTABLE: 0.60,
            InteroceptiveState.RECOVERING:  0.45,
            InteroceptiveState.TAXED:       0.35,
            InteroceptiveState.FATIGUED:    0.20,
            InteroceptiveState.DEPLETED:    0.05,
            InteroceptiveState.OVERHEATED:  0.15,
        }
        return tolerances.get(state, 0.50)


# ─── Wrongness Detector ───────────────────────────────────────────────────────

class WrongnessDetector:
    """
    Detects signals that feel "off" even when metrics look normal.

    The insula's role in disgust and moral cognition:
    Something can be technically correct but feel wrong.
    Internal inconsistency, pattern violation, value conflict.

    In FORGE, wrongness is computed from:
    - Signal self-contradiction (high trust + high threat simultaneously)
    - Metric incoherence (high confidence + high error density)
    - Temporal anomaly (sudden state change with no causal chain)
    - Structural oddity (inputs that don't fit known patterns)
    """

    def __init__(self, db: InsulaDB):
        self.db = db
        self.wrongness_level = 0.0
        self.recent_signals: deque = deque(maxlen=5)

    def evaluate(self, signal: dict, load: float,
                 confidence: float, error_density: float,
                 threat: int) -> tuple[float, Optional[WrongnessEvent]]:
        """
        Returns (wrongness_score, event_if_above_threshold).
        """
        wrongness = 0.0
        wtype     = WrongnessType.NONE
        description = ""

        social = signal.get("social", {}) or {}
        trust  = social.get("trust_score", 0.5)

        # Contradiction: high trust + high threat
        if trust > 0.7 and threat >= 3:
            wrongness = max(wrongness, 0.65)
            wtype     = WrongnessType.INCONSISTENCY
            description = f"trust={trust:.2f} but threat={threat}"

        # Contradiction: high confidence + high error density
        if confidence > 0.75 and error_density > 0.60:
            wrongness = max(wrongness, 0.55)
            wtype     = WrongnessType.INCONSISTENCY
            description = f"confidence={confidence:.2f} but error_density={error_density:.2f}"

        # Overload pattern: all metrics simultaneously high
        if load > 0.8 and error_density > 0.5 and threat >= 2:
            wrongness = max(wrongness, 0.70)
            wtype     = WrongnessType.OVERLOAD
            description = f"simultaneous overload: load={load:.2f} errors={error_density:.2f} threat={threat}"

        # Temporal anomaly: sudden state reversal
        if len(self.recent_signals) >= 2:
            prev_threat = self.recent_signals[-1].get("threat", 0)
            if abs(threat - prev_threat) >= 3:
                wrongness = max(wrongness, 0.60)
                wtype     = WrongnessType.VISCERAL
                description = f"sudden state reversal: threat {prev_threat}→{threat}"

        # Store signal for temporal comparison
        self.recent_signals.append({"threat": threat, "load": load})

        # Decay previous wrongness
        self.wrongness_level = round(
            max(0.0, self.wrongness_level * 0.7 + wrongness * 0.3), 4
        )

        event = None
        if wrongness >= WRONGNESS_THRESHOLD:
            event = WrongnessEvent(
                wrongness_type = wtype.value,
                magnitude      = round(wrongness, 4),
                source         = "insula_detector",
                description    = description
            )
            self.db.save_wrongness(event)

        return round(wrongness, 4), event


# ─── Insula Clock ─────────────────────────────────────────────────────────────

class InsulaClock:
    """
    Tracks rhythmic internal cycles.

    The biological insula is sensitive to bodily rhythms —
    heartbeat, respiratory cycle, circadian timing.
    These rhythms create a temporal scaffolding that
    affects how the brain processes experience.

    In FORGE, the insula clock tracks:
    - Processing rhythm (consistent vs irregular)
    - Load oscillation (natural peaks and troughs)
    - Recovery windows (post-high-load rest periods)

    The clock's value: irregular processing rhythms
    are themselves a signal of system stress.
    """

    def __init__(self):
        self.cycle_times: deque  = deque(maxlen=10)
        self.last_cycle_time: float = time.time()
        self.rhythm_score: float    = 1.0   # 1.0 = perfectly regular

    def tick(self) -> float:
        """Record a cycle and compute rhythm regularity."""
        now      = time.time()
        interval = now - self.last_cycle_time
        self.last_cycle_time = now
        self.cycle_times.append(interval)

        if len(self.cycle_times) >= 4:
            times = list(self.cycle_times)
            mean  = sum(times) / len(times)
            variance = sum((t - mean) ** 2 for t in times) / len(times)
            cv    = math.sqrt(variance) / mean if mean > 0 else 0
            # High CV = irregular rhythm = stress signal
            self.rhythm_score = round(max(0.0, 1.0 - min(1.0, cv)), 4)

        return self.rhythm_score

    def is_recovery_window(self, load_history: deque) -> bool:
        """
        True if we're in a natural recovery window
        (load just dropped after a sustained high period).
        """
        if len(load_history) < 4:
            return False
        recent = list(load_history)[-4:]
        was_high = any(l > LOAD_HIGH for l in recent[:2])
        now_low  = all(l < LOAD_MODERATE for l in recent[2:])
        return was_high and now_low


# ─── Insula Output ────────────────────────────────────────────────────────────

class InsulaOutput:
    """
    Projects interoceptive state to downstream modules.

    The insula's projections in the biological brain:
    → anterior cingulate (conflict weighting)
    → amygdala (threat calibration)
    → prefrontal (decision bias)
    → hypothalamus (arousal regulation)
    → motor cortex (action readiness)

    In FORGE, insula output is a bias layer that
    modulates every other module's outputs — not by
    overriding them, but by tilting the playing field.
    """

    def compute(self, state: InteroceptiveState,
                visceral: ViscerlaSignal,
                budget: float,
                thermal: float,
                wrongness: float,
                rhythm: float) -> dict:

        mapper = VisceralSignalMapper()
        conf_bias     = mapper.confidence_bias(state)
        risk_tol      = mapper.risk_tolerance(state)

        # ACC weighting — low budget amplifies conflict sensitivity
        acc_weight = round(1.0 + max(0.0, 0.5 - budget) * 0.6, 4)

        # Amygdala calibration — fatigue lowers fear threshold
        amygdala_calibration = round(
            1.0 + (1.0 - budget) * 0.2 + thermal * 0.15, 4
        )

        # Prefrontal bias — depleted state shifts toward habit/default
        prefrontal_bias = "EXPLORATORY" if budget > 0.6 else \
                          "BALANCED"    if budget > 0.35 else \
                          "CONSERVATIVE"

        # Creative window — open only in vital or comfortable states
        creative_window = state in [
            InteroceptiveState.VITAL,
            InteroceptiveState.COMFORTABLE,
            InteroceptiveState.RECOVERING
        ]

        # NE request — taxed/overheated states need arousal boost
        ne_request = round(
            (thermal * 0.3 + max(0.0, 0.5 - budget) * 0.4)
            * (1.0 if state == InteroceptiveState.OVERHEATED else 0.7),
            4
        )

        return {
            "confidence_bias":     conf_bias,
            "risk_tolerance":      risk_tol,
            "acc_weight":          acc_weight,
            "amygdala_calibration":amygdala_calibration,
            "prefrontal_bias":     prefrontal_bias,
            "creative_window":     creative_window,
            "ne_request":          ne_request,
            "wrongness_flag":      wrongness >= WRONGNESS_THRESHOLD,
            "rhythm_score":        rhythm,
        }


# ─── FORGE Insula ─────────────────────────────────────────────────────────────

class ForgeInsula:
    def __init__(self):
        self.db          = InsulaDB()
        self.sensor      = InteroceptiveSensor()
        self.budget      = BodyBudget(self.db)
        self.thermal     = ThermalTracker()
        self.mapper      = VisceralSignalMapper()
        self.wrongness   = WrongnessDetector(self.db)
        self.clock       = InsulaClock()
        self.output_calc = InsulaOutput()
        self.cycle       = 0
        self.prev_state  = InteroceptiveState.COMFORTABLE

        self.load_history: deque = deque(maxlen=HISTORY_WINDOW)

        # Counters
        self.total_alarms     = 0
        self.total_overheats  = 0
        self.total_depletions = 0
        self.wrongness_events = 0

    def process(self, signal: dict,
                acc_health: float = 0.6,
                amygdala_fear: float = 0.1) -> dict:
        """
        Full insula processing pipeline.

        signal:        raw input signal
        acc_health:    rolling health from forge_anterior_cingulate
        amygdala_fear: current fear score from forge_amygdala
        """
        t0         = time.time()
        self.cycle += 1
        threat     = signal.get("threat", 0)
        confidence = signal.get("confidence", 0.7)

        # 1. Clock tick
        rhythm = self.clock.tick()

        # 2. Sense internal metrics
        load, memory_pressure, latency, error_density = self.sensor.sense(
            signal, acc_health, amygdala_fear, self.cycle
        )
        self.load_history.append(load)

        # 3. Thermal update
        thermal = self.thermal.update(load)

        # 4. Body budget update
        budget_val, budget_event = self.budget.update(load, thermal, threat, self.cycle)

        # 5. Map to interoceptive state + visceral signal
        state, visceral = self.mapper.map_state(load, budget_val, thermal, error_density)

        # 6. Log state transitions
        if state != self.prev_state:
            self.db.log_transition(
                self.prev_state.value, state.value,
                f"load={load:.2f},budget={budget_val:.2f}", self.cycle
            )
            self.prev_state = state

        # 7. Wrongness detection
        wrongness_score, wrongness_event = self.wrongness.evaluate(
            signal, load, confidence, error_density, threat
        )

        # 8. Compute output projections
        output = self.output_calc.compute(
            state, visceral, budget_val, thermal,
            wrongness_score, rhythm
        )

        # 9. Save sample
        recovery_window = self.clock.is_recovery_window(self.load_history)
        sample = InteroceptiveSample(
            cycle           = self.cycle,
            load            = load,
            memory_pressure = memory_pressure,
            latency         = latency,
            error_density   = error_density,
            thermal         = thermal,
            body_budget     = budget_val,
            state           = state.value,
            visceral_signal = visceral.value,
            wrongness       = wrongness_score
        )
        self.db.save_sample(sample)

        # 10. Track alarms
        if budget_val < BUDGET_ALARM:
            self.total_alarms += 1
        if thermal >= THERMAL_ALARM:
            self.total_overheats += 1
        if state == InteroceptiveState.DEPLETED:
            self.total_depletions += 1
        if wrongness_event:
            self.wrongness_events += 1

        elapsed = (time.time() - t0) * 1000

        return {
            "cycle":            self.cycle,
            "state":            state.value,
            "visceral_signal":  visceral.value,
            "load":             load,
            "memory_pressure":  memory_pressure,
            "latency":          latency,
            "error_density":    error_density,
            "thermal":          thermal,
            "thermal_alarm":    self.thermal.alarm(),
            "body_budget":      budget_val,
            "budget_alarm":     budget_val < BUDGET_ALARM,
            "budget_event":     budget_event.cause if budget_event else "",
            "wrongness":        wrongness_score,
            "wrongness_type":   wrongness_event.wrongness_type if wrongness_event else WrongnessType.NONE.value,
            "wrongness_flag":   wrongness_event is not None,
            "rhythm_score":     rhythm,
            "recovery_window":  recovery_window,
            "output":           output,
            "processing_ms":    round(elapsed, 2),
            "total_alarms":     self.total_alarms,
            "total_overheats":  self.total_overheats,
            "wrongness_events": self.wrongness_events,
        }

    def get_status(self) -> dict:
        recent = self.db.get_recent_samples(5)
        return {
            "version":           VERSION,
            "cycle":             self.cycle,
            "current_state":     self.prev_state.value,
            "body_budget":       round(self.budget.budget, 4),
            "thermal":           round(self.thermal.thermal, 4),
            "thermal_peak":      round(self.thermal.peak, 4),
            "hot_cycles":        self.thermal.hot_cycles,
            "wrongness_level":   round(self.wrongness.wrongness_level, 4),
            "rhythm_score":      round(self.clock.rhythm_score, 4),
            "total_alarms":      self.total_alarms,
            "total_overheats":   self.total_overheats,
            "total_depletions":  self.total_depletions,
            "wrongness_events":  self.wrongness_events,
            "recent_states":     [r[4] for r in recent],
        }


# ─── Rich UI ──────────────────────────────────────────────────────────────────

STATE_COLORS = {
    "VITAL":       "bright_green",
    "COMFORTABLE": "green",
    "RECOVERING":  "cyan",
    "TAXED":       "yellow",
    "FATIGUED":    "red",
    "DEPLETED":    "bright_red",
    "OVERHEATED":  "orange3",
}

SIGNAL_COLORS = {
    "EXPAND":    "bright_green",
    "MAINTAIN":  "green",
    "CONSERVE":  "yellow",
    "CONTRACT":  "red",
    "EMERGENCY": "bright_red",
}

def render_insula(result: dict, signal: dict, idx: int):
    if not HAS_RICH: return

    state    = result["state"]
    sc       = STATE_COLORS.get(state, "white")
    visceral = result["visceral_signal"]
    vc       = SIGNAL_COLORS.get(visceral, "white")
    budget   = result["body_budget"]
    bc       = "bright_green" if budget > 0.6 else "yellow" if budget > 0.3 else "bright_red"
    thermal  = result["thermal"]
    tc       = "orange3" if thermal > THERMAL_ALARM else "yellow" if thermal > 0.4 else "dim"

    console.print(Rule(
        f"[bold cyan]⬡ INSULA[/bold cyan]  "
        f"[dim]#{idx}[/dim]  "
        f"[{sc}]{state}[/{sc}]  "
        f"[{vc}]{visceral}[/{vc}]  "
        f"[dim]budget={budget:.2f}  thermal={thermal:.2f}[/dim]"
    ))

    # Alarm banners
    if result["state"] == "DEPLETED":
        console.print(Panel(
            f"[bold bright_red]⚡ BODY BUDGET CRITICAL — EMERGENCY CONSERVATION[/bold bright_red]\n"
            f"Budget: {budget:.3f}  |  Thermal: {thermal:.3f}\n"
            f"[dim]System entering emergency conservative mode.[/dim]",
            border_style="bright_red"
        ))
    elif result["thermal_alarm"]:
        console.print(Panel(
            f"[bold orange3]🌡  THERMAL ALARM — SUSTAINED OVERLOAD[/bold orange3]\n"
            f"Thermal: {thermal:.3f}  |  Hot cycles: {result.get('total_overheats',0)}\n"
            f"[dim]Error rate elevated. Load reduction recommended.[/dim]",
            border_style="orange3"
        ))

    if result["wrongness_flag"]:
        console.print(Panel(
            f"[bold magenta]⚠  WRONGNESS DETECTED — {result['wrongness_type']}[/bold magenta]\n"
            f"Magnitude: {result['wrongness']:.3f}\n"
            f"[dim]Interoceptive mismatch flagged to ACC.[/dim]",
            border_style="magenta"
        ))

    # Metric bars
    def bar(val, width=12):
        filled = int(val * width)
        return "█" * filled + "░" * (width - filled)

    load = result["load"]
    lc   = "bright_red" if load > LOAD_HIGH else "yellow" if load > LOAD_MODERATE else "green"

    left_lines = [
        f"[bold]Load:    [/bold] [{lc}]{bar(load)} {load:.3f}[/{lc}]",
        f"[bold]Budget:  [/bold] [{bc}]{bar(budget)} {budget:.3f}[/{bc}]",
        f"[bold]Thermal: [/bold] [{tc}]{bar(thermal)} {thermal:.3f}[/{tc}]",
        f"[bold]Errors:  [/bold] {bar(result['error_density'])} {result['error_density']:.3f}",
        f"[bold]Memory:  [/bold] {bar(result['memory_pressure'])} {result['memory_pressure']:.3f}",
        f"",
        f"[bold]Rhythm:[/bold]   {result['rhythm_score']:.3f}",
        f"[bold]Recovery:[/bold] {'[cyan]YES[/cyan]' if result['recovery_window'] else '[dim]no[/dim]'}",
    ]

    out = result["output"]
    right_lines = [
        f"[bold]→ Prefrontal:[/bold]  {out['prefrontal_bias']}",
        f"[bold]→ Risk tol: [/bold]   {out['risk_tolerance']:.3f}",
        f"[bold]→ Conf bias:[/bold]   {out['confidence_bias']:+.3f}",
        f"[bold]→ ACC weight:[/bold]  {out['acc_weight']:.3f}",
        f"[bold]→ Amyg calib:[/bold] {out['amygdala_calibration']:.3f}",
        f"[bold]→ NE request:[/bold] {out['ne_request']:.3f}",
        f"[bold]→ Creative:  [/bold]  {'[green]OPEN[/green]' if out['creative_window'] else '[dim]closed[/dim]'}",
        f"[bold]→ Wrongness: [/bold]  {'[magenta]FLAG[/magenta]' if out['wrongness_flag'] else '[dim]clear[/dim]'}",
    ]

    console.print(Columns([
        Panel("\n".join(left_lines),  title=f"[bold {sc}]Internal State[/bold {sc}]", border_style=sc),
        Panel("\n".join(right_lines), title="[bold]Output Projections[/bold]",        border_style="dim")
    ]))


def run_demo():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FORGE INSULA[/bold cyan]\n"
            "[dim]Interoception · Body Budget · Visceral Signal · Wrongness Detection[/dim]\n"
            f"[dim]Version {VERSION}  |  Port {API_PORT}[/dim]",
            border_style="cyan"
        ))

    insula = ForgeInsula()

    scenarios = [
        # Fresh system — low load, good budget
        ({"threat":0,"anomaly":False,"confidence":0.85,"active_modules":8,
          "processing_ms":15,"social":{"trust_score":0.9}},
         0.80, 0.05, "Fresh start — optimal internal state"),

        # Moderate work — budget starts to spend
        ({"threat":1,"anomaly":False,"confidence":0.70,"active_modules":15,
          "processing_ms":30,"social":{"trust_score":0.7}},
         0.65, 0.15, "Moderate load — budget draining slowly"),

        # Sudden threat — sharp budget draw
        ({"threat":3,"anomaly":True,"confidence":0.45,"active_modules":20,
          "processing_ms":55,"social":{"trust_score":0.3}},
         0.40, 0.65, "Sudden threat — sharp budget draw"),

        # Sustained high load — thermal rising
        ({"threat":2,"anomaly":True,"confidence":0.40,"active_modules":25,
          "processing_ms":70,"social":{"trust_score":0.4}},
         0.35, 0.50, "Sustained high load — thermal climbing"),

        # Wrongness: high trust + high threat contradiction
        ({"threat":3,"anomaly":False,"confidence":0.80,"active_modules":18,
          "processing_ms":40,"social":{"trust_score":0.85}},
         0.45, 0.40, "Wrongness: trusted entity triggering threat"),

        # Thermal alarm — overheated
        ({"threat":2,"anomaly":True,"confidence":0.35,"active_modules":30,
          "processing_ms":90,"social":{"trust_score":0.2}},
         0.25, 0.70, "Thermal alarm — system overheated"),

        # Critical depletion
        ({"threat":3,"anomaly":True,"confidence":0.25,"active_modules":35,
          "processing_ms":95,"social":{"trust_score":0.1}},
         0.15, 0.85, "Critical depletion — emergency mode"),

        # Recovery begins
        ({"threat":0,"anomaly":False,"confidence":0.70,"active_modules":10,
          "processing_ms":20,"social":{"trust_score":0.8}},
         0.55, 0.10, "Recovery — load drops, budget begins refilling"),

        # Full recovery
        ({"threat":0,"anomaly":False,"confidence":0.85,"active_modules":8,
          "processing_ms":12,"social":{"trust_score":0.9}},
         0.80, 0.05, "Full recovery — vital state restored"),
    ]

    for i, (sig, acc_h, amyg_f, label) in enumerate(scenarios):
        if HAS_RICH:
            console.print(f"\n[bold dim]━━━ {i+1}: {label.upper()} ━━━[/bold dim]")
        result = insula.process(sig, acc_h, amyg_f)
        render_insula(result, sig, i+1)
        time.sleep(0.08)

    # Final status
    if HAS_RICH:
        console.print(Rule("[bold cyan]⬡ INSULA FINAL STATUS[/bold cyan]"))
        status = insula.get_status()

        st = Table(box=box.DOUBLE_EDGE, border_style="cyan", title="Insula Status")
        st.add_column("Metric",  style="cyan")
        st.add_column("Value",   style="white")
        st.add_row("Cycles",         str(status["cycle"]))
        st.add_row("Current State",  status["current_state"])
        st.add_row("Body Budget",    f"{status['body_budget']:.3f}")
        st.add_row("Thermal",        f"{status['thermal']:.3f}")
        st.add_row("Thermal Peak",   f"{status['thermal_peak']:.3f}")
        st.add_row("Hot Cycles",     str(status["hot_cycles"]))
        st.add_row("Rhythm Score",   f"{status['rhythm_score']:.3f}")
        st.add_row("Budget Alarms",  str(status["total_alarms"]))
        st.add_row("Overheats",      str(status["total_overheats"]))
        st.add_row("Depletions",     str(status["total_depletions"]))
        st.add_row("Wrongness Events",str(status["wrongness_events"]))
        console.print(st)

        if status["recent_states"]:
            console.print(Rule("[dim]Recent State History[/dim]"))
            for s in reversed(status["recent_states"]):
                sc2 = STATE_COLORS.get(s, "white")
                console.print(f"  [{sc2}]{s}[/{sc2}]")


# ─── HTTP API ─────────────────────────────────────────────────────────────────

def run_api(insula: ForgeInsula):
    if not HAS_FLASK: return
    app = Flask(__name__)

    @app.route("/process", methods=["POST"])
    def process():
        body   = request.json or {}
        signal = body.get("signal", {})
        acc_h  = body.get("acc_health", 0.6)
        amyg_f = body.get("amygdala_fear", 0.1)
        return jsonify(insula.process(signal, acc_h, amyg_f))

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(insula.get_status())

    @app.route("/samples", methods=["GET"])
    def samples():
        rows = insula.db.get_recent_samples(20)
        return jsonify([{
            "cycle": r[0], "load": r[1], "thermal": r[2],
            "budget": r[3], "state": r[4],
            "visceral": r[5], "wrongness": r[6]
        } for r in rows])

    @app.route("/budget", methods=["GET"])
    def budget():
        rows = insula.db.get_budget_history(20)
        return jsonify([{
            "cycle": r[0], "before": r[1], "after": r[2],
            "delta": r[3], "cause": r[4], "alarm": bool(r[5])
        } for r in rows])

    @app.route("/wrongness", methods=["GET"])
    def wrongness():
        rows = insula.db.get_wrongness_events(10)
        return jsonify([{
            "timestamp": r[0], "type": r[1], "magnitude": r[2],
            "source": r[3], "description": r[4]
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
