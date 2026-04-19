"""
FORGE Frontoparietal Network — forge_frontoparietal.py
=======================================================
AI analog of the brain's frontoparietal network (FPN).

The frontoparietal network is the brain's DYNAMIC CONFIGURATOR.
While the thalamus gates signals, the FPN goes deeper —
it restructures the entire processing pipeline based on
what the current task demands.

Key insight: The brain is not a fixed pipeline.
It reconfigures itself for every task type.
Reading a poem activates different networks than
dodging a car. The FPN orchestrates this reconfiguration.

Four core functions:

  1. TASK-SET RECONFIGURATION
     Rapidly switches the brain's configuration
     between different task modes.
     "We're now doing threat assessment — route everything
     through salience→amygdala→prefrontal."
     "We're now doing creative reflection — activate DMN,
     suppress sensorimotor, amplify hippocampus."

  2. WORKING MEMORY MAINTENANCE
     Holds the current task goal in active memory
     while all processing happens around it.
     The "what are we doing right now?" anchor.

  3. COGNITIVE FLEXIBILITY
     Detects when the current task set is no longer
     appropriate and triggers reconfiguration.
     "The threat is resolved — switch from crisis mode
     to recovery mode."

  4. TOP-DOWN ATTENTION CONTROL
     Sends goal-directed attention signals to all modules.
     Different from salience (bottom-up) — this is
     INTENTIONAL attention allocation.
     "Pay more attention to social signals right now."
     "De-prioritize DMN — we need to focus."

Architecture:
  TaskSetManager       → maintains current task configuration
  PipelineConfigurator → rewires module routing dynamically
  WorkingMemory        → holds active task context
  FlexibilityDetector  → detects when to switch task sets
  AttentionController  → top-down attention allocation
  ReconfigurationEngine→ executes pipeline rewiring
  TaskLibrary          → library of known task configurations
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

DB_PATH  = "forge_frontoparietal.db"
API_PORT = 7796
VERSION  = "1.0.0"

# Reconfiguration thresholds
RECONFIG_THRESHOLD   = 0.65   # mismatch score to trigger reconfiguration
FLEXIBILITY_WINDOW   = 5      # cycles to assess flexibility need
WORKING_MEM_CAPACITY = 7      # max items in working memory (Miller's law)

# Attention weights
MAX_ATTENTION_BOOST  = 2.0    # maximum module gain multiplier
MIN_ATTENTION_WEIGHT = 0.1    # minimum module gain (never fully off)

# Task stability
STABILITY_THRESHOLD  = 0.70   # how stable task set must be before switching
SWITCH_COST_MS       = 25.0   # cognitive cost of switching task sets

console = Console() if HAS_RICH else None

# ─── Enums ────────────────────────────────────────────────────────────────────

class TaskMode(Enum):
    THREAT_RESPONSE    = "THREAT_RESPONSE"     # crisis, danger
    SOCIAL_ENGAGEMENT  = "SOCIAL_ENGAGEMENT"   # cooperative interaction
    LEARNING           = "LEARNING"            # novel information acquisition
    MEMORY_RETRIEVAL   = "MEMORY_RETRIEVAL"    # recalling past episodes
    CREATIVE_REFLECTION= "CREATIVE_REFLECTION" # DMN-dominant, idle
    ROUTINE_MONITORING = "ROUTINE_MONITORING"  # low-load baseline
    CONFLICT_RESOLUTION= "CONFLICT_RESOLUTION" # resolving contradictions
    RECOVERY           = "RECOVERY"            # post-stress restoration
    DEEP_FOCUS         = "DEEP_FOCUS"          # sustained single-task

class ReconfigReason(Enum):
    THREAT_CHANGE      = "THREAT_CHANGE"
    TASK_COMPLETE      = "TASK_COMPLETE"
    CONTEXT_SHIFT      = "CONTEXT_SHIFT"
    PERFORMANCE_DROP   = "PERFORMANCE_DROP"
    EXTERNAL_TRIGGER   = "EXTERNAL_TRIGGER"
    SCHEDULED          = "SCHEDULED"
    FLEXIBILITY_PROBE  = "FLEXIBILITY_PROBE"

class AttentionMode(Enum):
    BROAD      = "BROAD"       # wide attention — catch everything
    FOCUSED    = "FOCUSED"     # narrow — one target
    DIVIDED    = "DIVIDED"     # split across multiple targets
    SUSTAINED  = "SUSTAINED"   # maintain over time
    EXECUTIVE  = "EXECUTIVE"   # top-down goal-directed

class WorkingMemoryStatus(Enum):
    CLEAR      = "CLEAR"       # plenty of capacity
    MODERATE   = "MODERATE"    # moderate load
    FULL       = "FULL"        # at capacity
    OVERLOADED = "OVERLOADED"  # exceeding capacity — performance drops

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class TaskSet:
    """A complete configuration of the cognitive pipeline for a task type."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mode:         str   = TaskMode.ROUTINE_MONITORING.value
    name:         str   = ""
    description:  str   = ""
    # Module attention weights (gain multipliers)
    module_weights: dict = field(default_factory=dict)
    # Pipeline routing order
    pipeline_order: list = field(default_factory=list)
    # Which modules are active vs suppressed
    active_modules: list = field(default_factory=list)
    suppressed_modules: list = field(default_factory=list)
    # Timing adjustments
    timing_adjustments: dict = field(default_factory=dict)
    # Working memory template
    wm_template:  list  = field(default_factory=list)
    # Conditions that trigger this task set
    trigger_conditions: dict = field(default_factory=dict)
    stability:    float = 0.0   # how stable this set currently is
    use_count:    int   = 0

@dataclass
class WorkingMemoryItem:
    """One item in working memory."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content:      str   = ""
    source:       str   = ""    # which module provided this
    priority:     float = 0.5
    decay_rate:   float = 0.05  # per cycle
    strength:     float = 1.0
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    task_relevant:bool  = True

@dataclass
class AttentionAllocation:
    """Current top-down attention allocation across modules."""
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    mode:         str   = AttentionMode.BROAD.value
    cycle:        int   = 0
    weights:      dict  = field(default_factory=dict)
    primary_target: str = ""
    secondary_targets: list = field(default_factory=list)
    attention_bandwidth: float = 1.0

@dataclass
class ReconfigurationEvent:
    """Records a pipeline reconfiguration."""
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    from_mode:    str   = ""
    to_mode:      str   = ""
    reason:       str   = ""
    trigger:      str   = ""
    switch_cost_ms: float = SWITCH_COST_MS
    mismatch_score: float = 0.0
    cycle:        int   = 0

@dataclass
class FPNOutput:
    """Complete FPN output for the cycle."""
    timestamp:    str   = field(default_factory=lambda: datetime.now().isoformat())
    cycle:        int   = 0
    task_mode:    str   = TaskMode.ROUTINE_MONITORING.value
    reconfigured: bool  = False
    reconfig_reason: str = ""
    attention:    AttentionAllocation = field(default_factory=AttentionAllocation)
    module_weights: dict = field(default_factory=dict)
    pipeline_order: list = field(default_factory=list)
    wm_items:     int   = 0
    wm_status:    str   = WorkingMemoryStatus.CLEAR.value
    flexibility_score: float = 0.0
    switch_cost_ms: float = 0.0

# ─── Database ─────────────────────────────────────────────────────────────────

class FPNDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS reconfig_events (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    from_mode TEXT, to_mode TEXT,
                    reason TEXT, trigger TEXT,
                    switch_cost_ms REAL, mismatch_score REAL, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS task_states (
                    cycle INTEGER PRIMARY KEY, timestamp TEXT,
                    task_mode TEXT, reconfigured INTEGER,
                    wm_items INTEGER, wm_status TEXT,
                    flexibility_score REAL, switch_cost_ms REAL
                );
                CREATE TABLE IF NOT EXISTS attention_log (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    mode TEXT, primary_target TEXT,
                    weights TEXT, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS working_memory_log (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    items TEXT, status TEXT, cycle INTEGER
                );
            """)
            self.conn.commit()

    def save_reconfig(self, r: ReconfigurationEvent):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO reconfig_events VALUES
                (?,?,?,?,?,?,?,?,?)
            """, (r.id, r.timestamp, r.from_mode, r.to_mode,
                  r.reason, r.trigger, r.switch_cost_ms,
                  r.mismatch_score, r.cycle))
            self.conn.commit()

    def save_task_state(self, cycle: int, mode: str, reconfig: bool,
                         wm_items: int, wm_status: str,
                         flex: float, cost: float):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO task_states VALUES
                (?,?,?,?,?,?,?,?)
            """, (cycle, datetime.now().isoformat(), mode,
                  int(reconfig), wm_items, wm_status, flex, cost))
            self.conn.commit()

    def save_attention(self, a: AttentionAllocation):
        with self.lock:
            self.conn.execute("""
                INSERT INTO attention_log VALUES (?,?,?,?,?,?)
            """, (str(uuid.uuid4())[:8], a.timestamp, a.mode,
                  a.primary_target, json.dumps(a.weights), a.cycle))
            self.conn.commit()

    def get_reconfig_history(self, limit=10):
        with self.lock:
            return self.conn.execute("""
                SELECT timestamp, from_mode, to_mode, reason, mismatch_score
                FROM reconfig_events ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()

    def get_task_states(self, limit=20):
        with self.lock:
            return self.conn.execute("""
                SELECT cycle, task_mode, reconfigured, wm_items,
                       wm_status, flexibility_score
                FROM task_states ORDER BY cycle DESC LIMIT ?
            """, (limit,)).fetchall()


# ─── Task Library ─────────────────────────────────────────────────────────────

class TaskLibrary:
    """
    Library of known task configurations.
    Each task set defines how the pipeline should be weighted
    for a specific cognitive demand.

    These are like the brain's "apps" — pre-configured
    processing modes that can be instantly loaded.
    """

    def build_library(self) -> dict[str, TaskSet]:
        library = {}

        # ── THREAT RESPONSE ───────────────────────────────────────────────
        library[TaskMode.THREAT_RESPONSE.value] = TaskSet(
            mode    = TaskMode.THREAT_RESPONSE.value,
            name    = "Threat Response Mode",
            description = "Crisis — survival channels maximum, reflection suppressed",
            module_weights = {
                "salience":       2.0,   # maximum threat detection
                "amygdala":       1.8,   # fear at full power
                "thalamus":       1.5,   # consciousness locked to crisis
                "temporal":       1.4,   # perception at full
                "sensorimotor":   1.8,   # reflex priority maximum
                "basal_ganglia":  1.3,   # habits over deliberation
                "prefrontal":     1.2,   # still needed for coordination
                "hippocampus":    1.4,   # encode crisis memories
                "limbic":         1.4,   # full emotional processing
                "neuromodulator": 1.3,   # cortisol/NE surge
                "swarm":          1.4,   # all agents active
                "bridge":         0.8,   # social less critical
                "dmn":            0.1,   # suppress reflection
                "visual":         1.5,   # maximum visual scanning
                "insula":         1.3,   # body state critical
                "anterior_cingulate": 1.4,
                "cerebellum":     1.2,
            },
            pipeline_order = ["salience","amygdala","thalamus","temporal",
                              "visual","sensorimotor","basal_ganglia",
                              "limbic","neuromodulator","prefrontal",
                              "hippocampus","swarm"],
            active_modules    = ["salience","amygdala","thalamus","temporal",
                                 "sensorimotor","basal_ganglia","visual"],
            suppressed_modules= ["dmn"],
            trigger_conditions= {"threat_min":3, "or_hijack":True},
            wm_template       = ["current_threat","entity_id","exit_routes",
                                 "available_actions","cortisol_level"]
        )

        # ── SOCIAL ENGAGEMENT ─────────────────────────────────────────────
        library[TaskMode.SOCIAL_ENGAGEMENT.value] = TaskSet(
            mode    = TaskMode.SOCIAL_ENGAGEMENT.value,
            name    = "Social Engagement Mode",
            description = "Cooperative interaction — bridge and trust systems amplified",
            module_weights = {
                "bridge":         1.8,   # social pipeline maximum
                "limbic":         1.5,   # emotional attunement
                "prefrontal":     1.4,   # social reasoning
                "hippocampus":    1.3,   # recall entity history
                "temporal":       1.2,   # attend to social signals
                "amygdala":       0.8,   # fear suppressed in safety
                "basal_ganglia":  1.2,   # cooperative habits
                "neuromodulator": 1.2,   # oxytocin/serotonin
                "dmn":            1.1,   # social simulation
                "salience":       1.0,
                "thalamus":       1.0,
                "sensorimotor":   0.9,
                "swarm":          1.3,   # collaborative agents
                "visual":         1.1,   # face/body reading
                "insula":         1.2,   # emotional resonance
                "anterior_cingulate": 1.2,
                "cerebellum":     0.9,
            },
            pipeline_order = ["temporal","bridge","limbic","prefrontal",
                              "hippocampus","swarm","dmn"],
            active_modules    = ["bridge","limbic","prefrontal","hippocampus"],
            suppressed_modules= [],
            trigger_conditions= {"intent":"COOPERATIVE_REQUEST","threat_max":1},
            wm_template       = ["entity_name","trust_score","shared_history",
                                 "current_goal","social_context"]
        )

        # ── LEARNING MODE ─────────────────────────────────────────────────
        library[TaskMode.LEARNING.value] = TaskSet(
            mode    = TaskMode.LEARNING.value,
            name    = "Learning Mode",
            description = "Novel information — hippocampus and temporal at full for encoding",
            module_weights = {
                "hippocampus":    1.9,   # memory encoding maximum
                "temporal":       1.6,   # deep perception
                "prefrontal":     1.4,   # analysis and categorization
                "dmn":            1.3,   # integration and insight
                "basal_ganglia":  1.3,   # pattern → habit formation
                "cerebellum":     1.2,   # timing and sequence
                "visual":         1.3,   # visual learning
                "limbic":         1.1,   # emotional tagging of memory
                "salience":       1.2,   # novelty detection
                "bridge":         1.0,
                "neuromodulator": 1.1,   # dopamine for learning
                "amygdala":       0.7,   # suppress fear during learning
                "sensorimotor":   0.8,
                "thalamus":       1.1,
                "swarm":          0.9,
                "insula":         1.0,
                "anterior_cingulate": 1.2,
            },
            pipeline_order = ["salience","temporal","visual","prefrontal",
                              "hippocampus","dmn","basal_ganglia"],
            active_modules    = ["hippocampus","temporal","prefrontal","dmn"],
            suppressed_modules= [],
            trigger_conditions= {"novelty_min":0.6,"threat_max":1},
            wm_template       = ["current_topic","key_concepts","open_questions",
                                 "related_memories","learning_goal"]
        )

        # ── MEMORY RETRIEVAL ──────────────────────────────────────────────
        library[TaskMode.MEMORY_RETRIEVAL.value] = TaskSet(
            mode    = TaskMode.MEMORY_RETRIEVAL.value,
            name    = "Memory Retrieval Mode",
            description = "Recalling past — hippocampus and DMN in replay mode",
            module_weights = {
                "hippocampus":    2.0,   # retrieval maximum
                "dmn":            1.7,   # memory replay network
                "prefrontal":     1.3,   # guided retrieval
                "temporal":       1.2,   # context reconstruction
                "amygdala":       1.2,   # emotional memory tagging
                "insula":         1.1,   # somatic marker retrieval
                "salience":       0.9,
                "bridge":         1.1,
                "limbic":         1.1,
                "basal_ganglia":  0.8,
                "sensorimotor":   0.6,
                "visual":         1.0,
                "neuromodulator": 0.9,
                "thalamus":       1.1,
                "swarm":          0.7,
                "anterior_cingulate": 1.0,
                "cerebellum":     0.8,
            },
            pipeline_order = ["prefrontal","hippocampus","dmn","temporal",
                              "amygdala","insula"],
            active_modules    = ["hippocampus","dmn","prefrontal"],
            suppressed_modules= ["sensorimotor"],
            trigger_conditions= {"memory_cue":True,"threat_max":1},
            wm_template       = ["retrieval_cue","target_episode","context_filter",
                                 "temporal_marker"]
        )

        # ── CREATIVE REFLECTION ───────────────────────────────────────────
        library[TaskMode.CREATIVE_REFLECTION.value] = TaskSet(
            mode    = TaskMode.CREATIVE_REFLECTION.value,
            name    = "Creative Reflection Mode",
            description = "DMN-dominant — insight generation, future simulation",
            module_weights = {
                "dmn":            2.0,   # maximum reflection
                "hippocampus":    1.5,   # memory recombination
                "prefrontal":     1.2,   # light guidance only
                "limbic":         1.3,   # emotional processing
                "insula":         1.2,   # interoceptive creativity
                "temporal":       0.8,
                "salience":       0.5,   # external world quieted
                "amygdala":       0.6,
                "sensorimotor":   0.3,   # body quiet
                "basal_ganglia":  0.7,
                "bridge":         0.8,
                "visual":         0.6,
                "neuromodulator": 1.1,   # dopamine for exploration
                "thalamus":       0.7,   # dreaming-like state
                "swarm":          0.8,
                "anterior_cingulate": 0.9,
                "cerebellum":     0.7,
            },
            pipeline_order = ["dmn","hippocampus","limbic","prefrontal"],
            active_modules    = ["dmn","hippocampus","limbic"],
            suppressed_modules= ["sensorimotor","salience"],
            trigger_conditions= {"threat_max":0,"dmn_active":True,"load_max":0.2},
            wm_template       = ["current_insight","open_problems",
                                 "recent_experiences","emotional_themes"]
        )

        # ── ROUTINE MONITORING ────────────────────────────────────────────
        library[TaskMode.ROUTINE_MONITORING.value] = TaskSet(
            mode    = TaskMode.ROUTINE_MONITORING.value,
            name    = "Routine Monitoring Mode",
            description = "Low-load baseline — balanced attention across all modules",
            module_weights = {mod: 1.0 for mod in [
                "salience","temporal","bridge","limbic","prefrontal",
                "hippocampus","swarm","dmn","visual","sensorimotor",
                "basal_ganglia","thalamus","amygdala","neuromodulator",
                "insula","anterior_cingulate","cerebellum"
            ]},
            pipeline_order = ["salience","temporal","bridge","limbic",
                              "prefrontal","hippocampus","swarm","dmn"],
            active_modules    = [],  # all active
            suppressed_modules= [],
            trigger_conditions= {"threat_max":1,"load_max":0.4},
            wm_template       = ["current_entity","last_event","baseline_state"]
        )

        # ── CONFLICT RESOLUTION ───────────────────────────────────────────
        library[TaskMode.CONFLICT_RESOLUTION.value] = TaskSet(
            mode    = TaskMode.CONFLICT_RESOLUTION.value,
            name    = "Conflict Resolution Mode",
            description = "Resolving contradictions — ACC and prefrontal amplified",
            module_weights = {
                "anterior_cingulate": 2.0,  # conflict monitoring maximum
                "prefrontal":     1.7,       # executive resolution
                "thalamus":       1.3,       # consciousness focused
                "hippocampus":    1.4,       # recall relevant precedents
                "bridge":         1.3,       # social context critical
                "limbic":         1.1,
                "temporal":       1.2,
                "amygdala":       1.0,
                "dmn":            0.9,
                "salience":       1.2,
                "basal_ganglia":  1.0,
                "sensorimotor":   0.8,
                "visual":         1.0,
                "neuromodulator": 1.0,
                "swarm":          1.2,
                "insula":         1.2,
                "cerebellum":     1.0,
            },
            pipeline_order = ["anterior_cingulate","prefrontal","hippocampus",
                              "bridge","thalamus","swarm"],
            active_modules    = ["anterior_cingulate","prefrontal","hippocampus"],
            suppressed_modules= [],
            trigger_conditions= {"conflict_detected":True},
            wm_template       = ["conflict_type","competing_signals",
                                 "resolution_options","precedents"]
        )

        # ── RECOVERY MODE ─────────────────────────────────────────────────
        library[TaskMode.RECOVERY.value] = TaskSet(
            mode    = TaskMode.RECOVERY.value,
            name    = "Recovery Mode",
            description = "Post-stress restoration — cortisol clearing, energy rebuilding",
            module_weights = {
                "neuromodulator": 1.6,   # chemical recovery
                "hippocampus":    1.4,   # consolidate crisis memories
                "dmn":            1.5,   # debrief and integrate
                "insula":         1.3,   # body restoration
                "limbic":         1.2,   # emotional processing
                "prefrontal":     0.9,   # reduced demand
                "salience":       0.8,   # lower threat sensitivity
                "amygdala":       0.9,   # extinction processing
                "sensorimotor":   0.7,
                "basal_ganglia":  1.0,
                "bridge":         1.1,
                "temporal":       0.9,
                "visual":         0.8,
                "thalamus":       1.1,
                "swarm":          0.8,
                "anterior_cingulate": 1.1,
                "cerebellum":     1.0,
            },
            pipeline_order = ["neuromodulator","hippocampus","dmn",
                              "insula","limbic","bridge"],
            active_modules    = ["neuromodulator","hippocampus","dmn","insula"],
            suppressed_modules= ["sensorimotor"],
            trigger_conditions= {"prev_threat_min":3,"threat_max":1},
            wm_template       = ["crisis_summary","lessons_learned",
                                 "recovery_progress","next_precautions"]
        )

        # ── DEEP FOCUS ────────────────────────────────────────────────────
        library[TaskMode.DEEP_FOCUS.value] = TaskSet(
            mode    = TaskMode.DEEP_FOCUS.value,
            name    = "Deep Focus Mode",
            description = "Sustained single-task — prefrontal at maximum, distractions suppressed",
            module_weights = {
                "prefrontal":     1.9,   # executive control maximum
                "thalamus":       1.5,   # focused consciousness
                "temporal":       1.4,   # deep perception
                "hippocampus":    1.3,   # working memory support
                "cerebellum":     1.3,   # timing precision
                "salience":       0.7,   # suppress distractors
                "amygdala":       0.6,
                "dmn":            0.2,   # suppress mind-wandering
                "bridge":         1.0,
                "limbic":         0.9,
                "basal_ganglia":  1.2,
                "sensorimotor":   1.0,
                "visual":         1.2,
                "neuromodulator": 1.0,
                "swarm":          0.8,
                "insula":         0.9,
                "anterior_cingulate": 1.3,
            },
            pipeline_order = ["prefrontal","thalamus","temporal","hippocampus",
                              "cerebellum","basal_ganglia"],
            active_modules    = ["prefrontal","thalamus","temporal"],
            suppressed_modules= ["dmn","salience"],
            trigger_conditions= {"sustained_task":True,"threat_max":1},
            wm_template       = ["task_goal","current_step","progress",
                                 "blockers","next_steps"]
        )

        return library


# ─── Task Classifier ─────────────────────────────────────────────────────────

class TaskClassifier:
    """
    Determines which task mode is most appropriate
    for the current signal and cognitive context.
    """

    def classify(self, signal: dict,
                 body_state: str = "COMFORTABLE",
                 consciousness: str = "AWAKE",
                 prev_threat: int = 0) -> tuple[TaskMode, float]:
        """
        Returns (task_mode, confidence).
        """
        threat   = signal.get("threat", 0)
        anomaly  = signal.get("anomaly", False)
        novelty  = signal.get("novelty", 0.5)
        social   = signal.get("social",{}) or {}
        intent   = social.get("inferred_intent","") if isinstance(social,dict) else ""
        load     = signal.get("load", signal.get("pipeline_load", 0.3))
        dmn_active= signal.get("dmn_active", False)
        conflict = signal.get("conflict_detected", False)
        mem_cue  = signal.get("memory_cue", False)

        # THREAT RESPONSE — highest priority
        if threat >= 3 or signal.get("hijack", False):
            return TaskMode.THREAT_RESPONSE, 0.95

        # RECOVERY — post-crisis
        if prev_threat >= 3 and threat <= 1:
            return TaskMode.RECOVERY, 0.85

        # CONFLICT RESOLUTION
        if conflict or signal.get("conflict_type","NONE") != "NONE":
            return TaskMode.CONFLICT_RESOLUTION, 0.80

        # MEMORY RETRIEVAL
        if mem_cue or (novelty < 0.2 and signal.get("memory_action") == ""):
            return TaskMode.MEMORY_RETRIEVAL, 0.70

        # LEARNING — novel + safe
        if novelty > 0.65 and threat == 0:
            return TaskMode.LEARNING, 0.75

        # CREATIVE REFLECTION — DMN active + calm
        if dmn_active and threat == 0 and load < 0.25:
            return TaskMode.CREATIVE_REFLECTION, 0.80

        # SOCIAL ENGAGEMENT
        if "COOPERATIVE" in intent and threat <= 1:
            return TaskMode.SOCIAL_ENGAGEMENT, 0.75

        # DEEP FOCUS — sustained, single task
        if load > 0.5 and threat <= 1 and not conflict:
            return TaskMode.DEEP_FOCUS, 0.65

        # ROUTINE MONITORING — default
        return TaskMode.ROUTINE_MONITORING, 0.60


# ─── Working Memory ───────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Holds the current task context in active memory.
    7 ± 2 items (Miller's Law).

    Items decay each cycle unless refreshed.
    High-priority items resist decay.
    When full — oldest/lowest priority items are displaced.
    """

    def __init__(self, db: FPNDB):
        self.db       = db
        self.items:   list[WorkingMemoryItem] = []
        self.capacity = WORKING_MEM_CAPACITY

    def update(self, signal: dict, task_set: TaskSet, cycle: int):
        """Update working memory based on current signal and task."""
        # Add new items from signal
        new_items = self._extract_items(signal, task_set)
        for item in new_items:
            self._insert(item)

        # Decay existing items
        self._decay()

        # Log
        self.db.conn.execute("""
            INSERT INTO working_memory_log VALUES (?,?,?,?,?)
        """, (str(uuid.uuid4())[:8], datetime.now().isoformat(),
              json.dumps([i.content for i in self.items[:5]]),
              self.status().value, cycle))
        self.db.conn.commit()

    def _extract_items(self, signal: dict,
                        task_set: TaskSet) -> list[WorkingMemoryItem]:
        """Extract relevant items from signal for this task."""
        items = []

        # Always include threat level
        threat = signal.get("threat", 0)
        if threat > 0:
            items.append(WorkingMemoryItem(
                content = f"threat_level:{threat}",
                source  = "salience",
                priority= threat / 4.0,
                strength= 1.0
            ))

        # Entity if present
        entity = signal.get("entity_name","")
        if entity and entity != "unknown":
            items.append(WorkingMemoryItem(
                content = f"entity:{entity}",
                source  = "bridge",
                priority= 0.6,
                strength= 0.9
            ))

        # Decision
        decision = str(signal.get("decision",""))
        if decision:
            items.append(WorkingMemoryItem(
                content = f"decision:{decision[:20]}",
                source  = "prefrontal",
                priority= 0.7,
                strength= 1.0,
                task_relevant=True
            ))

        # Task-specific items from template
        for template_key in task_set.wm_template[:3]:
            val = signal.get(template_key,"")
            if val:
                items.append(WorkingMemoryItem(
                    content = f"{template_key}:{str(val)[:20]}",
                    source  = "task_template",
                    priority= 0.5,
                    strength= 0.8
                ))

        return items[:4]  # max 4 new items per cycle

    def _insert(self, item: WorkingMemoryItem):
        """Insert item, displacing lowest priority if at capacity."""
        if len(self.items) >= self.capacity:
            # Remove lowest priority non-task-relevant item
            candidates = [i for i in self.items if not i.task_relevant]
            if candidates:
                lowest = min(candidates, key=lambda x: x.priority * x.strength)
                self.items.remove(lowest)
            else:
                # Remove lowest priority item
                self.items.sort(key=lambda x: x.priority * x.strength)
                self.items.pop(0)

        self.items.append(item)

    def _decay(self):
        """Decay all items each cycle. Remove dead items."""
        surviving = []
        for item in self.items:
            item.strength -= item.decay_rate
            if item.strength > 0.05:
                surviving.append(item)
        self.items = surviving

    def status(self) -> WorkingMemoryStatus:
        n = len(self.items)
        if n >= self.capacity:   return WorkingMemoryStatus.OVERLOADED
        if n >= self.capacity-2: return WorkingMemoryStatus.FULL
        if n >= self.capacity//2:return WorkingMemoryStatus.MODERATE
        return WorkingMemoryStatus.CLEAR

    def get_context(self) -> dict:
        return {
            "items":    [i.content for i in self.items],
            "count":    len(self.items),
            "capacity": self.capacity,
            "status":   self.status().value
        }


# ─── Flexibility Detector ─────────────────────────────────────────────────────

class FlexibilityDetector:
    """
    Detects when the current task set is no longer appropriate.
    Computes a MISMATCH SCORE between current task and current demands.
    When mismatch exceeds threshold, triggers reconfiguration.
    """

    def __init__(self):
        self.mismatch_history: deque = deque(maxlen=20)

    def compute_mismatch(self, current_mode: TaskMode,
                          signal: dict,
                          classifier: TaskClassifier,
                          prev_threat: int) -> tuple[float, TaskMode]:
        """
        Compute how mismatched the current task set is.
        Returns (mismatch_score, suggested_mode).
        """
        suggested, confidence = classifier.classify(
            signal,
            prev_threat=prev_threat
        )

        # If suggested mode matches current — low mismatch
        if suggested == current_mode:
            mismatch = 1.0 - confidence
        else:
            # Different modes — mismatch scales with confidence
            mismatch = confidence * 0.8 + 0.2

        self.mismatch_history.append(mismatch)
        return round(mismatch, 4), suggested

    def should_reconfig(self) -> bool:
        """Should we reconfigure based on recent mismatch history?"""
        if not self.mismatch_history: return False
        recent = list(self.mismatch_history)[-FLEXIBILITY_WINDOW:]
        avg_mismatch = sum(recent) / len(recent)
        return avg_mismatch > RECONFIG_THRESHOLD

    def flexibility_score(self) -> float:
        """How cognitively flexible has the system been?"""
        if not self.mismatch_history: return 0.5
        # Count mode switches as flexibility events
        recent = list(self.mismatch_history)[-10:]
        variance = sum((m-0.5)**2 for m in recent) / len(recent)
        return round(min(1.0, variance * 4), 3)


# ─── Attention Controller ─────────────────────────────────────────────────────

class AttentionController:
    """
    Generates top-down attention signals for all modules.
    This is GOAL-DIRECTED attention — different from salience's
    bottom-up "that was surprising."

    The FPN says: "Given our current task, pay MORE attention
    to these modules and LESS to those ones."
    """

    def allocate(self, task_set: TaskSet,
                 signal: dict, cycle: int) -> AttentionAllocation:
        """Generate attention allocation for current task."""
        weights = task_set.module_weights.copy()

        # Signal-specific adjustments
        threat = signal.get("threat", 0)
        if threat >= 2:
            # Boost survival modules regardless of task
            for mod in ["salience","amygdala","sensorimotor"]:
                weights[mod] = max(weights.get(mod,1.0), 1.3)

        # Determine attention mode
        active_count = len(task_set.active_modules)
        if active_count == 0:      attn_mode = AttentionMode.BROAD
        elif active_count == 1:    attn_mode = AttentionMode.FOCUSED
        elif active_count <= 3:    attn_mode = AttentionMode.DIVIDED
        else:                      attn_mode = AttentionMode.BROAD

        # Primary target
        primary = (task_set.active_modules[0]
                   if task_set.active_modules else "prefrontal")

        # Compute attention bandwidth
        max_weight = max(weights.values()) if weights else 1.0
        bandwidth  = round(sum(weights.values()) / len(weights), 3) if weights else 1.0

        alloc = AttentionAllocation(
            mode               = attn_mode.value,
            cycle              = cycle,
            weights            = {k: round(v,3) for k,v in weights.items()},
            primary_target     = primary,
            secondary_targets  = task_set.active_modules[1:4],
            attention_bandwidth= bandwidth
        )
        return alloc


# ─── Reconfiguration Engine ───────────────────────────────────────────────────

class ReconfigurationEngine:
    """
    Executes pipeline reconfiguration when triggered.
    Manages the transition between task sets.
    Applies switch costs — reconfiguration takes time.
    """

    def __init__(self, db: FPNDB):
        self.db     = db
        self.reconfig_count = 0
        self.total_switch_cost_ms = 0.0

    def reconfig(self, from_mode: str, to_mode: str,
                 to_set: TaskSet, mismatch: float,
                 reason: ReconfigReason, trigger: str,
                 cycle: int) -> ReconfigurationEvent:
        """Execute reconfiguration and return event record."""
        self.reconfig_count += 1

        # Switch cost increases with how different the modes are
        switch_cost = SWITCH_COST_MS
        if from_mode == TaskMode.CREATIVE_REFLECTION.value:
            switch_cost *= 1.5  # harder to exit reflection
        if to_mode == TaskMode.THREAT_RESPONSE.value:
            switch_cost *= 0.3  # fast entry to crisis mode

        switch_cost = round(switch_cost, 2)
        self.total_switch_cost_ms += switch_cost

        to_set.use_count += 1

        event = ReconfigurationEvent(
            from_mode      = from_mode,
            to_mode        = to_mode,
            reason         = reason.value,
            trigger        = trigger[:50],
            switch_cost_ms = switch_cost,
            mismatch_score = mismatch,
            cycle          = cycle
        )
        self.db.save_reconfig(event)
        return event


# ─── FORGE Frontoparietal Network ─────────────────────────────────────────────

class ForgeFrontoparietalNetwork:
    def __init__(self):
        self.db          = FPNDB()
        self.library     = TaskLibrary()
        self.task_sets   = self.library.build_library()
        self.classifier  = TaskClassifier()
        self.wm          = WorkingMemory(self.db)
        self.flex        = FlexibilityDetector()
        self.attn        = AttentionController()
        self.reconfig_eng= ReconfigurationEngine(self.db)
        self.cycle       = 0
        self.current_mode= TaskMode.ROUTINE_MONITORING
        self.current_set = self.task_sets[TaskMode.ROUTINE_MONITORING.value]
        self.prev_threat = 0
        self.reconfig_count   = 0
        self.total_cycles     = 0

    def process(self, signal: dict) -> dict:
        """
        Main FPN processing.
        Classify task, check for reconfig, allocate attention.
        """
        self.cycle      += 1
        self.total_cycles= self.cycle
        threat           = signal.get("threat", 0)

        # 1. Classify current task
        suggested_mode, confidence = self.classifier.classify(
            signal, prev_threat=self.prev_threat
        )

        # 2. Compute mismatch with current mode
        mismatch, _ = self.flex.compute_mismatch(
            self.current_mode, signal, self.classifier, self.prev_threat
        )

        # 3. Check if reconfiguration needed
        reconfigured   = False
        reconfig_event = None
        switch_cost_ms = 0.0

        if (suggested_mode != self.current_mode and
                mismatch > RECONFIG_THRESHOLD):
            # Execute reconfiguration
            from_mode    = self.current_mode.value
            self.current_mode = suggested_mode
            self.current_set  = self.task_sets.get(
                suggested_mode.value,
                self.task_sets[TaskMode.ROUTINE_MONITORING.value]
            )
            reconfig_event = self.reconfig_eng.reconfig(
                from_mode    = from_mode,
                to_mode      = suggested_mode.value,
                to_set       = self.current_set,
                mismatch     = mismatch,
                reason       = ReconfigReason.CONTEXT_SHIFT,
                trigger      = f"mismatch={mismatch:.2f},threat={threat}",
                cycle        = self.cycle
            )
            reconfigured   = True
            switch_cost_ms = reconfig_event.switch_cost_ms
            self.reconfig_count += 1

        # 4. Update working memory
        self.wm.update(signal, self.current_set, self.cycle)
        wm_ctx    = self.wm.get_context()

        # 5. Allocate attention
        attention = self.attn.allocate(self.current_set, signal, self.cycle)
        self.db.save_attention(attention)

        # 6. Compute flexibility score
        flex_score = self.flex.flexibility_score()

        # 7. Save task state
        self.db.save_task_state(
            self.cycle, self.current_mode.value, reconfigured,
            wm_ctx["count"], wm_ctx["status"], flex_score, switch_cost_ms
        )

        # Update prev threat
        self.prev_threat = threat

        return {
            "cycle":           self.cycle,
            "task_mode":       self.current_mode.value,
            "task_name":       self.current_set.name,
            "task_description":self.current_set.description,
            "reconfigured":    reconfigured,
            "mismatch_score":  mismatch,
            "switch_cost_ms":  switch_cost_ms,
            "reconfig_from":   reconfig_event.from_mode if reconfig_event else "",
            "attention": {
                "mode":         attention.mode,
                "primary":      attention.primary_target,
                "secondary":    attention.secondary_targets,
                "bandwidth":    attention.attention_bandwidth,
            },
            "module_weights":  {k: v for k,v in
                               sorted(attention.weights.items(),
                               key=lambda x: x[1], reverse=True)[:8]},
            "pipeline_order":  self.current_set.pipeline_order,
            "active_modules":  self.current_set.active_modules,
            "suppressed":      self.current_set.suppressed_modules,
            "working_memory":  wm_ctx,
            "flexibility_score": flex_score,
            "reconfig_count":  self.reconfig_count,
            "wm_template":     self.current_set.wm_template,
        }

    def get_status(self) -> dict:
        rows = self.db.get_reconfig_history(5)
        return {
            "version":        VERSION,
            "cycle":          self.cycle,
            "current_mode":   self.current_mode.value,
            "reconfig_count": self.reconfig_count,
            "total_switch_cost_ms": self.reconfig_eng.total_switch_cost_ms,
            "wm_items":       len(self.wm.items),
            "wm_status":      self.wm.status().value,
            "available_modes":[m.value for m in TaskMode],
            "reconfig_history":[{
                "from":r[1],"to":r[2],"reason":r[3],"mismatch":r[4]
            } for r in rows]
        }


# ─── Rich UI ──────────────────────────────────────────────────────────────────

MODE_COLORS = {
    "THREAT_RESPONSE":    "bright_red",
    "SOCIAL_ENGAGEMENT":  "magenta",
    "LEARNING":           "cyan",
    "MEMORY_RETRIEVAL":   "blue",
    "CREATIVE_REFLECTION":"dim",
    "ROUTINE_MONITORING": "green",
    "CONFLICT_RESOLUTION":"yellow",
    "RECOVERY":           "orange3",
    "DEEP_FOCUS":         "bright_cyan",
}

def render_fpn(result: dict, label: str, idx: int):
    if not HAS_RICH: return

    mode   = result["task_mode"]
    mc     = MODE_COLORS.get(mode, "white")
    recon  = result["reconfigured"]
    attn   = result["attention"]

    console.print(Rule(
        f"[bold cyan]⬡ FPN[/bold cyan]  [dim]#{idx}[/dim]  "
        f"[{mc}]{mode}[/{mc}]  "
        f"{'[bold yellow]⟳ RECONFIGURED[/bold yellow]  ' if recon else ''}"
        f"[dim]mismatch={result['mismatch_score']:.2f}[/dim]"
    ))

    if recon:
        console.print(Panel(
            f"[bold yellow]⟳ PIPELINE RECONFIGURED[/bold yellow]\n"
            f"[dim]{result['reconfig_from']}[/dim] → [{mc}]{mode}[/{mc}]\n"
            f"Switch cost: {result['switch_cost_ms']:.1f}ms\n"
            f"[dim]{result['task_description']}[/dim]",
            border_style="yellow"
        ))

    # Module weights
    weights = result["module_weights"]
    wt = Table(box=box.SIMPLE, show_header=False, expand=True)
    wt.add_column("module", style="dim", width=18)
    wt.add_column("weight")
    wt.add_column("bar")

    for mod, w in list(weights.items())[:8]:
        wc    = "bright_green" if w>1.5 else "green" if w>1.1 else \
                "dim" if w<0.8 else "yellow" if w<1.0 else "white"
        bar   = "█" * int(w * 5) + "░" * max(0, 10-int(w*5))
        active= "▶" if mod in result["active_modules"] else \
                "✗" if mod in result["suppressed"] else " "
        wt.add_row(
            f"{active} {mod}",
            f"[{wc}]×{w:.1f}[/{wc}]",
            f"[{wc}]{bar}[/{wc}]"
        )

    # Pipeline + WM
    pipeline_str = " → ".join(
        f"[{mc}]{m}[/{mc}]" if m in result["active_modules"] else
        f"[dim]{m}[/dim]"
        for m in result["pipeline_order"][:6]
    )

    wm      = result["working_memory"]
    wm_stat = wm["status"]
    wmc     = {"CLEAR":"green","MODERATE":"yellow","FULL":"orange3",
               "OVERLOADED":"bright_red"}.get(wm_stat,"white")

    right_lines = [
        f"[bold]Pipeline:[/bold]",
        f"  {pipeline_str}",
        f"",
        f"[bold]Attention:[/bold]  {attn['mode']}",
        f"[bold]Primary:[/bold]    [{mc}]{attn['primary']}[/{mc}]",
        f"[bold]Bandwidth:[/bold]  {attn['bandwidth']:.2f}",
        f"",
        f"[bold]WM Status:[/bold]  [{wmc}]{wm_stat}[/{wmc}] ({wm['count']}/{wm['capacity']})",
    ]
    if wm.get("items"):
        for item in wm["items"][:3]:
            right_lines.append(f"  [dim]• {item[:35]}[/dim]")

    right_lines += [
        f"",
        f"[bold]Flexibility:[/bold] {result['flexibility_score']:.3f}",
        f"[bold]Reconfigs:[/bold]  {result['reconfig_count']}",
    ]

    console.print(Columns([
        Panel(wt,                    title=f"[bold {mc}]Module Weights[/bold {mc}]", border_style=mc),
        Panel("\n".join(right_lines),title="[bold]Pipeline + WM[/bold]",            border_style="dim")
    ]))


def run_demo():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FORGE FRONTOPARIETAL NETWORK[/bold cyan]\n"
            "[dim]Dynamic Pipeline Reconfiguration · Working Memory · Task Sets[/dim]\n"
            f"[dim]Version {VERSION}  |  {len(TaskMode)} task modes[/dim]",
            border_style="cyan"
        ))

    fpn = ForgeFrontoparietalNetwork()

    scenarios = [
        # Routine start
        ({"threat":0,"anomaly":False,"entity_name":"alice_tech",
          "salience_score":0.2,"decision":"MONITOR","load":0.2,
          "social":{"inferred_intent":"COOPERATIVE_REQUEST"},
          "novelty":0.3,"dmn_active":False,"conflict_detected":False},
         "Routine monitoring — baseline"),

        # Novel discovery → LEARNING mode
        ({"threat":0,"anomaly":False,"entity_name":"system",
          "salience_score":0.45,"decision":"INVESTIGATE","load":0.3,
          "social":{"inferred_intent":"NEUTRAL_INTERACTION"},
          "novelty":0.82,"dmn_active":False,"conflict_detected":False},
         "Novel pattern — switch to LEARNING"),

        # Cooperative → SOCIAL ENGAGEMENT
        ({"threat":0,"anomaly":False,"entity_name":"alice_tech",
          "salience_score":0.35,"decision":"COLLABORATE","load":0.25,
          "social":{"inferred_intent":"COOPERATIVE_REQUEST"},
          "novelty":0.2,"dmn_active":False,"conflict_detected":False},
         "Cooperative interaction — SOCIAL ENGAGEMENT"),

        # Conflict detected → CONFLICT RESOLUTION
        ({"threat":2,"anomaly":False,"entity_name":"unknown_x",
          "salience_score":0.6,"decision":"ALERT","load":0.45,
          "social":{"inferred_intent":"COERCIVE_DEMAND"},
          "novelty":0.5,"conflict_detected":True,
          "conflict_type":"INFORMATION_CONFLICT"},
         "Conflict detected — CONFLICT RESOLUTION"),

        # CRISIS → THREAT RESPONSE
        ({"threat":4,"anomaly":True,"entity_name":"unknown_x",
          "salience_score":0.95,"decision":"EMERGENCY_BLOCK","load":0.8,
          "social":{"inferred_intent":"INTRUSION_ATTEMPT"},
          "novelty":0.7,"hijack":True,"conflict_detected":False},
         "CRITICAL — THREAT RESPONSE reconfiguration"),

        # Still high threat
        ({"threat":3,"anomaly":True,"entity_name":"unknown_x",
          "salience_score":0.85,"decision":"BLOCK","load":0.75,
          "social":{"inferred_intent":"INTRUSION_ATTEMPT"},
          "novelty":0.5,"conflict_detected":False},
         "Sustained threat — THREAT RESPONSE maintained"),

        # De-escalation → RECOVERY
        ({"threat":1,"anomaly":False,"entity_name":"security_team",
          "salience_score":0.3,"decision":"MONITOR","load":0.35,
          "social":{"inferred_intent":"COOPERATIVE_REQUEST"},
          "novelty":0.3,"conflict_detected":False},
         "De-escalation — switch to RECOVERY"),

        # DMN idle → CREATIVE REFLECTION
        ({"threat":0,"anomaly":False,"entity_name":"system",
          "salience_score":0.1,"decision":"STANDBY","load":0.12,
          "social":{"inferred_intent":"NEUTRAL_INTERACTION"},
          "novelty":0.2,"dmn_active":True,"conflict_detected":False},
         "Idle + DMN — CREATIVE REFLECTION"),
    ]

    for i, (sig, label) in enumerate(scenarios):
        if HAS_RICH:
            console.print(f"\n[bold dim]━━━ {i+1}: {label.upper()} ━━━[/bold dim]")
        result = fpn.process(sig)
        render_fpn(result, label, i+1)
        time.sleep(0.1)

    # Final
    if HAS_RICH:
        console.print(Rule("[bold cyan]⬡ FPN FINAL STATUS[/bold cyan]"))
        status = fpn.get_status()

        st = Table(box=box.DOUBLE_EDGE, border_style="cyan", title="FPN Status")
        st.add_column("Metric", style="cyan")
        st.add_column("Value",  style="white")
        mc2 = MODE_COLORS.get(status["current_mode"],"white")
        st.add_row("Cycles",          str(status["cycle"]))
        st.add_row("Current Mode",    f"[{mc2}]{status['current_mode']}[/{mc2}]")
        st.add_row("Reconfigurations",str(status["reconfig_count"]))
        st.add_row("Switch Cost Total",f"{status['total_switch_cost_ms']:.1f}ms")
        st.add_row("WM Items",        str(status["wm_items"]))
        st.add_row("WM Status",       status["wm_status"])
        console.print(st)

        if status["reconfig_history"]:
            rh = Table(box=box.SIMPLE, title="Reconfiguration History")
            rh.add_column("From",    width=20)
            rh.add_column("→ To",    width=22)
            rh.add_column("Mismatch",justify="right",width=10)
            for r in status["reconfig_history"]:
                fc = MODE_COLORS.get(r["from"],"white")
                tc = MODE_COLORS.get(r["to"],"white")
                rh.add_row(
                    f"[{fc}]{r['from'][:18]}[/{fc}]",
                    f"[{tc}]{r['to'][:20]}[/{tc}]",
                    f"{r['mismatch']:.3f}"
                )
            console.print(rh)


# ─── HTTP API ─────────────────────────────────────────────────────────────────

def run_api(fpn: ForgeFrontoparietalNetwork):
    if not HAS_FLASK: return
    app = Flask(__name__)

    @app.route("/process", methods=["POST"])
    def process():
        return jsonify(fpn.process(request.json or {}))

    @app.route("/mode", methods=["GET"])
    def mode():
        return jsonify({
            "mode":         fpn.current_mode.value,
            "task_name":    fpn.current_set.name,
            "pipeline":     fpn.current_set.pipeline_order,
            "active":       fpn.current_set.active_modules,
            "suppressed":   fpn.current_set.suppressed_modules,
        })

    @app.route("/weights", methods=["GET"])
    def weights():
        return jsonify(fpn.current_set.module_weights)

    @app.route("/working_memory", methods=["GET"])
    def working_memory():
        return jsonify(fpn.wm.get_context())

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(fpn.get_status())

    @app.route("/reconfig", methods=["POST"])
    def force_reconfig():
        data = request.json or {}
        mode_str = data.get("mode", TaskMode.ROUTINE_MONITORING.value)
        try:
            new_mode = TaskMode(mode_str)
            fpn.current_mode = new_mode
            fpn.current_set  = fpn.task_sets.get(
                mode_str,
                fpn.task_sets[TaskMode.ROUTINE_MONITORING.value]
            )
            return jsonify({"status":"reconfigured","mode":mode_str})
        except ValueError:
            return jsonify({"error":f"Unknown mode: {mode_str}"}), 400

    app.run(host="0.0.0.0", port=API_PORT, debug=False)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    fpn = ForgeFrontoparietalNetwork()
    if "--api" in sys.argv:
        t = threading.Thread(target=run_api, args=(fpn,), daemon=True)
        t.start()
    run_demo()
