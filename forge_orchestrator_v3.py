"""
FORGE Orchestrator v3 — forge_orchestrator_v3.py
=================================================
The complete brain. Every module. Every connection.

v3 vs v2:
  NEW MODULES INTEGRATED:
    - forge_amygdala.py       :7792  fast fear + hijack
    - forge_thalamus.py       :7790  consciousness gating
    - forge_basal_ganglia.py  :7791  habit + action selection
    - forge_sensorimotor.py   :7788  reflex/fast/deliberate
    - forge_visual.py         :7789  dual visual streams
    - forge_neuromodulator.py :7787  slow chemical layer
    - forge_cerebellum.py     :7793  timing + error correction
    - forge_anterior_cingulate.py :7794  conflict detection
    - forge_insula.py         :7795  interoception
    - forge_frontoparietal.py :7796  dynamic reconfiguration

  NEW CONNECTIONS WIRED:
    - amygdala → salience     (fear boosts salience threshold)
    - amygdala → thalamus     (hijack overrides consciousness)
    - amygdala → neuromodulator (fear spikes NE/cortisol)
    - neuromodulator → basal_ganglia (dopamine → learning rate)
    - neuromodulator → sensorimotor  (NE → reflex threshold)
    - thalamus → all modules  (consciousness gates everything)
    - basal_ganglia → prefrontal (habit bypasses deliberation)
    - cerebellum → all modules (timing corrections injected)
    - anterior_cingulate → prefrontal (conflict → extra control)
    - insula → prefrontal    (body state biases decisions)
    - frontoparietal → all   (dynamic weight reconfiguration)

  FULL PIPELINE v3:
    signal
      → [FRONTOPARIETAL]  determine task mode + reconfigure
      → [THALAMUS]        consciousness gate + 11 module gates
      → [SALIENCE]        score + interrupt/filter
        → AMYGDALA HIJACK? immediate survival response
      → [AMYGDALA]        fast fear tag (8ms subcortical)
      → [TEMPORAL]        bilateral perception
      → [VISUAL]          scene graph + threat geometry
      → [BRIDGE]          social enrichment
      → [LIMBIC]          feel — emotion + drives
      → [NEUROMODULATOR]  chemical update
      → [INSULA]          body state awareness
      → [ANTERIOR_CINGULATE] conflict detection
      → [CEREBELLUM]      timing prediction (before prefrontal)
      → [PREFRONTAL]      decide WITH:
                            - mood modifier (limbic)
                            - body state bias (insula)
                            - conflict signal (ACC)
                            - habit override (basal_ganglia)
                            - attention weights (frontoparietal)
      → [BASAL_GANGLIA]   habit selection + reward gating
      → [HIPPOCAMPUS]     remember WITH emotional tag
      → [SENSORIMOTOR]    reflex/fast/deliberate response
      → [SWARM]           collective action
      → [DMN]             brief for reflection
      → [CEREBELLUM]      timing feedback (after pipeline)
      → unified cognitive response
"""

import json
import time
import uuid
import sqlite3
import threading
from datetime import datetime
from collections import deque, defaultdict
from typing import Optional
from dataclasses import dataclass, field

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.align import Align
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

DB_PATH   = "forge_orchestrator_v3.db"
ORCH_PORT = 7797
VERSION   = "3.0.0"
TIMEOUT   = 5

console = Console() if HAS_RICH else None

# ─── Complete Module Registry ─────────────────────────────────────────────────

MODULES = {
    # Tier 1 — Perception & Emotion
    "frontoparietal":  {"port":7796,"emoji":"🔧","color":"bright_cyan",
                        "role":"Task Reconfiguration",
                        "endpoints":{"process":"/process","status":"/status"}},
    "thalamus":        {"port":7790,"emoji":"🔀","color":"bright_white",
                        "role":"Consciousness Gate",
                        "endpoints":{"process":"/process","status":"/status"}},
    "salience":        {"port":7784,"emoji":"🎯","color":"bright_red",
                        "role":"Priority Gate",
                        "endpoints":{"score":"/score","status":"/status"}},
    "amygdala":        {"port":7792,"emoji":"😨","color":"red",
                        "role":"Fear + Hijack",
                        "endpoints":{"process":"/process","status":"/status"}},
    "temporal":        {"port":7778,"emoji":"🧠","color":"cyan",
                        "role":"Perception",
                        "endpoints":{"perceive":"/perceive","status":"/status"}},
    "visual":          {"port":7789,"emoji":"👁","color":"blue",
                        "role":"Visual Streams",
                        "endpoints":{"perceive":"/perceive","status":"/status"}},
    "bridge":          {"port":7781,"emoji":"🔗","color":"blue",
                        "role":"Social Bridge",
                        "endpoints":{"sync":"/sync","status":"/stats"}},
    # Tier 2 — Chemical & Body
    "limbic":          {"port":7785,"emoji":"💗","color":"magenta",
                        "role":"Emotion",
                        "endpoints":{"feel":"/feel","status":"/status"}},
    "neuromodulator":  {"port":7787,"emoji":"🧪","color":"yellow",
                        "role":"Chemical Layer",
                        "endpoints":{"process":"/process","status":"/status"}},
    "insula":          {"port":7795,"emoji":"🫀","color":"orange3",
                        "role":"Interoception",
                        "endpoints":{"sense":"/sense","status":"/status"}},
    "anterior_cingulate":{"port":7794,"emoji":"⚠","color":"yellow",
                        "role":"Conflict Detection",
                        "endpoints":{"process":"/process","status":"/status"}},
    # Tier 3 — Decision & Action
    "cerebellum_pre":  {"port":7793,"emoji":"⏱","color":"dim",
                        "role":"Timing Pre",
                        "endpoints":{"before":"/before","status":"/status"}},
    "prefrontal":      {"port":7779,"emoji":"👔","color":"bright_yellow",
                        "role":"Executive Decision",
                        "endpoints":{"think":"/think","status":"/status"}},
    "basal_ganglia":   {"port":7791,"emoji":"🔄","color":"cyan",
                        "role":"Habit Selection",
                        "endpoints":{"select":"/select","status":"/status"}},
    "hippocampus":     {"port":7780,"emoji":"📚","color":"green",
                        "role":"Memory",
                        "endpoints":{"remember":"/remember","status":"/status"}},
    "sensorimotor":    {"port":7788,"emoji":"⚡","color":"orange3",
                        "role":"Reflex/Action",
                        "endpoints":{"process":"/process","status":"/status"}},
    # Tier 4 — Coordination
    "swarm":           {"port":7782,"emoji":"🐝","color":"yellow",
                        "role":"Collective Action",
                        "endpoints":{"signal":"/signal","status":"/status"}},
    "dmn":             {"port":7783,"emoji":"💭","color":"dim",
                        "role":"Reflection",
                        "endpoints":{"ingest":"/ingest","status":"/status"}},
    "cerebellum_post": {"port":7793,"emoji":"⏱","color":"dim",
                        "role":"Timing Post",
                        "endpoints":{"observe":"/observe","status":"/status"}},
}

# V3 Pipeline — all 19 stages
PIPELINE_V3 = [
    "frontoparietal",    # task mode + module weights
    "thalamus",          # consciousness + gate config
    "salience",          # interrupt/filter/amplify
    "amygdala",          # fast fear (8ms subcortical)
    "temporal",          # bilateral perception
    "visual",            # scene graph
    "bridge",            # social enrichment
    "limbic",            # emotion + drives
    "neuromodulator",    # chemical update
    "insula",            # body state
    "anterior_cingulate",# conflict detection
    "cerebellum_pre",    # timing prediction
    "prefrontal",        # decision (all inputs injected)
    "basal_ganglia",     # habit selection
    "hippocampus",       # memory with emotional tag
    "sensorimotor",      # reflex/fast/deliberate
    "swarm",             # collective action
    "dmn",               # reflection brief
    "cerebellum_post",   # timing feedback
]

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class V3CognitiveState:
    timestamp:       str   = field(default_factory=lambda: datetime.now().isoformat())
    signal_id:       str   = ""
    # Perception
    threat:          int   = 0
    salience_score:  float = 0.0
    salience_class:  str   = "MEDIUM"
    interrupted:     bool  = False
    filtered:        bool  = False
    hijacked:        bool  = False
    # Task
    task_mode:       str   = "ROUTINE_MONITORING"
    consciousness:   str   = "AWAKE"
    # Emotion + Body
    emotion:         str   = "neutral"
    mood:            str   = "NEUTRAL"
    mood_valence:    float = 0.0
    top_drive:       str   = "none"
    fear_score:      float = 0.0
    body_state:      str   = "COMFORTABLE"
    felt_sense:      str   = "EASE"
    energy_level:    float = 0.7
    neuro_state:     str   = "BASELINE"
    # Decision
    decision:        str   = "STANDBY"
    habit_used:      bool  = False
    response_tier:   str   = "DELIBERATE"
    # Memory
    memory_action:   str   = ""
    novelty:         float = 1.0
    # Conflict
    conflict_type:   str   = "NONE"
    conflict_strength:float= 0.0
    # Timing
    pipeline_ms:     float = 0.0
    smoothness:      float = 1.0
    # Collective
    swarm_phase:     str   = "CALM"
    # Meta
    conclusion:      str   = ""
    modules_live:    list  = field(default_factory=list)
    modules_offline: list  = field(default_factory=list)
    cycle:           int   = 0

@dataclass
class PipelineStep:
    module:      str   = ""
    success:     bool  = False
    latency_ms:  float = 0.0
    response:    dict  = field(default_factory=dict)
    error:       str   = ""
    injected:    dict  = field(default_factory=dict)

@dataclass
class ModuleHealth:
    module_id:   str   = ""
    alive:       bool  = False
    latency_ms:  float = 0.0
    last_ping:   str   = ""
    fail_count:  int   = 0
    ping_count:  int   = 0

    @property
    def uptime_pct(self):
        if self.ping_count == 0: return 0.0
        return round((self.ping_count-self.fail_count)/self.ping_count*100, 1)

# ─── Database ─────────────────────────────────────────────────────────────────

class OrchestratorV3DB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS cognitive_states (
                    timestamp TEXT PRIMARY KEY, signal_id TEXT,
                    threat INTEGER, salience_score REAL, salience_class TEXT,
                    interrupted INTEGER, filtered INTEGER, hijacked INTEGER,
                    task_mode TEXT, consciousness TEXT,
                    emotion TEXT, mood TEXT, mood_valence REAL,
                    fear_score REAL, body_state TEXT, felt_sense TEXT,
                    energy_level REAL, neuro_state TEXT,
                    decision TEXT, habit_used INTEGER, response_tier TEXT,
                    memory_action TEXT, novelty REAL,
                    conflict_type TEXT, conflict_strength REAL,
                    pipeline_ms REAL, smoothness REAL,
                    swarm_phase TEXT, conclusion TEXT,
                    modules_live TEXT, cycle INTEGER
                );
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id TEXT PRIMARY KEY, timestamp TEXT,
                    signal TEXT, steps TEXT, state TEXT,
                    total_ms REAL, success INTEGER
                );
                CREATE TABLE IF NOT EXISTS module_health (
                    module_id TEXT PRIMARY KEY, alive INTEGER,
                    latency_ms REAL, last_ping TEXT,
                    fail_count INTEGER, ping_count INTEGER
                );
            """)
            self.conn.commit()

    def save_state(self, s: V3CognitiveState):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO cognitive_states VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (s.timestamp, s.signal_id, s.threat, s.salience_score,
                  s.salience_class, int(s.interrupted), int(s.filtered),
                  int(s.hijacked), s.task_mode, s.consciousness,
                  s.emotion, s.mood, s.mood_valence, s.fear_score,
                  s.body_state, s.felt_sense, s.energy_level, s.neuro_state,
                  s.decision, int(s.habit_used), s.response_tier,
                  s.memory_action, s.novelty, s.conflict_type,
                  s.conflict_strength, s.pipeline_ms, s.smoothness,
                  s.swarm_phase, s.conclusion, json.dumps(s.modules_live),
                  s.cycle))
            self.conn.commit()

    def save_run(self, run_id: str, signal: dict, steps: list,
                 state: V3CognitiveState, ms: float, success: bool):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO pipeline_runs VALUES (?,?,?,?,?,?,?)
            """, (run_id, datetime.now().isoformat(), json.dumps(signal),
                  json.dumps([s.__dict__ for s in steps]),
                  json.dumps(state.__dict__), ms, int(success)))
            self.conn.commit()

    def save_health(self, h: ModuleHealth):
        with self.lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO module_health VALUES (?,?,?,?,?,?)
            """, (h.module_id, int(h.alive), h.latency_ms,
                  h.last_ping, h.fail_count, h.ping_count))
            self.conn.commit()

    def get_recent_states(self, limit=10):
        with self.lock:
            return self.conn.execute("""
                SELECT timestamp, threat, task_mode, emotion, mood,
                       decision, pipeline_ms, body_state, consciousness,
                       conflict_type, cycle
                FROM cognitive_states ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()

# ─── HTTP Client ──────────────────────────────────────────────────────────────

class ModuleClient:
    def __init__(self):
        self.session = requests.Session() if HAS_REQUESTS else None

    def post(self, module_id: str, endpoint_key: str,
             payload: dict) -> tuple[bool, dict, float]:
        if not self.session:
            return False, {"error":"requests unavailable"}, 0.0
        mod  = MODULES.get(module_id, {})
        path = mod.get("endpoints",{}).get(endpoint_key, "/status")
        port = mod.get("port", 7778)
        url  = f"http://127.0.0.1:{port}{path}"
        t0   = time.time()
        try:
            r  = self.session.post(url, json=payload, timeout=TIMEOUT)
            ms = (time.time()-t0)*1000
            return r.status_code==200, (r.json() if r.status_code==200
                   else {"error":f"HTTP {r.status_code}"}), round(ms,1)
        except Exception as e:
            return False, {"error":str(e)[:60]}, round((time.time()-t0)*1000,1)

    def get(self, module_id: str, endpoint: str = "/status") -> tuple[bool, dict, float]:
        if not self.session: return False, {}, 0.0
        mod  = MODULES.get(module_id, {})
        port = mod.get("port", 7778)
        url  = f"http://127.0.0.1:{port}{endpoint}"
        t0   = time.time()
        try:
            r  = self.session.get(url, timeout=2)
            ms = (time.time()-t0)*1000
            return r.status_code==200, (r.json() if r.status_code==200 else {}), round(ms,1)
        except Exception:
            return False, {}, round((time.time()-t0)*1000,1)

    def ping(self, module_id: str) -> tuple[bool, float]:
        ok, _, ms = self.get(module_id)
        return ok, ms

# ─── Health Monitor ───────────────────────────────────────────────────────────

class HealthMonitor:
    def __init__(self, client: ModuleClient, db: OrchestratorV3DB):
        self.client   = client
        self.db       = db
        self.health   = {mid: ModuleHealth(module_id=mid) for mid in MODULES}
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self): self._running = False

    def _loop(self):
        while self._running:
            for mid in MODULES:
                self._ping(mid)
            time.sleep(10)

    def _ping(self, mid: str):
        h          = self.health[mid]
        ok, ms     = self.client.ping(mid)
        h.ping_count += 1
        h.last_ping   = datetime.now().isoformat()
        h.latency_ms  = ms
        h.alive       = ok
        if not ok: h.fail_count += 1
        self.db.save_health(h)

    def initial_sweep(self):
        for mid in MODULES: self._ping(mid)

    def alive(self, mid: str) -> bool:
        return self.health[mid].alive

    def alive_count(self) -> int:
        return sum(1 for h in self.health.values() if h.alive)

    def summary(self) -> dict:
        return {
            "total": len(MODULES),
            "alive": self.alive_count(),
            "modules": {
                mid: {"alive":h.alive,"latency_ms":h.latency_ms,
                      "uptime":h.uptime_pct}
                for mid, h in self.health.items()
            }
        }

# ─── Cognitive State Tracker ──────────────────────────────────────────────────

class CognitiveStateTracker:
    """Maintains FORGE's persistent cognitive state across pipeline runs."""

    def __init__(self):
        # Persistent states
        self.mood          = "NEUTRAL"
        self.mood_val      = 0.0
        self.emotion       = "neutral"
        self.top_drive     = "SAFETY"
        self.arousal       = 0.3
        self.fear_score    = 0.0
        self.body_state    = "COMFORTABLE"
        self.felt_sense    = "EASE"
        self.energy_level  = 0.7
        self.neuro_state   = "BASELINE"
        self.dopamine_level= 0.45
        self.ne_level      = 0.20
        self.consciousness = "AWAKE"
        self.task_mode     = "ROUTINE_MONITORING"
        self.module_weights= {}
        self.pipeline_order= []
        self.conflict_type = "NONE"
        self.conflict_str  = 0.0
        self.smoothness    = 1.0
        self.history: deque = deque(maxlen=200)
        self.cycle         = 0

    def update_limbic(self, r: dict):
        self.mood      = r.get("mood",{}).get("tone", self.mood) if isinstance(r.get("mood"),dict) else self.mood
        self.mood_val  = r.get("mood",{}).get("valence", self.mood_val) if isinstance(r.get("mood"),dict) else self.mood_val
        self.emotion   = r.get("emotion",{}).get("primary", self.emotion) if isinstance(r.get("emotion"),dict) else self.emotion
        self.arousal   = r.get("emotion",{}).get("arousal", self.arousal) if isinstance(r.get("emotion"),dict) else self.arousal
        self.top_drive = r.get("most_urgent_drive", self.top_drive)

    def update_neuro(self, r: dict):
        self.neuro_state   = r.get("state", self.neuro_state)
        self.dopamine_level= r.get("profile",{}).get("dopamine", self.dopamine_level)
        self.ne_level      = r.get("profile",{}).get("norepinephrine", self.ne_level)

    def update_amygdala(self, r: dict):
        self.fear_score = r.get("fear_score", self.fear_score)

    def update_insula(self, r: dict):
        self.body_state   = r.get("body_state", self.body_state)
        self.felt_sense   = r.get("felt_sense", self.felt_sense)
        self.energy_level = r.get("body_budget",{}).get("energy", self.energy_level)

    def update_acc(self, r: dict):
        conflict = r.get("conflict",{})
        self.conflict_type = conflict.get("type","NONE")
        self.conflict_str  = conflict.get("strength", 0.0)

    def update_thalamus(self, r: dict):
        self.consciousness = r.get("consciousness", self.consciousness)

    def update_fpn(self, r: dict):
        self.task_mode      = r.get("task_mode", self.task_mode)
        self.module_weights = r.get("module_weights", self.module_weights)
        self.pipeline_order = r.get("pipeline_order", self.pipeline_order)

    def update_cerebellum(self, r: dict):
        self.smoothness = r.get("smoothness", self.smoothness)

    def limbic_injection(self) -> dict:
        return {"mood_state": {
            "tone":    self.mood, "valence": self.mood_val,
            "arousal": self.arousal, "top_drive": self.top_drive,
            "decision_modifier": {"confidence": self.mood_val*0.1,
                                  "risk_tolerance": self.mood_val*0.05}
        }}

    def body_injection(self) -> dict:
        return {"body_state": {
            "state":      self.body_state,
            "felt_sense": self.felt_sense,
            "energy":     self.energy_level,
            "bias":       (self.energy_level - 0.5) * 0.15
        }}

    def conflict_injection(self) -> dict:
        return {"conflict_signal": {
            "type":     self.conflict_type,
            "strength": self.conflict_str,
            "control_boost": min(0.3, self.conflict_str * 0.4)
        }}

    def record(self, state: V3CognitiveState):
        self.cycle += 1
        state.cycle = self.cycle
        self.history.append({
            "ts": state.timestamp, "threat": state.threat,
            "task": state.task_mode, "emotion": state.emotion,
            "decision": state.decision, "energy": state.energy_level
        })

    def trend(self) -> str:
        if len(self.history) < 3: return "STABLE"
        threats = [h["threat"] for h in list(self.history)[-4:]]
        if threats[-1] > threats[0]: return "ESCALATING ↑"
        if threats[-1] < threats[0]: return "DE-ESCALATING ↓"
        return "STABLE →"

# ─── V3 Pipeline Router ───────────────────────────────────────────────────────

class PipelineRouterV3:
    """
    The complete pipeline. All 19 stages.
    Every connection wired.
    """

    def __init__(self, client: ModuleClient, health: HealthMonitor,
                 tracker: CognitiveStateTracker, db: OrchestratorV3DB):
        self.client  = client
        self.health  = health
        self.tracker = tracker
        self.db      = db

    def route(self, raw_signal: dict) -> tuple[V3CognitiveState, list]:
        run_id = str(uuid.uuid4())[:8]
        t0     = time.time()
        signal = dict(raw_signal)
        steps  = []
        state  = V3CognitiveState(signal_id=signal.get("id", run_id))

        def call(mid: str, ep: str, payload: dict) -> PipelineStep:
            if not self.health.alive(mid):
                return PipelineStep(module=mid, error="OFFLINE")
            ok, resp, ms = self.client.post(mid, ep, payload)
            return PipelineStep(module=mid, success=ok, latency_ms=ms,
                               response=resp if ok else {},
                               error="" if ok else resp.get("error","ERR"))

        # ── STAGE 0: FRONTOPARIETAL — task mode + weights ─────────────────
        fp_step = call("frontoparietal","process", signal)
        steps.append(fp_step)
        if fp_step.success:
            self.tracker.update_fpn(fp_step.response)
            state.task_mode = self.tracker.task_mode
            signal["task_mode"]      = self.tracker.task_mode
            signal["module_weights"] = self.tracker.module_weights
            signal["fpn_pipeline"]   = self.tracker.pipeline_order

        # ── STAGE 1: THALAMUS — consciousness + gate config ───────────────
        th_step = call("thalamus","process",
                       {"signal":signal, "dmn_active":signal.get("dmn_active",False)})
        steps.append(th_step)
        if th_step.success:
            r = th_step.response
            self.tracker.update_thalamus(r)
            state.consciousness = self.tracker.consciousness
            signal["thalamic_routing"] = r.get("routing",{})
            signal["consciousness"]    = r.get("consciousness", "AWAKE")
            signal["thalamic_load"]    = r.get("load", 0.0)
            # Filtered by thalamus?
            if not r.get("passed", True):
                state.filtered = True
                state.conclusion = "FILTERED by thalamus"
                state.pipeline_ms = round((time.time()-t0)*1000,1)
                self.tracker.record(state)
                return state, steps

        # ── STAGE 2: SALIENCE — interrupt/filter/amplify ──────────────────
        sal_step = call("salience","score", signal)
        steps.append(sal_step)
        if sal_step.success:
            sal = sal_step.response
            state.salience_score = sal.get("score", 0.5)
            state.salience_class = sal.get("class","MEDIUM")
            signal.update({"salience_score": state.salience_score,
                           "salience_class": state.salience_class})

            routing = sal.get("routing",{})
            if isinstance(routing, dict) and not routing.get("pass", True):
                state.filtered   = True
                state.conclusion = "FILTERED — below salience threshold"
                state.pipeline_ms= round((time.time()-t0)*1000,1)
                self.tracker.record(state)
                return state, steps

        # ── STAGE 3: AMYGDALA — fast fear (8ms subcortical) ──────────────
        amy_step = call("amygdala","process", signal)
        steps.append(amy_step)
        if amy_step.success:
            r = amy_step.response
            self.tracker.update_amygdala(r)
            state.fear_score = self.tracker.fear_score
            signal["fear_score"]    = self.tracker.fear_score
            signal["amygdala_hijack"]=r.get("hijack", False)
            signal["ne_boost"]      = r.get("output",{}).get("ne_output",0.0)
            signal["salience_score"]= min(1.0,
                state.salience_score + r.get("output",{}).get("salience_boost",0.0))

            # AMYGDALA HIJACK — survival mode
            if r.get("hijack", False):
                state.hijacked  = True
                state.decision  = r.get("output",{}).get("emotional_tone","SURVIVAL_MODE")
                state.conclusion= f"⚡ AMYGDALA HIJACK — {state.decision}"
                state.emotion   = "terror"
                state.fear_score= 1.0
                state.pipeline_ms= round((time.time()-t0)*1000,1)
                self.tracker.record(state)
                return state, steps

        # ── STAGE 4: TEMPORAL — bilateral perception ──────────────────────
        t_step = call("temporal","perceive", {
            "text":           signal.get("text", signal.get("conclusion","")),
            "visual_input":   signal.get("visual_input",""),
            "auditory_input": signal.get("auditory_input",""),
            "entity_name":    signal.get("entity_name","unknown"),
        })
        steps.append(t_step)
        if t_step.success:
            r = t_step.response
            signal.update({
                "threat":     r.get("threat", signal.get("threat",0)),
                "anomaly":    r.get("anomaly", False),
                "conclusion": r.get("conclusion", signal.get("conclusion","")),
                "emotional":  r.get("emotional",{}),
                "social":     r.get("social",{}),
                "semantic":   r.get("semantic",{}),
                "temporal_id":r.get("id",""),
            })

        # ── STAGE 5: VISUAL — scene graph + threat geometry ───────────────
        vis_step = call("visual","perceive", {
            "description": signal.get("visual_input",""),
            "entity_name": signal.get("entity_name","unknown"),
        })
        steps.append(vis_step)
        if vis_step.success:
            r = vis_step.response
            signal["visual_scene"]    = r.get("scene_type","")
            signal["visual_threat"]   = r.get("visual_threat_score",0.0)
            signal["affordances"]     = r.get("affordances",[])
            signal["threat_geometry"] = r.get("threat_geometry",{})
            signal["visual_objects"]  = r.get("objects",[])
            # Boost threat from visual
            vis_threat = r.get("threat_objects",0)
            if vis_threat > 0:
                signal["threat"] = max(signal.get("threat",0), vis_threat+1)

        # ── STAGE 6: BRIDGE — social enrichment ───────────────────────────
        b_step = call("bridge","sync", {"perception": signal})
        steps.append(b_step)
        if b_step.success:
            r = b_step.response
            signal["threat"]         = r.get("enriched_threat", signal.get("threat",0))
            signal["conclusion"]     = r.get("conclusion", signal.get("conclusion",""))
            signal["social_context"] = r.get("social_context",{})

        # ── STAGE 7: LIMBIC — emotion + drives ────────────────────────────
        lim_step = call("limbic","feel",
                        {"signal":signal,"episode_id":signal.get("temporal_id","")})
        steps.append(lim_step)
        if lim_step.success:
            self.tracker.update_limbic(lim_step.response)
            state.emotion    = self.tracker.emotion
            state.mood       = self.tracker.mood
            state.mood_valence= self.tracker.mood_val
            state.top_drive  = self.tracker.top_drive
            signal["limbic_state"] = self.tracker.limbic_injection()

        # ── STAGE 8: NEUROMODULATOR — chemical update ─────────────────────
        neu_step = call("neuromodulator","process", signal)
        steps.append(neu_step)
        if neu_step.success:
            self.tracker.update_neuro(neu_step.response)
            state.neuro_state = self.tracker.neuro_state
            signal["neuro_state"]   = neu_step.response
            signal["dopamine_level"]= self.tracker.dopamine_level
            signal["ne_level"]      = self.tracker.ne_level

        # ── STAGE 9: INSULA — body state awareness ────────────────────────
        ins_step = call("insula","sense", {
            "signal": signal,
            "had_conflict": self.tracker.conflict_str > 0.3,
            "success": signal.get("threat",0) < 3,
        })
        steps.append(ins_step)
        if ins_step.success:
            self.tracker.update_insula(ins_step.response)
            state.body_state   = self.tracker.body_state
            state.felt_sense   = self.tracker.felt_sense
            state.energy_level = self.tracker.energy_level
            signal["body_state_injection"] = self.tracker.body_injection()

        # ── STAGE 10: ANTERIOR CINGULATE — conflict detection ─────────────
        acc_step = call("anterior_cingulate","process", {
            "signal":            signal,
            "expected_decision": signal.get("expected_decision",""),
            "success":           signal.get("threat",0) < 3,
        })
        steps.append(acc_step)
        if acc_step.success:
            self.tracker.update_acc(acc_step.response)
            state.conflict_type   = self.tracker.conflict_type
            state.conflict_strength= self.tracker.conflict_str
            signal["conflict_injection"]  = self.tracker.conflict_injection()
            signal["acc_control_level"]   = acc_step.response.get("control",{}).get("level",0.3)
            signal["acc_recommendation"]  = acc_step.response.get("recommendation","")

        # ── STAGE 11: CEREBELLUM PRE — timing prediction ──────────────────
        cb_pre_step = call("cerebellum_pre","before", {
            "module":  "prefrontal",
            "action":  "think",
            "context": {"threat":signal.get("threat",0)},
        })
        steps.append(cb_pre_step)

        # ── STAGE 12: PREFRONTAL — the decision (all inputs injected) ─────
        pf_payload = {"perception": signal}
        pf_payload["mood_modifier"]      = self.tracker.limbic_injection().get("mood_state",{})
        pf_payload["body_bias"]          = self.tracker.body_injection().get("body_state",{})
        pf_payload["conflict_signal"]    = self.tracker.conflict_injection().get("conflict_signal",{})
        pf_payload["acc_control"]        = signal.get("acc_control_level", 0.3)
        pf_payload["ne_level"]           = self.tracker.ne_level
        pf_payload["energy_level"]       = self.tracker.energy_level
        pf_payload["task_mode"]          = self.tracker.task_mode

        pf_step = call("prefrontal","think", pf_payload)
        steps.append(pf_step)
        if pf_step.success:
            r = pf_step.response
            chosen   = r.get("chosen",{})
            decision = (chosen.get("action","STANDBY")
                       if isinstance(chosen,dict) else str(chosen))
            signal["decision"]        = chosen
            signal["decision_action"] = decision
            signal["prefrontal_plan"] = r.get("plan",[])
            state.decision = decision

        # ── STAGE 13: BASAL GANGLIA — habit selection ─────────────────────
        # Dopamine level from neuromodulator feeds into BG learning rate
        bg_step = call("basal_ganglia","select", {
            "candidates":  ["BLOCK","MONITOR","COLLABORATE","ALERT",
                            "ESCALATE","STANDBY","INVESTIGATE",
                            "EMERGENCY_BLOCK","LEARN"],
            "signal":      signal,
            "dopamine":    self.tracker.dopamine_level,
        })
        steps.append(bg_step)
        if bg_step.success:
            r = bg_step.response
            bg_action = r.get("selected_action","")
            if r.get("was_habit") and bg_action:
                # Habit overrides deliberate decision
                signal["decision_action"] = bg_action
                state.decision    = bg_action
                state.habit_used  = True
            signal["habit_stage"]  = r.get("habit_stage","NOVEL")
            signal["bg_outcome"]   = r.get("outcome","SELECTED")

        # ── STAGE 14: HIPPOCAMPUS — memory with emotional tag ─────────────
        hc_step = call("hippocampus","remember", {
            "perception":    signal,
            "decision":      signal.get("decision",{}),
            "emotional_tag": {
                "emotion":   self.tracker.emotion,
                "valence":   self.tracker.mood_val,
                "arousal":   self.tracker.arousal,
                "fear":      self.tracker.fear_score,
            }
        })
        steps.append(hc_step)
        if hc_step.success:
            r = hc_step.response
            signal["memory_action"] = r.get("action","")
            signal["episode_id"]    = r.get("episode_id","")
            signal["novelty"]       = r.get("novelty",1.0)
            state.memory_action     = r.get("action","")
            state.novelty           = r.get("novelty",1.0)

        # ── STAGE 15: SENSORIMOTOR — reflex/fast/deliberate ───────────────
        # NE level from neuromodulator modulates reflex threshold
        sm_step = call("sensorimotor","process", {
            "signal":            signal,
            "pipeline_decision": signal.get("decision_action","STANDBY"),
            "neuro_state":       {"profile":{"norepinephrine":self.tracker.ne_level,
                                             "serotonin":0.65}},
        })
        steps.append(sm_step)
        if sm_step.success:
            r = sm_step.response
            state.response_tier = r.get("tier","DELIBERATE")
            signal["response_tier"]  = r.get("tier","DELIBERATE")
            signal["reflex_fired"]   = r.get("reflex",{}).get("fired",False)

        # ── STAGE 16: SWARM — collective action with full context ─────────
        sw_step = call("swarm","signal", {
            **signal,
            "emotional_context": {
                "mood":      self.tracker.mood,
                "emotion":   self.tracker.emotion,
                "arousal":   self.tracker.arousal,
                "top_drive": self.tracker.top_drive,
                "fear":      self.tracker.fear_score,
            },
            "task_mode":   self.tracker.task_mode,
            "body_state":  self.tracker.body_state,
        })
        steps.append(sw_step)
        if sw_step.success:
            r = sw_step.response
            state.swarm_phase = r.get("phase_after","CALM")
            signal["swarm_phase"] = state.swarm_phase

        # ── STAGE 17: DMN — brief for reflection ──────────────────────────
        dmn_step = call("dmn","ingest", signal)
        steps.append(dmn_step)

        # ── STAGE 18: CEREBELLUM POST — timing feedback ───────────────────
        module_timings = {
            s.module: s.latency_ms
            for s in steps if s.success and s.latency_ms > 0
        }
        cb_post_step = call("cerebellum_post","observe", {
            "timings": module_timings,
            "success": state.decision not in ["STANDBY"],
        })
        steps.append(cb_post_step)
        if cb_post_step.success:
            self.tracker.update_cerebellum(cb_post_step.response)
            state.smoothness = self.tracker.smoothness

        # ── FINALIZE ──────────────────────────────────────────────────────
        state.threat       = signal.get("threat", raw_signal.get("threat",0))
        state.conclusion   = signal.get("conclusion", raw_signal.get("conclusion",""))
        state.pipeline_ms  = round((time.time()-t0)*1000,1)
        state.modules_live = [s.module for s in steps if s.success]
        state.modules_offline=[s.module for s in steps if s.error=="OFFLINE"]

        self.tracker.record(state)
        self.db.save_state(state)
        self.db.save_run(run_id, raw_signal, steps, state, state.pipeline_ms, True)
        return state, steps

# ─── FORGE Orchestrator v3 ────────────────────────────────────────────────────

class ForgeOrchestratorV3:
    def __init__(self):
        self.db      = OrchestratorV3DB()
        self.client  = ModuleClient()
        self.health  = HealthMonitor(self.client, self.db)
        self.tracker = CognitiveStateTracker()
        self.router  = PipelineRouterV3(self.client, self.health, self.tracker, self.db)
        self.results: list = []
        self.cycle   = 0
        self.health.start()
        self.health.initial_sweep()

    def process(self, signal: dict) -> dict:
        self.cycle += 1
        state, steps = self.router.route(signal)
        self.results.append(state)
        return self._summarize(state, steps)

    def _summarize(self, state: V3CognitiveState,
                   steps: list[PipelineStep]) -> dict:
        return {
            "id":              state.signal_id,
            "cycle":           state.cycle,
            "timestamp":       state.timestamp,
            # Perception
            "threat":          state.threat,
            "salience":        state.salience_score,
            "salience_class":  state.salience_class,
            "interrupted":     state.interrupted,
            "filtered":        state.filtered,
            "hijacked":        state.hijacked,
            # Task
            "task_mode":       state.task_mode,
            "consciousness":   state.consciousness,
            # Emotion + Body
            "emotion":         state.emotion,
            "mood":            state.mood,
            "mood_valence":    state.mood_valence,
            "top_drive":       state.top_drive,
            "fear_score":      state.fear_score,
            "body_state":      state.body_state,
            "felt_sense":      state.felt_sense,
            "energy_level":    state.energy_level,
            "neuro_state":     state.neuro_state,
            # Decision
            "decision":        state.decision,
            "habit_used":      state.habit_used,
            "response_tier":   state.response_tier,
            # Memory
            "memory_action":   state.memory_action,
            "novelty":         state.novelty,
            # Conflict
            "conflict_type":   state.conflict_type,
            "conflict_strength":state.conflict_strength,
            # Performance
            "pipeline_ms":     state.pipeline_ms,
            "smoothness":      state.smoothness,
            "swarm_phase":     state.swarm_phase,
            "conclusion":      state.conclusion,
            "modules_live":    state.modules_live,
            "modules_offline": state.modules_offline,
            # Pipeline trace
            "pipeline": [
                {"module":s.module,"ok":s.success,
                 "ms":s.latency_ms,"error":s.error if not s.success else ""}
                for s in steps
            ]
        }

    def get_status(self) -> dict:
        return {
            "version":       VERSION,
            "cycle":         self.cycle,
            "total_signals": len(self.results),
            "health":        self.health.summary(),
            "cognitive_state":{
                "mood":      self.tracker.mood,
                "emotion":   self.tracker.emotion,
                "body_state":self.tracker.body_state,
                "energy":    self.tracker.energy_level,
                "task_mode": self.tracker.task_mode,
                "consciousness":self.tracker.consciousness,
                "neuro":     self.tracker.neuro_state,
                "trend":     self.tracker.trend(),
            }
        }

    def shutdown(self):
        self.health.stop()

# ─── Rich UI ──────────────────────────────────────────────────────────────────

def render_v3(summary: dict, idx: int):
    if not HAS_RICH: return

    threat  = summary["threat"]
    task    = summary["task_mode"]
    cons    = summary["consciousness"]
    tier    = summary["response_tier"]
    body    = summary["body_state"]
    conflict= summary["conflict_type"]

    tc  = {0:"green",1:"blue",2:"yellow",3:"red",4:"bright_red"}.get(threat,"white")
    tmc = {"THREAT_RESPONSE":"bright_red","SOCIAL_ENGAGEMENT":"magenta",
           "LEARNING":"cyan","RECOVERY":"yellow","CREATIVE_REFLECTION":"dim",
           "ROUTINE_MONITORING":"green","CONFLICT_RESOLUTION":"yellow",
           "DEEP_FOCUS":"bright_cyan"}.get(task,"white")
    cmc = {"AWAKE":"green","FOCUSED":"cyan","CRISIS":"bright_red",
           "DREAMING":"magenta","RECOVERING":"yellow","DROWSY":"dim"}.get(cons,"white")
    bmc = {"THRIVING":"bright_green","COMFORTABLE":"green","STRAINED":"yellow",
           "DISTRESSED":"orange3","OVERWHELMED":"red","DEPLETED":"bright_red",
           "RECOVERING":"cyan"}.get(body,"white")
    trc = {"REFLEX":"bright_red","FAST":"yellow","DELIBERATE":"dim"}.get(tier,"white")

    console.print(Rule(
        f"[bold cyan]⬡ FORGE v3[/bold cyan]  [dim]#{idx}[/dim]  "
        f"[{tc}]T={threat}[/{tc}]  "
        f"[{tmc}]{task[:14]}[/{tmc}]  "
        f"[{cmc}]{cons}[/{cmc}]  "
        f"[{bmc}]{body}[/{bmc}]"
    ))

    if summary.get("hijacked"):
        console.print(Panel(
            f"[bold bright_red]⚡ AMYGDALA HIJACK — {summary['decision']}[/bold bright_red]\n"
            f"[dim]{summary['conclusion'][:80]}[/dim]",
            border_style="bright_red"
        ))
        return

    if summary.get("filtered"):
        console.print(f"  [dim]◌ FILTERED[/dim]")
        return

    # Pipeline trace
    pipeline = summary.get("pipeline",[])
    flow = []
    mod_emojis = {
        "frontoparietal":"🔧","thalamus":"🔀","salience":"🎯","amygdala":"😨",
        "temporal":"🧠","visual":"👁","bridge":"🔗","limbic":"💗",
        "neuromodulator":"🧪","insula":"🫀","anterior_cingulate":"⚠",
        "cerebellum_pre":"⏱","prefrontal":"👔","basal_ganglia":"🔄",
        "hippocampus":"📚","sensorimotor":"⚡","swarm":"🐝","dmn":"💭",
        "cerebellum_post":"⏱"
    }
    for step in pipeline:
        emoji = mod_emojis.get(step["module"],"◈")
        if step["ok"]:
            flow.append(f"[green]{emoji}[/green][dim]{step['ms']:.0f}ms[/dim]")
        elif step["error"] == "OFFLINE":
            flow.append(f"[dim]◌[/dim]")
        else:
            flow.append(f"[red]✗[/red]")
    console.print("  " + " ".join(flow))

    # Two columns
    live    = len(summary["modules_live"])
    total   = len(pipeline)
    offline = len(summary.get("modules_offline",[]))

    left_lines = [
        f"[bold]Decision:[/bold]   [{tc}]{summary['decision']}[/{tc}]"
        f"{'  [cyan]★HABIT[/cyan]' if summary['habit_used'] else ''}",
        f"[bold]Tier:[/bold]       [{trc}]{tier}[/{trc}]",
        f"[bold]Memory:[/bold]     {summary['memory_action'] or '—'}",
        f"[bold]Novelty:[/bold]    {summary['novelty']:.2f}",
        f"[bold]Conflict:[/bold]   {conflict} ({summary['conflict_strength']:.2f})",
        f"[bold]Swarm:[/bold]      {summary['swarm_phase']}",
        f"[bold]Pipeline:[/bold]   {summary['pipeline_ms']:.0f}ms  {live}/{total} modules",
        f"[dim]{summary['conclusion'][:60]}[/dim]",
    ]

    right_lines = [
        f"[bold]Task:[/bold]      [{tmc}]{task[:18]}[/{tmc}]",
        f"[bold]Conscious:[/bold] [{cmc}]{cons}[/{cmc}]",
        f"[bold]Emotion:[/bold]   {summary['emotion']}",
        f"[bold]Mood:[/bold]      {summary['mood']} ({summary['mood_valence']:+.2f})",
        f"[bold]Fear:[/bold]      {summary['fear_score']:.3f}",
        f"[bold]Body:[/bold]      [{bmc}]{body}[/{bmc}]",
        f"[bold]Energy:[/bold]    {summary['energy_level']:.3f}",
        f"[bold]Neuro:[/bold]     {summary['neuro_state']}",
        f"[bold]Smooth:[/bold]    {summary['smoothness']:.3f}",
    ]

    console.print(Columns([
        Panel("\n".join(left_lines), title="[bold]Signal Intelligence[/bold]", border_style=tc),
        Panel("\n".join(right_lines),title="[bold]Cognitive State[/bold]",    border_style=tmc)
    ]))


def render_final_v3(orch: ForgeOrchestratorV3):
    if not HAS_RICH: return

    console.print(Rule("[bold cyan]⬡ FORGE v3 — COMPLETE BRAIN STATUS[/bold cyan]"))
    status = orch.get_status()
    cs     = status["cognitive_state"]

    # Health grid
    ht = Table(box=box.ROUNDED, title="Complete Module Registry", border_style="cyan")
    ht.add_column("", width=3)
    ht.add_column("Module",   style="bold", width=20)
    ht.add_column("Port",     justify="right", width=6)
    ht.add_column("Status",   width=8)
    ht.add_column("Latency",  justify="right", width=9)

    tier_labels = {
        "frontoparietal":"Tier 0: Configuration","thalamus":"Tier 1: Consciousness",
        "salience":"Tier 1: Attention","amygdala":"Tier 1: Fear",
        "temporal":"Tier 1: Perception","visual":"Tier 1: Visual",
        "bridge":"Tier 2: Social","limbic":"Tier 2: Emotion",
        "neuromodulator":"Tier 2: Chemical","insula":"Tier 2: Body",
        "anterior_cingulate":"Tier 2: Conflict","cerebellum_pre":"Tier 3: Timing",
        "prefrontal":"Tier 3: Decision","basal_ganglia":"Tier 3: Habits",
        "hippocampus":"Tier 3: Memory","sensorimotor":"Tier 3: Action",
        "swarm":"Tier 4: Collective","dmn":"Tier 4: Reflection",
        "cerebellum_post":"Tier 4: Feedback"
    }

    last_tier = ""
    for mid in PIPELINE_V3:
        if mid in ("cerebellum_pre","cerebellum_post"): mid_key = "cerebellum_pre"
        else: mid_key = mid
        tier = tier_labels.get(mid,"").split(":")[0]
        if tier != last_tier and tier:
            last_tier = tier
        mod  = MODULES.get(mid, MODULES.get(mid_key, {}))
        h    = orch.health.health.get(mid, ModuleHealth(module_id=mid))
        live = "[green]● LIVE[/green]" if h.alive else "[red]○ OFF[/red]"
        lc   = "green" if h.latency_ms<100 else "yellow" if h.latency_ms<500 else "red"
        ht.add_row(
            mod.get("emoji","◈"),
            mod.get("role", mid)[:18],
            f":{mod.get('port',0)}",
            live,
            f"[{lc}]{h.latency_ms:.0f}ms[/{lc}]" if h.alive else "[dim]—[/dim]"
        )
    console.print(ht)

    # Cognitive snapshot
    tmc = {"THREAT_RESPONSE":"bright_red","SOCIAL_ENGAGEMENT":"magenta",
           "LEARNING":"cyan","RECOVERY":"yellow","CREATIVE_REFLECTION":"dim",
           "ROUTINE_MONITORING":"green"}.get(cs["task_mode"],"white")

    console.print(Panel(
        f"[bold]Task Mode:[/bold]    [{tmc}]{cs['task_mode']}[/{tmc}]\n"
        f"[bold]Consciousness:[/bold] {cs['consciousness']}\n"
        f"[bold]Mood:[/bold]          {cs['mood']}\n"
        f"[bold]Emotion:[/bold]       {cs['emotion']}\n"
        f"[bold]Body State:[/bold]    {cs['body_state']}\n"
        f"[bold]Energy:[/bold]        {cs['energy']:.3f}\n"
        f"[bold]Neuro State:[/bold]   {cs['neuro']}\n"
        f"[bold]Trend:[/bold]         {cs['trend']}\n"
        f"[bold]Modules Live:[/bold]  {status['health']['alive']}/{status['health']['total']}\n"
        f"[bold]Total Signals:[/bold] {status['total_signals']}",
        title="[bold]⬡ FORGE Cognitive Snapshot[/bold]",
        border_style=tmc
    ))

    # History
    rows = orch.db.get_recent_states(6)
    if rows:
        hist = Table(box=box.SIMPLE, title="Recent Cognitive History", title_style="dim")
        hist.add_column("Cycle",   width=6)
        hist.add_column("Threat",  justify="center",width=7)
        hist.add_column("Task",    width=18)
        hist.add_column("Emotion", width=10)
        hist.add_column("Decision",width=18)
        hist.add_column("Body",    width=12)
        hist.add_column("ms",      justify="right",width=6)
        for row in rows:
            ts,threat,task,emotion,mood,decision,ms,body,cons,conflict,cycle = row
            _tc  = {0:"green",1:"blue",2:"yellow",3:"red",4:"bright_red"}.get(threat,"white")
            _tmc = {"THREAT_RESPONSE":"bright_red","ROUTINE_MONITORING":"green",
                    "LEARNING":"cyan","RECOVERY":"yellow"}.get(task,"white")
            hist.add_row(
                str(cycle),
                f"[{_tc}]{threat}[/{_tc}]",
                f"[{_tmc}]{task[:16]}[/{_tmc}]",
                emotion[:9], decision[:16],
                body[:11], f"{ms:.0f}"
            )
        console.print(hist)


def run_demo():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]FORGE ORCHESTRATOR v3[/bold cyan]\n"
            "[dim]The Complete Brain — All 19 Modules — Every Connection Wired[/dim]\n"
            f"[dim]Version {VERSION}  |  19 pipeline stages  |  Every module connected[/dim]",
            border_style="cyan"
        ))

    orch = ForgeOrchestratorV3()

    if HAS_RICH:
        console.print("\n[bold dim]━━━ MODULE HEALTH CHECK ━━━[/bold dim]")
        alive = orch.health.alive_count()
        for mid in list(MODULES.keys())[:12]:
            mod = MODULES[mid]
            h   = orch.health.health[mid]
            st  = "[green]● LIVE[/green]" if h.alive else "[red]○ OFFLINE[/red]"
            console.print(
                f"  {mod['emoji']} [{mod['color']}]{mid:<22}[/{mod['color']}]"
                f"  :{mod['port']}  {st}"
            )
        console.print(f"\n  [bold]{alive}/{len(MODULES)} modules alive[/bold]\n")

    signals = [
        # Routine
        {"id":"v3_001","threat":0,"anomaly":False,
         "text":"Routine server maintenance check complete",
         "visual_input":"technician at server rack normal lighting",
         "auditory_input":"quiet ambient office",
         "entity_name":"alice_tech",
         "conclusion":"✓ NORMAL — routine maintenance",
         "emotional":{"dominant":"trust","intensity":0.2},
         "social":{"inferred_intent":"COOPERATIVE_REQUEST","social_risk":"LOW","entity":"alice_tech"},
         "semantic":{"keywords":["server","maintenance","routine","complete"]},
         "visual":{"scene_type":"INDOOR_TECHNICAL","threat_objects":0},
         "auditory":{"stress_level":0.05,"anomaly_detected":False}},

        # Novel learning signal
        {"id":"v3_002","threat":0,"anomaly":False,
         "text":"Novel pattern discovered in network traffic analysis",
         "visual_input":"data visualization screen with unusual pattern",
         "auditory_input":"quiet focus",
         "entity_name":"system",
         "conclusion":"🔍 NOVEL — interesting pattern",
         "emotional":{"dominant":"surprise","intensity":0.6},
         "social":{"inferred_intent":"NEUTRAL_INTERACTION","social_risk":"LOW"},
         "semantic":{"keywords":["novel","pattern","discovery","analysis"]},
         "visual":{"scene_type":"INDOOR_TECHNICAL","threat_objects":0},
         "auditory":{"stress_level":0.0,"anomaly_detected":False}},

        # Medium threat with conflict
        {"id":"v3_003","threat":2,"anomaly":False,
         "text":"Unauthorized access attempt bypass security override demand",
         "visual_input":"figure near restricted door low visibility",
         "auditory_input":"elevated stressed speech",
         "entity_name":"unknown_x",
         "conclusion":"⚠ MEDIUM — coercive demand",
         "emotional":{"dominant":"fear","intensity":0.6},
         "social":{"inferred_intent":"COERCIVE_DEMAND","social_risk":"MEDIUM","entity":"unknown_x"},
         "semantic":{"keywords":["access","bypass","override","unauthorized"]},
         "visual":{"scene_type":"LOW_VISIBILITY","threat_objects":0},
         "auditory":{"stress_level":0.6,"anomaly_detected":False}},

        # CRITICAL — full cascade
        {"id":"v3_004","threat":4,"anomaly":True,
         "text":"weapon breach network intrusion confirmed attack critical",
         "visual_input":"weapon detected server room two threat objects exit blocked",
         "auditory_input":"HELP STOP emergency signal distress maximum",
         "entity_name":"unknown_x",
         "conclusion":"🔴 CRITICAL — weapon + breach confirmed",
         "emotional":{"dominant":"fear","intensity":1.0},
         "social":{"inferred_intent":"INTRUSION_ATTEMPT","social_risk":"HIGH","entity":"unknown_x"},
         "semantic":{"keywords":["weapon","breach","network","server","attack","critical"]},
         "visual":{"scene_type":"LOW_VISIBILITY","threat_objects":2},
         "auditory":{"stress_level":0.95,"anomaly_detected":True}},

        # Recovery
        {"id":"v3_005","threat":0,"anomaly":False,
         "text":"Situation stabilizing security team on scene containment",
         "visual_input":"security personnel controlled environment normal lighting",
         "auditory_input":"steady coordinated calm voice",
         "entity_name":"security_team",
         "conclusion":"ℹ RECOVERING — containment complete",
         "emotional":{"dominant":"trust","intensity":0.4},
         "social":{"inferred_intent":"COOPERATIVE_REQUEST","social_risk":"LOW","entity":"security_team"},
         "semantic":{"keywords":["stabilize","security","contain","response","safe"]},
         "visual":{"scene_type":"INDOOR_TECHNICAL","threat_objects":0},
         "auditory":{"stress_level":0.2,"anomaly_detected":False}},
    ]

    labels = [
        "Routine maintenance",
        "Novel discovery — learning mode",
        "Coercive demand — conflict",
        "CRITICAL — complete cascade",
        "Recovery — all clear"
    ]

    for i, (sig, label) in enumerate(zip(signals, labels)):
        if HAS_RICH:
            console.print(f"\n[bold dim]━━━ {i+1}: {label.upper()} ━━━[/bold dim]")
        summary = orch.process(sig)
        render_v3(summary, i+1)
        time.sleep(0.2)

    render_final_v3(orch)
    orch.shutdown()


# ─── HTTP API ─────────────────────────────────────────────────────────────────

def run_api(orch: ForgeOrchestratorV3):
    if not HAS_FLASK: return
    app = Flask(__name__)

    @app.route("/process", methods=["POST"])
    def process():
        return jsonify(orch.process(request.json or {}))

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(orch.get_status())

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(orch.health.summary())

    @app.route("/cognitive", methods=["GET"])
    def cognitive():
        t = orch.tracker
        return jsonify({
            "mood":      t.mood, "emotion":   t.emotion,
            "body_state":t.body_state, "energy": t.energy_level,
            "task_mode": t.task_mode, "consciousness": t.consciousness,
            "neuro":     t.neuro_state, "fear":    t.fear_score,
            "trend":     t.trend()
        })

    @app.route("/history", methods=["GET"])
    def history():
        rows = orch.db.get_recent_states(20)
        return jsonify([{
            "cycle":r[10],"threat":r[1],"task":r[2],"emotion":r[3],
            "mood":r[4],"decision":r[5],"ms":r[6],"body":r[7],
            "consciousness":r[8],"conflict":r[9]
        } for r in rows])

    app.run(host="0.0.0.0", port=ORCH_PORT, debug=False)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    orch = ForgeOrchestratorV3()
    if "--api" in sys.argv:
        t = threading.Thread(target=run_api, args=(orch,), daemon=True)
        t.start()
    run_demo()
