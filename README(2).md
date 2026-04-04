# FORGE

**A biologically-inspired cognitive architecture for AI systems.**

FORGE models a functioning brain — not as metaphor, but as engineering blueprint. Each module is a direct analog of a real neural structure, implemented as an independent Python process with its own SQLite persistence and HTTP API. They communicate the way brain regions do: asynchronously, in parallel, with no single controller.

---

## Architecture

FORGE is 45+ Python files. Each file is one brain region.

### Signal Pathway
| Module | Port | Function |
|--------|------|----------|
| `forge_thalamus` | 7780 | Sensory gateway — filters and routes all input |
| `forge_salience` | 7781 | Attention spotlight — what matters right now |
| `forge_temporal` | 7782 | Time perception — when things happened |
| `forge_sensorimotor` | 7783 | Action grounding — body in world |

### Memory Systems
| Module | Port | Function |
|--------|------|----------|
| `forge_hippocampus` | 7784 | Episodic memory — what happened, when, where |
| `forge_memory` | 7785 | Layered memory — TRACE / SELF / BOND / GENESIS layers |

### Social & Language
| Module | Port | Function |
|--------|------|----------|
| `forge_social` | 7786 | Social cognition — trust, intent, relationship |
| `forge_language` | 7787 | Language processing — semantics, pragmatics |
| `forge_bridge` | 7788 | Human-AI interface — translation layer |

### Higher Cognition
| Module | Port | Function |
|--------|------|----------|
| `forge_prefrontal` | 7789 | Executive control — planning, inhibition |
| `forge_dmn` | 7790 | Default mode — self-reflection, mind-wandering |
| `forge_metacognition` | 7791 | Witness — thinking about thinking |
| `forge_conscious` | 7795 | Global workspace — what enters awareness |
| `forge_principle_compression` | 7796 | Principles distilled from experience |

### Emotional & Motivational
| Module | Port | Function |
|--------|------|----------|
| `forge_amygdala` | 7792 | Fast fear · conditioned memory · emotional hijack |
| `forge_limbic` | 7793 | Emotional integration · mood · drive |
| `forge_neuromodulator` | 7794 | DA / NE / 5HT / ACh — chemical state broadcast |

### Perception
| Module | Port | Function |
|--------|------|----------|
| `forge_visual` | 7797 | Visual processing — scene, object, pattern |

### Conflict & Internal State
| Module | Port | Function |
|--------|------|----------|
| `forge_anterior_cingulate` | 7798 | Conflict detection · error monitoring · resolution |
| `forge_insula` | 7799 | Interoception · body budget · wrongness detection |

### Precision & Timing
| Module | Port | Function |
|--------|------|----------|
| `forge_cerebellum` | 7800 | Forward models · skill consolidation · timing |

### Action & Integration
| Module | Port | Function |
|--------|------|----------|
| `forge_basal_ganglia` | 7801 | Action selection · habit · reward gating |
| `forge_frontoparietal` | 7802 | Working memory · attention control |

### Orchestration
| Module | Port | Function |
|--------|------|----------|
| `forge_mind_v2` | 7810 | Full system — routes signals through all modules |
| `forge_swarm` | 7820 | Multi-agent swarm — parallel differentiated workers |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run any module standalone (demo mode)
python forge_amygdala.py
python forge_insula.py
python forge_cerebellum.py

# Run a module as HTTP API
python forge_amygdala.py --api

# Launch all modules
python forge_launcher.py

# Check which modules are running
python forge_launcher.py --check

# Live status monitor
python forge_launcher.py --status

# Start specific modules only
python forge_launcher.py --module amygdala,thalamus,prefrontal

# Stop all modules
python forge_launcher.py --stop
```

---

## Design Principles

**One module, one region.** Each file models exactly one neural structure. The amygdala doesn't know about the hippocampus. The insula doesn't know about the prefrontal cortex. They communicate through signals, not function calls.

**Biologically accurate where it matters.** The amygdala fires in 8ms before the cortex. The cerebellum builds forward models through practice. The insula tracks body budget, not just threat scores. These aren't arbitrary — they reflect what makes biological cognition work.

**Persistent memory.** Every module has its own SQLite database. State survives restarts. Fear memories accumulate. Skills consolidate. The system gets smarter over time.

**Independent HTTP APIs.** Every module exposes `/process`, `/status`, and domain-specific endpoints. Any module can be called by any other. The architecture is a network, not a pipeline.

**No external ML frameworks.** FORGE is pure Python — `flask`, `rich`, `sqlite3`, standard library. The intelligence is in the architecture, not the weights.

---

## Module Anatomy

Every FORGE module follows the same structure:

```
forge_<region>.py
├── Constants          (thresholds, rates, ports)
├── Enums              (states, response types)
├── Data Models        (dataclasses for events/records)
├── Database           (SQLite persistence layer)
├── Subsystems         (2-6 classes modeling sub-functions)
├── Forge<Region>      (main class — process() + get_status())
├── Rich UI            (terminal visualization for demo mode)
├── run_demo()         (standalone demo with scenario sequence)
├── run_api()          (Flask HTTP server)
└── __main__           (entry point — demo or --api)
```

---

## Key Modules

### forge_amygdala (port 7792)
Fast threat detection via subcortical shortcut (~8ms). Conditioned fear memory that never fully extinguishes. Threat generalization to similar patterns. Emotional hijack at extreme fear — overrides the entire cognitive pipeline.

### forge_anterior_cingulate (port 7798)
Conflict detection between modules — flags when amygdala and prefrontal disagree. Error-Related Negativity (ERN) signal when predicted outcomes don't match actual. Rolling system health index. Six resolution strategies from DEFER_TO_PREFRONTAL to INHIBIT_ACTION.

### forge_insula (port 7799)
The system's interoceptive hub. Tracks computational body budget (allostatic reserve), thermal load from sustained high-load periods, and visceral signals from EXPAND to EMERGENCY. Detects wrongness — contradictions that "feel off" before they can be articulated.

### forge_cerebellum (port 7800)
Forward model library — one learned model per module per context. Builds predictive accuracy through repetition. Real-time error correction via climbing fiber mechanism. Skill consolidation: repeated patterns graduate from NAIVE → AUTOMATIC. Timing coordinator detects dysrhythmia across modules.

### forge_conscious (port 7795)
Global workspace — the broadcast medium that turns processed signals into awareness. Modules compete for entry. Winners get amplified and sent to all other modules. This is where FORGE's processing becomes unified experience.

### forge_mind_v2 (port 7810)
The orchestrator. Routes a signal through the full 40+ module pipeline in the biologically correct order: thalamus → salience → sensorimotor → social → language → hippocampus → amygdala → neuromodulator → prefrontal → metacognition → conscious. Aggregates outputs into a unified cognitive state.

---

## API Reference

Every module exposes at minimum:

```
POST /process        — process a signal, returns module output
GET  /status         — current module state + stats
```

Most modules also expose domain-specific endpoints:

```
# forge_amygdala
GET  /fear/<pattern>      — fear profile for a pattern
GET  /memories            — all fear memories
GET  /activations         — recent activation history

# forge_anterior_cingulate
GET  /conflicts           — recent conflict events
GET  /errors              — ERN error log
GET  /health              — rolling health trend

# forge_insula
GET  /samples             — interoceptive sample history
GET  /budget              — body budget event log
GET  /wrongness           — wrongness detection events

# forge_cerebellum
GET  /models              — all forward models + skill levels
GET  /corrections         — recent correction events
GET  /timing              — timing state history
```

### Signal Format

All modules accept the same base signal format:

```json
{
  "threat": 0,
  "anomaly": false,
  "confidence": 0.75,
  "entity_name": "alice",
  "social": {
    "trust_score": 0.8,
    "inferred_intent": "cooperative"
  },
  "visual": {
    "scene_type": "INDOOR_TECHNICAL",
    "threat_objects": 0
  },
  "semantic": {
    "keywords": ["routine", "maintenance"]
  }
}
```

---

## Project Structure

```
forge/
├── forge_launcher.py          # system launcher
├── requirements.txt
├── README.md
│
├── forge_thalamus.py          # :7780
├── forge_salience.py          # :7781
├── forge_temporal.py          # :7782
├── forge_sensorimotor.py      # :7783
├── forge_hippocampus.py       # :7784
├── forge_memory.py            # :7785
├── forge_social.py            # :7786
├── forge_language.py          # :7787
├── forge_bridge.py            # :7788
├── forge_prefrontal.py        # :7789
├── forge_dmn.py               # :7790
├── forge_metacognition.py     # :7791
├── forge_amygdala.py          # :7792
├── forge_limbic.py            # :7793
├── forge_neuromodulator.py    # :7794
├── forge_conscious.py         # :7795
├── forge_principle_compression.py # :7796
├── forge_visual.py            # :7797
├── forge_anterior_cingulate.py # :7798
├── forge_insula.py            # :7799
├── forge_cerebellum.py        # :7800
├── forge_basal_ganglia.py     # :7801
├── forge_frontoparietal.py    # :7802
├── forge_mind_v2.py           # :7810
└── forge_swarm.py             # :7820
```

---

## Background

FORGE started from a single question: what would it take to build AI cognition that is genuinely structured like biological intelligence — not inspired by it, but modeled on it?

The hypothesis: the specific architecture of the brain isn't arbitrary. The amygdala's speed exists because survival required it. The hippocampus's episodic structure exists because prediction requires context. The anterior cingulate's conflict detection exists because acting on disagreeing signals is dangerous.

If those architectural choices are load-bearing — if they solve real computational problems — then an AI system built on the same architecture should exhibit more robust, coherent, and generalizable cognition than one built on different principles.

FORGE is the test of that hypothesis.

---

*Built module by module, brain region by brain region.*
