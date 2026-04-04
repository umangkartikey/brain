"""
forge_launcher.py — FORGE System Launcher
==========================================
Starts all FORGE brain modules as independent HTTP API processes.

Usage:
    python forge_launcher.py              # start all modules
    python forge_launcher.py --list       # list all modules + ports
    python forge_launcher.py --check      # health check all running modules
    python forge_launcher.py --stop       # stop all modules
    python forge_launcher.py --module amygdala          # start one module
    python forge_launcher.py --module amygdala,thalamus # start specific modules
    python forge_launcher.py --no-ui      # start without rich status display
"""

import os
import sys
import time
import json
import signal
import argparse
import subprocess
import urllib.request
import urllib.error
import threading
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.rule import Rule
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None

# ─── Module Registry ──────────────────────────────────────────────────────────
#
# Each entry: (module_name, port, short_description)
# Ports follow the original module definitions.

MODULES = [
    # ── Core signal pathway ──────────────────────────────────────────────────
    ("forge_thalamus",            7780, "Sensory gateway — filters & routes all input"),
    ("forge_salience",            7781, "Attention spotlight — what matters right now"),
    ("forge_temporal",            7782, "Time perception — when things happened"),
    ("forge_sensorimotor",        7783, "Action grounding — body in world"),

    # ── Memory systems ───────────────────────────────────────────────────────
    ("forge_hippocampus",         7784, "Episodic memory — what happened, when, where"),
    ("forge_memory",              7785, "Layered memory — TRACE/SELF/BOND/GENESIS"),

    # ── Emotional & motivational ─────────────────────────────────────────────
    ("forge_amygdala",            7792, "Fast fear · conditioned memory · hijack"),
    ("forge_limbic",              7793, "Emotional integration · mood · drive"),
    ("forge_neuromodulator",      7794, "DA/NE/5HT/ACh — chemical state broadcast"),

    # ── Social & language ────────────────────────────────────────────────────
    ("forge_social",              7786, "Social cognition — trust, intent, relationship"),
    ("forge_language",            7787, "Language processing — semantics, pragmatics"),
    ("forge_bridge",              7788, "Human-AI interface — translation layer"),

    # ── Higher cognition ─────────────────────────────────────────────────────
    ("forge_prefrontal",          7789, "Executive control — planning, inhibition"),
    ("forge_dmn",                 7790, "Default mode — self-reflection, mind-wandering"),
    ("forge_metacognition",       7791, "Witness — thinking about thinking"),
    ("forge_conscious",           7795, "Global workspace — what enters awareness"),

    # ── Compressed knowledge ─────────────────────────────────────────────────
    ("forge_principle_compression",7796, "Principles distilled from experience"),
    ("forge_visual",              7797, "Visual processing — scene, object, pattern"),

    # ── Conflict & internal state ────────────────────────────────────────────
    ("forge_anterior_cingulate",  7798, "Conflict detection · error monitoring"),
    ("forge_insula",              7799, "Interoception · body budget · wrongness"),

    # ── Precision & timing ───────────────────────────────────────────────────
    ("forge_cerebellum",          7800, "Forward models · skill consolidation · timing"),

    # ── Integration ──────────────────────────────────────────────────────────
    ("forge_basal_ganglia",       7801, "Action selection · habit · reward gating"),
    ("forge_frontoparietal",      7802, "Working memory · attention control"),
    ("forge_mind_v2",             7810, "Full system integration — 40+ module orchestrator"),

    # ── Swarm ────────────────────────────────────────────────────────────────
    ("forge_swarm",               7820, "Multi-agent swarm — parallel workers"),
]

MODULE_MAP   = {name: (port, desc) for name, port, desc in MODULES}
PID_FILE     = ".forge_pids.json"

# ─── Process Management ───────────────────────────────────────────────────────

def load_pids() -> dict:
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_pids(pids: dict):
    with open(PID_FILE, "w") as f:
        json.dump(pids, f, indent=2)

def start_module(name: str, port: int):
    """Launch a module as a subprocess with --api flag."""
    script = f"{name}.py"
    if not os.path.exists(script):
        return None

    proc = subprocess.Popen(
        [sys.executable, script, "--api"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    return proc.pid

def stop_pid(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except (ProcessLookupError, PermissionError):
        return False

def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False

# ─── Health Check ─────────────────────────────────────────────────────────────

def check_health(port: int, timeout: float = 1.5) -> tuple[bool, str]:
    """
    Ping /status endpoint on given port.
    Returns (healthy, info_string).
    """
    url = f"http://localhost:{port}/status"
    try:
        req  = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read())
        cycle = data.get("cycle", data.get("cycles", "?"))
        return True, f"cycle={cycle}"
    except urllib.error.URLError:
        return False, "unreachable"
    except Exception as e:
        return False, str(e)[:30]

def wait_for_module(port: int, timeout: float = 8.0) -> bool:
    """Wait until a module's /status endpoint responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, _ = check_health(port, timeout=0.5)
        if ok:
            return True
        time.sleep(0.3)
    return False

# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_list():
    """Print all registered modules."""
    if HAS_RICH:
        t = Table(box=box.SIMPLE_HEAVY, title="[bold cyan]FORGE Module Registry[/bold cyan]")
        t.add_column("Module",      style="cyan",  width=30)
        t.add_column("Port",        style="yellow", justify="right", width=6)
        t.add_column("Description", style="white")
        for name, port, desc in MODULES:
            t.add_row(name, str(port), desc)
        console.print(t)
        console.print(f"\n[dim]Total: {len(MODULES)} modules[/dim]")
    else:
        for name, port, desc in MODULES:
            print(f"  {name:<35} :{port}  {desc}")
        print(f"\nTotal: {len(MODULES)} modules")


def cmd_check():
    """Health check all modules."""
    if HAS_RICH:
        t = Table(box=box.SIMPLE, title="[bold cyan]FORGE Health Check[/bold cyan]")
        t.add_column("Module",  style="cyan", width=30)
        t.add_column("Port",    justify="right", width=6)
        t.add_column("Status",  width=12)
        t.add_column("Info",    style="dim")

    results = []
    for name, port, _ in MODULES:
        ok, info = check_health(port)
        results.append((name, port, ok, info))
        if HAS_RICH:
            status = "[green]● UP[/green]" if ok else "[red]○ DOWN[/red]"
            t.add_row(name, str(port), status, info)
        else:
            status = "UP  " if ok else "DOWN"
            print(f"  {status}  {name:<35} :{port}  {info}")

    up   = sum(1 for *_, ok, _ in results if ok)
    down = len(results) - up

    if HAS_RICH:
        console.print(t)
        console.print(
            f"\n[green]{up} up[/green]  [red]{down} down[/red]"
            f"  [dim]of {len(results)} modules[/dim]"
        )
    else:
        print(f"\n{up} up, {down} down of {len(results)} modules")


def cmd_start(module_filter: list = None, wait: bool = True):
    """Start modules (all or filtered list)."""
    pids = load_pids()
    targets = []

    for name, port, desc in MODULES:
        if module_filter and name.replace("forge_","") not in module_filter \
                         and name not in module_filter:
            continue
        if not os.path.exists(f"{name}.py"):
            continue
        targets.append((name, port, desc))

    if not targets:
        msg = "No modules found. Are you running from the FORGE directory?"
        print(msg)
        return

    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold cyan]FORGE LAUNCHER[/bold cyan]\n"
            f"[dim]Starting {len(targets)} modules...[/dim]",
            border_style="cyan"
        ))

    started = []
    skipped = []
    failed  = []

    for name, port, desc in targets:
        # Check if already running
        if name in pids and is_running(pids[name]):
            skipped.append((name, port))
            if HAS_RICH:
                console.print(f"  [dim]⟳  {name:<35} :{port}  already running[/dim]")
            continue

        pid = start_module(name, port)
        if pid:
            pids[name] = pid
            started.append((name, port, pid))
            if HAS_RICH:
                console.print(f"  [green]▶  {name:<35}[/green] :{port}  pid={pid}")
            else:
                print(f"  START  {name}  :{port}  pid={pid}")
        else:
            failed.append((name, port))
            if HAS_RICH:
                console.print(f"  [red]✗  {name:<35}[/red] :{port}  script not found")

    save_pids(pids)

    if wait and started:
        if HAS_RICH:
            console.print(f"\n[dim]Waiting for {len(started)} modules to come online...[/dim]")
        else:
            print(f"\nWaiting for modules...")

        ready = 0
        for name, port, pid in started:
            ok = wait_for_module(port, timeout=10.0)
            if ok:
                ready += 1
                if HAS_RICH:
                    console.print(f"  [green]✓  {name}[/green]")
            else:
                if HAS_RICH:
                    console.print(f"  [yellow]?  {name}  (slow start)[/yellow]")

        if HAS_RICH:
            console.print(f"\n[bold green]{ready}/{len(started)} modules ready[/bold green]")


def cmd_stop(module_filter: list = None):
    """Stop running modules."""
    pids = load_pids()
    stopped = 0
    not_running = 0

    for name, port, _ in MODULES:
        if module_filter and name.replace("forge_","") not in module_filter \
                         and name not in module_filter:
            continue
        if name in pids:
            if stop_pid(pids[name]):
                stopped += 1
                if HAS_RICH:
                    console.print(f"  [red]■  {name}[/red]  stopped")
                del pids[name]
            else:
                not_running += 1

    save_pids(pids)

    if HAS_RICH:
        console.print(f"\n[dim]Stopped {stopped} modules[/dim]")
    else:
        print(f"Stopped {stopped} modules")


def cmd_status():
    """Live status display — refreshes every 3 seconds."""
    if not HAS_RICH:
        cmd_check()
        return

    console.print(
        "[bold cyan]FORGE Live Status[/bold cyan]  "
        "[dim]Ctrl+C to exit[/dim]"
    )

    try:
        while True:
            lines = []
            up = 0
            for name, port, desc in MODULES:
                ok, info = check_health(port, timeout=0.8)
                if ok:
                    up += 1
                    lines.append(
                        f"  [green]●[/green] [cyan]{name:<32}[/cyan] "
                        f"[dim]:{port}[/dim]  [green]{info}[/green]"
                    )
                else:
                    lines.append(
                        f"  [red]○[/red] [dim]{name:<32}[/dim] "
                        f"[dim]:{port}[/dim]  [red]{info}[/red]"
                    )

            ts = datetime.now().strftime("%H:%M:%S")
            console.clear()
            console.print(Rule(
                f"[bold cyan]FORGE[/bold cyan]  "
                f"[green]{up}[/green][dim]/{len(MODULES)} up[/dim]  "
                f"[dim]{ts}[/dim]"
            ))
            for line in lines:
                console.print(line)
            time.sleep(3)
    except KeyboardInterrupt:
        console.print("\n[dim]Status monitor stopped.[/dim]")


# ─── Entry Point ──────────────────────────────────────────────────────────────

# Fix Optional_pid type hint
def main():
    parser = argparse.ArgumentParser(
        description="FORGE System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--list",   action="store_true", help="List all modules")
    parser.add_argument("--check",  action="store_true", help="Health check all modules")
    parser.add_argument("--stop",   action="store_true", help="Stop all modules")
    parser.add_argument("--status", action="store_true", help="Live status monitor")
    parser.add_argument("--module", type=str, default=None,
                        help="Comma-separated module names to target")
    parser.add_argument("--no-wait",action="store_true",
                        help="Don't wait for modules to come online")
    args = parser.parse_args()

    module_filter = [m.strip() for m in args.module.split(",")] \
                    if args.module else None

    if args.list:
        cmd_list()
    elif args.check:
        cmd_check()
    elif args.stop:
        cmd_stop(module_filter)
    elif args.status:
        cmd_status()
    else:
        # Default: start
        cmd_start(module_filter, wait=not args.no_wait)

if __name__ == "__main__":
    main()
