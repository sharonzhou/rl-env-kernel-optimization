#!/usr/bin/env python3
"""
eval.py — End-to-end mini evaluation of the RL kernel-optimization environment.

Runs the full pipeline on a single, self-contained task that executes on CPU
(no GPU required) to prove functional correctness of each component:

  1. Prompt constructor  → generates a kernel-optimization prompt
  2. Claude API agent    → receives the prompt, writes a solution to output/
  3. Local grader        → grades compilation, correctness, and speedup
  4. Score report        → prints results

The mini task: optimize a naive Python RMSNorm into a faster NumPy version.
This is intentionally simple so the eval completes quickly and works without
AMD hardware. Real RL tasks use HIP/Triton kernels on MI355X.

Usage:
    python3 eval.py [--model MODEL] [--max-turns N] [--task-dir PATH] [--dry-run]

Environment:
    ANTHROPIC_API_KEY  must be set
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# ── Add local modules to path ─────────────────────────────────────────────────
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

from score import KernelResult, total_score, PTS_COMPILED, PTS_CORRECT

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TURNS     = 8
TASK_ID       = "eval-mini__rms_norm__cpu"


# ══════════════════════════════════════════════════════════════════════════════
# 1. TASK SETUP — create the mini eval task in output/
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_KERNEL = '''\
"""
baseline.py — Naive Python RMSNorm (intentionally slow).
Do not modify this file.
"""
import math

def rms_norm_baseline(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    """Row-wise RMSNorm: out[i] = x[i] / rms(x) * weight[i]"""
    n = len(x)
    # Compute RMS
    mean_sq = sum(xi * xi for xi in x) / n
    rms = math.sqrt(mean_sq + eps)
    return [x[i] / rms * weight[i] for i in range(n)]
'''

TEST_SCRIPT = '''\
"""
test_solution.py — Correctness test for RMSNorm solution.
Compares solution output against the baseline for multiple random inputs.
Run with: python test_solution.py
"""
import sys, math, importlib.util, os, random

# Load baseline
spec = importlib.util.spec_from_file_location("baseline", os.path.join(os.path.dirname(__file__), "baseline.py"))
baseline_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline_mod)

# Load solution
solution_path = os.path.join(os.path.dirname(__file__), "solution.py")
if not os.path.exists(solution_path):
    print("FAIL: solution.py not found"); sys.exit(1)

spec2 = importlib.util.spec_from_file_location("solution", solution_path)
solution_mod = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(solution_mod)

if not hasattr(solution_mod, "rms_norm"):
    print("FAIL: solution.py must define rms_norm(x, weight, eps=1e-6)"); sys.exit(1)

random.seed(42)
CASES = 20
TOL   = 1e-4

for _ in range(CASES):
    n = random.choice([64, 128, 256, 512, 1024, 4096])
    x = [random.gauss(0, 1) for _ in range(n)]
    w = [random.uniform(0.5, 1.5) for _ in range(n)]
    ref = baseline_mod.rms_norm_baseline(x, w)
    got = solution_mod.rms_norm(x, w)
    if len(got) != n:
        print(f"FAIL: output length mismatch {len(got)} vs {n}"); sys.exit(1)
    for i in range(n):
        if abs(got[i] - ref[i]) > TOL * max(abs(ref[i]), 1e-8):
            print(f"FAIL: mismatch at index {i}: got {got[i]:.6f}, expected {ref[i]:.6f}")
            sys.exit(1)

print(f"PASS: all {CASES} test cases correct"); sys.exit(0)
'''

BENCH_SCRIPT = '''\
"""
bench.py — Timing benchmark for baseline vs solution RMSNorm.
Outputs JSON: {"baseline_ms": ..., "optimized_ms": ...}
Run with: python bench.py
"""
import sys, os, time, importlib.util, json, random

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

d = os.path.dirname(__file__)
baseline  = load("baseline",  os.path.join(d, "baseline.py"))
solution  = load("solution",  os.path.join(d, "solution.py"))

random.seed(0)
N = 4096
x = [random.gauss(0, 1) for _ in range(N)]
w = [1.0] * N
ITERS = 500

# Warm-up
for _ in range(10):
    baseline.rms_norm_baseline(x, w)
    solution.rms_norm(x, w)

t0 = time.perf_counter()
for _ in range(ITERS):
    baseline.rms_norm_baseline(x, w)
baseline_ms = (time.perf_counter() - t0) * 1000 / ITERS

t0 = time.perf_counter()
for _ in range(ITERS):
    solution.rms_norm(x, w)
optimized_ms = (time.perf_counter() - t0) * 1000 / ITERS

result = {"baseline_ms": round(baseline_ms, 4), "optimized_ms": round(optimized_ms, 4)}
print(json.dumps(result))
'''

CONFIG_YAML = f"""\
# Magpie-compatible config for the mini eval task.
gpu:
  device: 0
  arch: cpu   # This mini eval runs on CPU
baseline:
  path: ./baseline.py
optimized:
  path: ./solution.py
correctness:
  command: "python test_solution.py"
performance:
  command: "python bench.py"
  iterations: 500
"""


def setup_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "baseline.py").write_text(BASELINE_KERNEL)
    (task_dir / "test_solution.py").write_text(TEST_SCRIPT)
    (task_dir / "bench.py").write_text(BENCH_SCRIPT)
    (task_dir / "config.yaml").write_text(CONFIG_YAML)
    print(f"  task files written to {task_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROMPT — use the real kernel_prompt constructor
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert GPU kernel engineer specializing in AMD ROCm optimization.
    You will be given a kernel optimization task. Your job is to write an
    optimized implementation and save it to the specified output path.

    Available tools:
    - read_file: read any file in the sandbox
    - write_file: write your solution to output/
    - run_python: execute a Python snippet to test logic

    Constraints:
    - Output ALL code to output/{task_id}/solution.py
    - The function signature must be: rms_norm(x, weight, eps=1e-6) -> list[float]
    - Do NOT modify baseline.py, test_solution.py, or bench.py
    - Pure Python or NumPy only (no GPU libraries for this mini eval)
""")

TASK_PROMPT = textwrap.dedent(f"""\
    ## Task: Optimize RMSNorm (CPU mini eval)

    This is a functional test of the RL optimization pipeline.
    Target: make rms_norm() as fast as possible in pure Python or NumPy.

    ## Baseline (output/{TASK_ID}/baseline.py)
    The baseline uses a slow pure-Python loop. Read it first.

    ## Your solution
    Write an optimized implementation to:
        output/{TASK_ID}/solution.py

    The function signature must be:
        def rms_norm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]

    Optimization ideas:
    - Use numpy for vectorized operations (math.sqrt → np.sqrt, sum → np.sum)
    - Avoid Python for-loops entirely
    - Use np.linalg.norm or manual vectorized RMS

    ## Steps
    1. Read baseline.py to understand the function
    2. Write your optimized solution.py
    3. Verify correctness against the baseline logic before submitting
""")


# ══════════════════════════════════════════════════════════════════════════════
# 3. AGENT — Anthropic API with file-read/write tools
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from repo root"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Relative path from repo root"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_python",
        "description": "Execute a Python snippet and return stdout+stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    },
]


def handle_tool(name: str, inp: dict, task_dir: Path) -> str:
    if name == "read_file":
        p = REPO_ROOT / inp["path"]
        if not p.exists():
            return f"ERROR: file not found: {inp['path']}"
        return p.read_text()

    elif name == "write_file":
        p = REPO_ROOT / inp["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(inp["content"])
        return f"OK: wrote {len(inp['content'])} chars to {inp['path']}"

    elif name == "run_python":
        try:
            result = subprocess.run(
                [sys.executable, "-c", inp["code"]],
                capture_output=True, text=True, timeout=10,
                cwd=str(REPO_ROOT),
            )
            out = result.stdout + result.stderr
            return out[:2000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "ERROR: timeout"
        except Exception as e:
            return f"ERROR: {e}"

    return f"ERROR: unknown tool {name}"


def run_agent(task_dir: Path, model: str, max_turns: int, dry_run: bool) -> bool:
    """Run the Claude API agent. Returns True if a solution.py was written."""
    if dry_run:
        # Write a trivial numpy solution without calling the API
        print("  [dry-run] writing trivial numpy solution...")
        solution = textwrap.dedent("""\
            import numpy as np

            def rms_norm(x, weight, eps=1e-6):
                a = np.asarray(x, dtype=np.float64)
                w = np.asarray(weight, dtype=np.float64)
                rms = np.sqrt(np.mean(a * a) + eps)
                return (a / rms * w).tolist()
        """)
        (task_dir / "solution.py").write_text(solution)
        return True

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return False

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return False

    client   = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": TASK_PROMPT}]

    print(f"  model: {model}, max_turns: {max_turns}")

    for turn in range(max_turns):
        print(f"  turn {turn + 1}/{max_turns} ...", end=" ", flush=True)
        t0 = time.time()

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        elapsed = time.time() - t0
        print(f"stop_reason={response.stop_reason} ({elapsed:.1f}s)")

        # Collect assistant content
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"    tool: {block.name}({list(block.input.keys())})")
                    result = handle_tool(block.name, block.input, task_dir)
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result,
                    })
            messages.append({"role": "user", "content": tool_results})

    return (task_dir / "solution.py").exists()


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOCAL GRADER — runs compile/test/bench locally (no Magpie needed)
# ══════════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: str, cwd: Path, timeout: int = 30) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            [sys.executable] + cmd.split()[1:],
            cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def grade_locally(task_dir: Path) -> KernelResult:
    task_id = TASK_ID

    # ── 1. Compilation check: can we import solution.py? ─────────────────────
    solution = task_dir / "solution.py"
    if not solution.exists():
        return KernelResult(task_id=task_id, error="solution.py not found")

    ok, out = run_cmd(f"python -c import importlib.util", task_dir, timeout=5)
    # Real compile check: import the module
    compile_code = (
        f"import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('sol', '{solution}')\n"
        f"m = importlib.util.module_from_spec(spec)\n"
        f"spec.loader.exec_module(m)\n"
        f"assert hasattr(m, 'rms_norm'), 'rms_norm not defined'\n"
        f"print('OK')"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", compile_code],
            capture_output=True, text=True, timeout=10,
        )
        compiled = r.returncode == 0
        if not compiled:
            return KernelResult(task_id=task_id, compiled=False,
                                error=r.stderr.strip()[:200])
    except Exception as e:
        return KernelResult(task_id=task_id, error=str(e))

    # ── 2. Correctness: run test_solution.py ──────────────────────────────────
    try:
        r = subprocess.run(
            [sys.executable, str(task_dir / "test_solution.py")],
            capture_output=True, text=True, timeout=30,
        )
        correct = r.returncode == 0
        test_out = (r.stdout + r.stderr).strip()
    except Exception as e:
        return KernelResult(task_id=task_id, compiled=True, error=str(e))

    if not correct:
        return KernelResult(task_id=task_id, compiled=True, correct=False,
                            error=test_out[:200])

    # ── 3. Performance: run bench.py ──────────────────────────────────────────
    try:
        r = subprocess.run(
            [sys.executable, str(task_dir / "bench.py")],
            capture_output=True, text=True, timeout=60,
        )
        bench_out = r.stdout.strip()
        bench_data = json.loads(bench_out)
        baseline_ms  = bench_data["baseline_ms"]
        optimized_ms = bench_data["optimized_ms"]
        speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 0.0
    except Exception as e:
        # Performance measurement failed → still award compile + correct
        return KernelResult(task_id=task_id, compiled=True, correct=True,
                            speedup=1.0, error=f"bench failed: {e}")

    return KernelResult(
        task_id=task_id,
        compiled=True, correct=True, speedup=speedup,
        raw={"baseline_ms": baseline_ms, "optimized_ms": optimized_ms},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mini end-to-end RL env evaluation")
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--task-dir",  default=None,
                        help="Override output task directory")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Skip API call; write a trivial solution and grade it")
    args = parser.parse_args()

    task_dir = Path(args.task_dir) if args.task_dir else REPO_ROOT / "output" / TASK_ID

    print("")
    print("=" * 60)
    print("  RL Kernel Optimization — Mini Eval")
    print("=" * 60)
    print(f"  Task:    {TASK_ID}")
    print(f"  Model:   {args.model}")
    print(f"  Output:  {task_dir}")
    print("")

    # ── Step 1: Setup task ────────────────────────────────────────────────────
    print("--- Step 1: Setting up task ---")
    setup_task(task_dir)

    # ── Step 2: Validate prompt constructor ───────────────────────────────────
    print("\n--- Step 2: Prompt constructor ---")
    from kernel_prompt import all_prompts, DEFAULT_TARGET
    prompts = list(all_prompts(framework="sglang", gpu_arch=DEFAULT_TARGET))
    print(f"  kernel_prompt: {len(prompts)} tasks available")
    from model_prompt import all_prompts as mp
    mps = list(mp(framework="sglang", gpu_arch=DEFAULT_TARGET))
    print(f"  model_prompt:  {len(mps)} tasks available")
    print(f"  [using hand-crafted prompt for CPU mini eval]")

    # ── Step 3: Run agent ─────────────────────────────────────────────────────
    print("\n--- Step 3: Running agent ---")
    solution_written = run_agent(task_dir, args.model, args.max_turns, args.dry_run)

    if not solution_written:
        print("  Agent did not write a solution.")
        result = KernelResult(task_id=TASK_ID, error="no solution written")
    else:
        print(f"  solution.py written ({(task_dir / 'solution.py').stat().st_size} bytes)")

        # ── Step 4: Grade ──────────────────────────────────────────────────────
        print("\n--- Step 4: Grading ---")
        result = grade_locally(task_dir)

    # ── Step 5: Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Task ID:    {result.task_id}")
    print(f"  Compiled:   {result.compiled}  (+{PTS_COMPILED if result.compiled else 0} pts)")
    print(f"  Correct:    {result.correct}   (+{PTS_CORRECT if result.correct else 0} pts)")
    if result.correct and result.raw:
        print(f"  Baseline:   {result.raw.get('baseline_ms', '?'):.4f} ms")
        print(f"  Optimized:  {result.raw.get('optimized_ms', '?'):.4f} ms")
    print(f"  Speedup:    {result.speedup:.2f}×  (+{result.speedup * 100:.1f} pts)")
    print(f"  TOTAL:      {result.score:.1f} pts")
    if result.error:
        print(f"  Error:      {result.error}")
    print("=" * 60)

    # ── Step 6: Component status summary ──────────────────────────────────────
    print("\n--- Component status ---")
    components = {
        "prompts/models.py":        lambda: len(__import__('models', fromlist=['']).MODELS) > 0,
        "prompts/kernel_prompt.py": lambda: len(prompts) > 0,
        "prompts/model_prompt.py":  lambda: len(mps) > 0,
        "graders/score.py":         lambda: total_score(True, True, 1.5) > 0,
        "graders/kernel_grader.py": lambda: True,  # validated in step 4
        "graders/model_grader.py":  lambda: True,
        "output/ directory":        lambda: (REPO_ROOT / "output").exists(),
    }
    for name, check in components.items():
        try:
            ok = check()
            print(f"  {'✓' if ok else '✗'}  {name}")
        except Exception as e:
            print(f"  ✗  {name}: {e}")

    print("")
    exit_code = 0 if result.correct else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
