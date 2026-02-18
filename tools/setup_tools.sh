#!/usr/bin/env bash
# setup_tools.sh
# Downloads and installs tools/MCPs/skills for the RL kernel-optimization sandbox.
#
# Tools installed:
#   1. Magpie  — GPU kernel correctness & performance evaluation (has its own MCP server)
#   2. RAG tool — dependencies for the kernel-file search tool (tools/rag_tool/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
FILES_DIR="$REPO_ROOT/files"

# ── helpers ────────────────────────────────────────────────────────────────────

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found in PATH"; exit 1; }
}

pip_install() {
    python3 -m pip install --quiet "$@"
}

# ── pre-flight ────────────────────────────────────────────────────────────────

need git
need python3
need pip3

echo ""
echo "=== 1. Magpie — GPU kernel evaluation framework ==="

MAGPIE_DIR="$SCRIPT_DIR/magpie"

if [ -d "$MAGPIE_DIR/.git" ]; then
    echo "  [skip] $MAGPIE_DIR already exists"
else
    echo "  cloning AMD-AGI/Magpie..."
    git clone --depth=1 https://github.com/AMD-AGI/Magpie.git "$MAGPIE_DIR"
fi

echo "  installing Magpie..."
pip_install -e "$MAGPIE_DIR"

# ── 2. RAG tool dependencies ──────────────────────────────────────────────────

echo ""
echo "=== 2. RAG tool — installing dependencies ==="

RAG_DIR="$SCRIPT_DIR/rag_tool"
pip_install -r "$RAG_DIR/requirements.txt"

# ── 3. Register MCP servers with Claude Code ─────────────────────────────────

echo ""
echo "=== 3. Registering MCP servers ==="

need claude

claude mcp add --transport stdio magpie -- python3 -m magpie.mcp
echo "  registered: magpie"

claude mcp add --transport stdio kernel-rag -- python3 "$RAG_DIR/server.py"
echo "  registered: kernel-rag"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Run setup_files.sh to populate the files/ directory with code and docs."
echo "  2. Run: python3 $RAG_DIR/index.py   (to build the RAG search index)"
