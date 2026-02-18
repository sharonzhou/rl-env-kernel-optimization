#!/usr/bin/env bash
# setup_tools.sh
# Downloads and installs tools/MCPs/skills for the RL kernel-optimization sandbox.
#
# Tools installed:
#   1. Magpie  — GPU kernel correctness & performance evaluation (has its own MCP server)
#   2. RAG tool — finds most relevant files for a given kernel being optimized

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

# Magpie's built-in MCP server lives at magpie/mcp/.
# Register it for Claude Code:
MAGPIE_MCP_DIR="$MAGPIE_DIR/mcp"
echo ""
echo "  Magpie MCP server: $MAGPIE_MCP_DIR"
echo "  To start it manually:"
echo "    python3 -m magpie.mcp  (or see \$MAGPIE_MCP_DIR for details)"

# ── 2. RAG Tool ───────────────────────────────────────────────────────────────

echo ""
echo "=== 2. RAG tool — finds relevant files for a kernel being optimized ==="

RAG_DIR="$SCRIPT_DIR/rag_tool"
mkdir -p "$RAG_DIR"

echo "  installing RAG dependencies..."
pip_install \
    chromadb \
    sentence-transformers \
    mcp \
    tiktoken \
    rich

echo "  writing RAG tool sources..."

# ── index.py ──────────────────────────────────────────────────────────────────
cat > "$RAG_DIR/index.py" << 'PYTHON'
#!/usr/bin/env python3
"""
index.py — Build a ChromaDB vector index over the files/ directory.

Usage:
    python3 index.py [--files-dir PATH] [--db-dir PATH]

Indexes .cpp .hip .cu .h .hpp .py .md files from files/code/ and files/docs/.
Run this once after setup_files.sh populates the files/ directory.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.progress import track

console = Console()

INDEXED_EXTENSIONS = {".cpp", ".hip", ".cu", ".h", ".hpp", ".py", ".md", ".rst"}
MAX_CHUNK_CHARS = 1500   # keep chunks small enough for good retrieval
OVERLAP_CHARS   = 200


def chunk_text(text: str, path: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        chunk = text[start:end]
        chunk_id = hashlib.md5(f"{path}:{start}".encode()).hexdigest()
        chunks.append({
            "id":       chunk_id,
            "text":     chunk,
            "metadata": {
                "path":  path,
                "start": start,
                "end":   end,
            },
        })
        if end == len(text):
            break
        start = end - OVERLAP_CHARS
    return chunks


def collect_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in INDEXED_EXTENSIONS:
            # Skip .git and build dirs
            parts = p.parts
            if any(d in parts for d in (".git", "__pycache__", "build", "CMakeFiles")):
                continue
            files.append(p)
    return files


def build_index(files_dir: Path, db_dir: Path):
    console.print(f"[bold]Indexing:[/bold] {files_dir}")
    console.print(f"[bold]Database:[/bold] {db_dir}")

    client = chromadb.PersistentClient(path=str(db_dir))
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
    )
    collection = client.get_or_create_collection(
        name="kernel_files",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    code_dir = files_dir / "code"
    docs_dir = files_dir / "docs"

    all_files: list[Path] = []
    for d in [code_dir, docs_dir]:
        if d.exists():
            all_files.extend(collect_files(d))

    console.print(f"Found [green]{len(all_files)}[/green] files to index.")

    BATCH = 64
    batch_ids, batch_docs, batch_metas = [], [], []

    def flush():
        if batch_ids:
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    for path in track(all_files, description="Indexing files..."):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(files_dir))
        for chunk in chunk_text(text, rel):
            batch_ids.append(chunk["id"])
            batch_docs.append(chunk["text"])
            batch_metas.append(chunk["metadata"])
            if len(batch_ids) >= BATCH:
                flush()
    flush()

    count = collection.count()
    console.print(f"[green]Done.[/green] {count} chunks indexed in {db_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-dir", default=None,
                        help="Path to the files/ directory (default: ../files relative to this script)")
    parser.add_argument("--db-dir", default=None,
                        help="Path to store the ChromaDB database (default: ./chroma_db)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    files_dir  = Path(args.files_dir) if args.files_dir else script_dir.parent.parent / "files"
    db_dir     = Path(args.db_dir)    if args.db_dir    else script_dir / "chroma_db"

    if not files_dir.exists():
        console.print(f"[red]ERROR:[/red] files dir not found: {files_dir}")
        console.print("Run setup_files.sh first.")
        sys.exit(1)

    build_index(files_dir, db_dir)
PYTHON

# ── server.py (MCP) ───────────────────────────────────────────────────────────
cat > "$RAG_DIR/server.py" << 'PYTHON'
#!/usr/bin/env python3
"""
server.py — MCP server for the kernel-optimization RAG tool.

Exposes two tools:
  - find_relevant_files(kernel_code, n_results) → list of relevant file excerpts
  - search_docs(query, n_results) → list of relevant documentation excerpts

Start:
    python3 server.py [--db-dir PATH]

Register in Claude Code (.claude/mcp.json or via `claude mcp add`):
    {
      "mcpServers": {
        "kernel-rag": {
          "command": "python3",
          "args": ["/path/to/tools/rag_tool/server.py"]
        }
      }
    }
"""
import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── setup ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--db-dir", default=None)
args, _ = parser.parse_known_args()

SCRIPT_DIR = Path(__file__).parent
DB_DIR = Path(args.db_dir) if args.db_dir else SCRIPT_DIR / "chroma_db"

if not DB_DIR.exists():
    sys.stderr.write(
        f"[kernel-rag] Database not found at {DB_DIR}. "
        "Run index.py first.\n"
    )
    sys.exit(1)

_client = chromadb.PersistentClient(path=str(DB_DIR))
_embed  = SentenceTransformerEmbeddingFunction(
    model_name="jinaai/jina-embeddings-v2-base-code",
    trust_remote_code=True,
)
_collection = _client.get_collection("kernel_files", embedding_function=_embed)

# ── MCP server ────────────────────────────────────────────────────────────────

app = Server("kernel-rag")


def _format_results(results) -> str:
    ids        = results["ids"][0]
    docs       = results["documents"][0]
    metas      = results["metadatas"][0]
    distances  = results["distances"][0]
    lines = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        score = round(1 - dist, 3)          # cosine similarity
        path  = meta.get("path", "?")
        start = meta.get("start", 0)
        lines.append(f"### Result {i+1}  (score={score})  {path}:{start}")
        lines.append("```")
        lines.append(doc[:800] + ("..." if len(doc) > 800 else ""))
        lines.append("```")
        lines.append("")
    return "\n".join(lines) if lines else "No results found."


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="find_relevant_files",
            description=(
                "Given a HIP/Triton/CUDA kernel (or a description of the operation "
                "being optimized), return the most relevant source files, headers, "
                "and examples from the indexed codebase."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_code": {
                        "type": "string",
                        "description": "The kernel source code or a natural-language description of the operation.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 8).",
                        "default": 8,
                    },
                },
                "required": ["kernel_code"],
            },
        ),
        Tool(
            name="search_docs",
            description=(
                "Search the indexed ROCm, HIP, CK, and Triton documentation "
                "for content relevant to a query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Documentation search query.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 6).",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "find_relevant_files":
        query    = arguments["kernel_code"]
        n        = int(arguments.get("n_results", 8))
        results  = _collection.query(
            query_texts=[query],
            n_results=n,
            where={"path": {"$contains": "code"}},   # prefer source files
        )
        # Fall back to all content if code-only returned nothing
        if not results["ids"][0]:
            results = _collection.query(query_texts=[query], n_results=n)
        return [TextContent(type="text", text=_format_results(results))]

    elif name == "search_docs":
        query   = arguments["query"]
        n       = int(arguments.get("n_results", 6))
        results = _collection.query(
            query_texts=[query],
            n_results=n,
            where={"path": {"$contains": "docs"}},
        )
        if not results["ids"][0]:
            results = _collection.query(query_texts=[query], n_results=n)
        return [TextContent(type="text", text=_format_results(results))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
PYTHON

# ── mcp.json snippet ──────────────────────────────────────────────────────────
cat > "$RAG_DIR/mcp_config_snippet.json" << JSON
{
  "mcpServers": {
    "kernel-rag": {
      "command": "python3",
      "args": ["${RAG_DIR}/server.py"]
    },
    "magpie": {
      "command": "python3",
      "args": ["-m", "magpie.mcp"]
    }
  }
}
JSON

# Substitute actual path into the snippet
sed -i "s|\${RAG_DIR}|${RAG_DIR}|g" "$RAG_DIR/mcp_config_snippet.json"

chmod +x "$RAG_DIR/index.py" "$RAG_DIR/server.py"

echo ""
echo "=== Done ==="
echo ""
echo "Tools installed:"
echo ""
echo "  Magpie"
echo "    dir:    $MAGPIE_DIR"
echo "    MCP:    python3 -m magpie.mcp"
echo "    usage:  magpie --help"
echo ""
echo "  RAG tool"
echo "    dir:    $RAG_DIR"
echo "    index:  python3 $RAG_DIR/index.py     (run after setup_files.sh)"
echo "    MCP:    python3 $RAG_DIR/server.py"
echo "    config: $RAG_DIR/mcp_config_snippet.json"
echo ""
echo "Next steps:"
echo "  1. Run setup_files.sh to populate the files/ directory with code and docs."
echo "  2. Run: python3 $RAG_DIR/index.py"
echo "  3. Add the MCP servers from mcp_config_snippet.json to ~/.claude/mcp.json"
echo "     (or run: claude mcp add)"
