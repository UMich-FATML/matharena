import asyncio
import atexit
from concurrent.futures import Future
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
import shutil
import tempfile
from pathlib import Path

from matharena.utils import normalize_conversation


DEFAULT_LEAN_ENVIRONMENT = "lean-4.29.0"
LOOGLE_DIR_ENV = "MATHARENA_LOOGLE_DIR"
LOOGLE_INDEX_ENV = "MATHARENA_LOOGLE_INDEX_PATH"
LEAN_PROJECT_ENV = "MATHARENA_LEAN_PROJECT_ROOT"
LEAN_BACKEND_ENV = "MATHARENA_LEAN_BACKEND"
LEAN_LSP_MCP_POOL_SIZE_ENV = "MATHARENA_LEAN_LSP_MCP_POOL_SIZE"
LEAN_LSP_MCP_COMMAND_ENV = "MATHARENA_LEAN_LSP_MCP_COMMAND"
REPO_ROOT = Path(__file__).resolve().parents[3]
COMPARATOR_BIN = REPO_ROOT / "external" / "comparator" / ".lake" / "build" / "bin" / "comparator"
LEAN4EXPORT_BIN = REPO_ROOT / "external" / "lean4export" / ".lake" / "build" / "bin" / "lean4export"
LANDRUN_BIN = REPO_ROOT / "external" / "landrun" / "bin" / "landrun"
COMPARATOR_PROJECT_DIR = REPO_ROOT / "external" / "comparator_project"
COMPARATOR_LOCK = threading.Lock()
COMPARATOR_AXIOMS = ["propext", "Quot.sound", "Classical.choice"]
COMPARATOR_TIMEOUT_WARNING = "Comparator timed out; accepted based on Axle only."
LOOGLE_PROCESS = None
LEAN_EXPLORE_SERVICE = None
LEAN_LSP_MCP_POOL = None
LEAN_LSP_MCP_POOL_CONFIG = None
LOOGLE_LOCK = threading.Lock()
LEAN_EXPLORE_LOCK = threading.Lock()
LEAN_CODE_BLOCK_RE = re.compile(r"```(?:lean)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
LEAN_BY_RE = re.compile(r":=\s*by\b")
LEAN_DECL_NAME_RE = re.compile(
    r"(?m)^\s*(?:theorem|lemma|def|definition|abbrev|opaque|instance)\s+([A-Za-z_][\w.']*)"
)
ADDED_TO_FILE_HEADER = "### Added To File ###"


async def _check_with_axle(content, environment=DEFAULT_LEAN_ENVIRONMENT):
    from axle import AxleClient

    content = "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("import "))
    async with AxleClient() as client:
        return await client.check(content=content, environment=environment, ignore_imports=True, timeout_seconds=600)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}
    error = {}

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except Exception as exc:
            error["value"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def _loogle_bin():
    if path := os.getenv(LOOGLE_DIR_ENV):
        return Path(path).expanduser().resolve() / ".lake" / "build" / "bin" / "loogle"
    raise RuntimeError(f"Set {LOOGLE_DIR_ENV}.")


def _loogle_dir():
    return Path(os.environ[LOOGLE_DIR_ENV]).expanduser().resolve()


def _lean_project_dir():
    path = os.getenv(LEAN_PROJECT_ENV)
    if not path:
        raise RuntimeError(f"Set {LEAN_PROJECT_ENV} to the shared Lean/Mathlib project.")
    project = Path(path).expanduser().resolve()
    if not (project / "lean-toolchain").is_file() or not (project / "lakefile.lean").is_file():
        raise RuntimeError(f"{project} is not a Lean Lake project.")
    return project


class _RemoteEmbeddingClient:
    def __init__(self, endpoint, model):
        self.endpoint = endpoint
        self.model = model

    async def embed(self, texts, is_query=False):
        def request():
            import requests

            response = requests.post(
                self.endpoint,
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings = [item["embedding"] for item in sorted(payload["data"], key=lambda item: item["index"])]
            return type("EmbeddingResponse", (), {"texts": texts, "embeddings": embeddings, "model": self.model})()

        return await asyncio.to_thread(request)


class _RemoteRerankerClient:
    def __init__(self, endpoint, model):
        self.endpoint = endpoint
        self.model = model

    async def rerank(self, query, documents, batch_size=None):
        def request():
            import requests

            response = requests.post(
                self.endpoint,
                json={"model": self.model, "query": query, "documents": documents},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            if "scores" in payload:
                scores = payload["scores"]
            else:
                scores = [0.0] * len(documents)
                for item in payload.get("results", []):
                    scores[item["index"]] = item.get("relevance_score", item.get("score", 0.0))
            return type("RerankerResponse", (), {"query": query, "scores": scores, "model": self.model})()

        return await asyncio.to_thread(request)


def _get_lean_explore_service():
    global LEAN_EXPLORE_SERVICE
    if LEAN_EXPLORE_SERVICE is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logging.getLogger("lean_explore").setLevel(logging.ERROR)
        from lean_explore.search import SearchEngine, Service

        embedding_endpoint = os.getenv("MATHARENA_LEAN_EXPLORE_EMBEDDING_ENDPOINT")
        reranker_endpoint = os.getenv("MATHARENA_LEAN_EXPLORE_RERANK_ENDPOINT")
        engine_kwargs = {"use_local_data": False}
        if embedding_endpoint:
            engine_kwargs.update(
                embedding_client=_RemoteEmbeddingClient(
                    embedding_endpoint,
                    os.getenv("MATHARENA_LEAN_EXPLORE_EMBEDDING_MODEL", "qwen/qwen3-embedding-0.6b"),
                ),
                embedding_model_name=os.getenv(
                    "MATHARENA_LEAN_EXPLORE_EMBEDDING_MODEL", "qwen/qwen3-embedding-0.6b"
                ),
            )
        if reranker_endpoint:
            engine_kwargs.update(
                reranker_client=_RemoteRerankerClient(
                    reranker_endpoint,
                    os.getenv("MATHARENA_LEAN_EXPLORE_RERANK_MODEL", "qwen/qwen3-reranker-0.6b"),
                ),
                reranker_model_name=os.getenv(
                    "MATHARENA_LEAN_EXPLORE_RERANK_MODEL", "qwen/qwen3-reranker-0.6b"
                ),
            )
        LEAN_EXPLORE_SERVICE = Service(SearchEngine(**engine_kwargs))
    return LEAN_EXPLORE_SERVICE


@atexit.register
def _close_lean_explore_service():
    if LEAN_EXPLORE_SERVICE is not None:
        _run_async(LEAN_EXPLORE_SERVICE.engine.engine.dispose())


def _get_loogle_process():
    global LOOGLE_PROCESS
    loogle_bin = _loogle_bin()
    if LOOGLE_PROCESS is None or LOOGLE_PROCESS.poll() is not None:
        LOOGLE_PROCESS = subprocess.Popen(
            [
                "lake",
                "env",
                str(loogle_bin),
                "--json",
                "--interactive",
                "--module",
                "Mathlib",
                "--index-mode",
                "read",
                "--index-file",
                os.environ[LOOGLE_INDEX_ENV],
            ],
            cwd=_lean_project_dir(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        LOOGLE_PROCESS.stdout.readline()
    return LOOGLE_PROCESS


def loogle(query, max_results=10):
    try:
        with LOOGLE_LOCK:
            loogle_process = _get_loogle_process()
            loogle_process.stdin.write(f"{query}\n")
            loogle_process.stdin.flush()
            payload = json.loads(loogle_process.stdout.readline())
    except FileNotFoundError:
        return f"Error: loogle is not installed at {_loogle_bin()}."
    except Exception as exc:
        return f"Error running loogle: {exc}"

    if "error" in payload:
        return payload["error"]

    hits = payload.get("hits", [])[:max_results]
    lines = [payload.get("header", f"Found {payload.get('count', len(hits))} result(s).")]
    for idx, hit in enumerate(hits, start=1):
        lines.append(f"{idx}. {hit['name']} : {hit['type']}")
        if hit.get("module"):
            lines.append(f"   from {hit['module']}")
    return "\n".join(lines)


@atexit.register
def _close_loogle_process():
    if LOOGLE_PROCESS is not None and LOOGLE_PROCESS.poll() is None:
        LOOGLE_PROCESS.terminate()


def _mcp_result_payload(result):
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured.get("result", structured)
    for item in getattr(result, "content", []):
        text = getattr(item, "text", "")
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload.get("result", payload)
    raise RuntimeError("lean-lsp-mcp returned no structured lean_run_code result.")


def _feedback_from_mcp_result(result):
    if getattr(result, "isError", False):
        detail = "; ".join(getattr(item, "text", str(item)) for item in getattr(result, "content", []))
        raise RuntimeError(detail or "lean_run_code failed")
    payload = _mcp_result_payload(result)
    feedback = {"okay": bool(payload.get("success")), "errors": [], "warnings": [], "infos": []}
    if payload.get("timed_out"):
        feedback["errors"].append("Lean elaboration timed out.")
        feedback["okay"] = False
    for diagnostic in payload.get("diagnostics", []):
        message = diagnostic.get("message", str(diagnostic)) if isinstance(diagnostic, dict) else str(diagnostic)
        severity = diagnostic.get("severity", "info") if isinstance(diagnostic, dict) else "info"
        key = "errors" if severity == "error" else "warnings" if severity == "warning" else "infos"
        feedback[key].append(message)
    return feedback


class _LeanLspMcpWorker:
    def __init__(self, project, command):
        self.project = Path(project)
        self.command = command
        self.requests = queue.Queue()
        self.ready = Future()
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        self.ready.result(timeout=120)

    def _thread_main(self):
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            while True:
                try:
                    item = self.requests.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[1].set_exception(exc)

    async def _serve(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.command,
            args=["--lean-project-path", str(self.project)],
            cwd=self.project,
            env=os.environ.copy(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self.ready.set_result(True)
                while True:
                    try:
                        item = self.requests.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    if item is None:
                        return
                    code, result_future = item
                    try:
                        result = await asyncio.wait_for(
                            session.call_tool("lean_run_code", {"code": f"import Mathlib\n\n{code}"}),
                            timeout=600,
                        )
                        result_future.set_result(_feedback_from_mcp_result(result))
                    except BaseException as exc:
                        result_future.set_exception(exc)

    def check(self, code):
        if not self.thread.is_alive():
            raise RuntimeError("lean-lsp-mcp worker exited")
        result = Future()
        self.requests.put((code, result))
        return result.result(timeout=660)

    def close(self):
        self.requests.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=10)


class _LeanLspMcpPool:
    def __init__(self, project, size, command):
        self.project = project
        self.command = command
        self.workers = [None] * size
        self.available = queue.Queue()
        for index in range(size):
            self.available.put(index)

    def check(self, code):
        index = self.available.get()
        try:
            for attempt in range(2):
                try:
                    if self.workers[index] is None:
                        self.workers[index] = _LeanLspMcpWorker(self.project, self.command)
                    return self.workers[index].check(code)
                except Exception:
                    if self.workers[index] is not None:
                        self.workers[index].close()
                    self.workers[index] = None
                    if attempt == 1:
                        raise
        finally:
            self.available.put(index)

    def close(self):
        for worker in self.workers:
            if worker is not None:
                worker.close()


def _get_lean_lsp_mcp_pool():
    global LEAN_LSP_MCP_POOL, LEAN_LSP_MCP_POOL_CONFIG
    project = _lean_project_dir()
    size = int(os.getenv(LEAN_LSP_MCP_POOL_SIZE_ENV, "4"))
    if size < 1:
        raise RuntimeError(f"{LEAN_LSP_MCP_POOL_SIZE_ENV} must be positive.")
    command = os.getenv(LEAN_LSP_MCP_COMMAND_ENV, "lean-lsp-mcp")
    config = (project, size, command)
    if LEAN_LSP_MCP_POOL is None or config != LEAN_LSP_MCP_POOL_CONFIG:
        if LEAN_LSP_MCP_POOL is not None:
            LEAN_LSP_MCP_POOL.close()
        LEAN_LSP_MCP_POOL = _LeanLspMcpPool(project, size, command)
        LEAN_LSP_MCP_POOL_CONFIG = config
    return LEAN_LSP_MCP_POOL


@atexit.register
def _close_lean_lsp_mcp_pool():
    if LEAN_LSP_MCP_POOL is not None:
        LEAN_LSP_MCP_POOL.close()


def lean_explore_search(query, max_results=10):
    try:
        with LEAN_EXPLORE_LOCK:
            service = _get_lean_explore_service()
            response = _run_async(
                service.search(
                    query=str(query),
                    limit=max_results,
                    rerank_top=max_results,
                    packages=["Mathlib", "Lean", "Init", "Std"],
                )
            )
            payload = {
                "count": response.count,
                "results": [
                    {
                        "id": result.id,
                        "name": result.name,
                        "module": result.module,
                        "description": result.informalization or result.docstring,
                    }
                    for result in response.results
                ],
            }
    except Exception as exc:
        return f"Error running LeanExplore: {exc}"

    lines = [f"Found {payload['count']} LeanExplore result(s)."]
    for idx, hit in enumerate(payload["results"], start=1):
        lines.append(f"{idx}. [{hit['id']}] {hit['name']}")
        lines.append(f"   from {hit['module']}")
        if hit.get("description"):
            lines.append(f"   {hit['description'].splitlines()[0]}")
    return "\n".join(lines)


def get_lean_feedback_dict(statement, environment=DEFAULT_LEAN_ENVIRONMENT):
    backend = os.getenv(LEAN_BACKEND_ENV, "axle").strip().lower()
    if backend == "lean-lsp-mcp":
        try:
            return _get_lean_lsp_mcp_pool().check(statement)
        except Exception as exc:
            return {
                "okay": False,
                "errors": [f"lean-lsp-mcp failed after one worker restart: {exc}"],
                "warnings": [],
                "infos": [],
            }
    if backend != "axle":
        return {
            "okay": False,
            "errors": [f"Unsupported {LEAN_BACKEND_ENV} value: {backend}"],
            "warnings": [],
            "infos": [],
        }

    result = None
    retry = 0
    while retry < 5:
        try:
            result = _run_async(_check_with_axle(statement, environment=environment))
            break
        except Exception as exc:
            print(f"Error checking with Axle: {exc}. Retrying...")
            retry += 1
            time.sleep(60)
    if retry == 5 or result is None or not hasattr(result, "lean_messages"):
        return {
            "okay": False,
            "errors": ["Failed to get feedback from Axle after 5 retries."],
            "warnings": [],
            "infos": [],
        }
    feedback = {
        "okay": result.okay,
        "errors": result.lean_messages.errors + result.tool_messages.errors,
        "warnings": result.lean_messages.warnings + result.tool_messages.warnings,
        "infos": result.lean_messages.infos + result.tool_messages.infos,
    }
    feedback["warnings"] = [
        warning for warning in feedback["warnings"] if not warning.startswith("Imports mismatch") and warning != "Using defaults..."
    ]
    return feedback


def format_lean_feedback(feedback):
    valid = feedback["okay"] and not feedback["errors"]
    parts = [f"### Compiles ###\n{feedback['okay']}", f"### Valid Proof ###\n{valid}"]
    for key in ["errors", "warnings", "infos"]:
        part = f"""### {key.capitalize()} ###\n""" + "\n".join(feedback[key])
        parts.append(part)
    return "\n\n".join(parts)


_format_feedback = format_lean_feedback


def _add_to_file_succeeded(content):
    lines = content.splitlines()
    return len(lines) >= 2 and lines[0] == ADDED_TO_FILE_HEADER and lines[1] == "True"


def _persistent_lean_prefix(messages):
    if not messages:
        return ""

    clean_messages = normalize_conversation(messages)
    pending = {}
    fallback = []
    blocks = []
    for message in clean_messages:
        if message["role"] == "assistant" and message.get("type") == "tool_call" and message.get("tool_name") == "add_to_file":
            arguments = message.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            code = arguments.get("code", "").strip()
            if not code:
                continue
            if message.get("tool_call_id") is None:
                fallback.append(code)
            else:
                pending[message["tool_call_id"]] = code
        elif (
            message["role"] == "tool_response"
            and message.get("tool_name") == "add_to_file"
            and _add_to_file_succeeded(message.get("content", ""))
        ):
            if message.get("tool_call_id") is None:
                code = fallback.pop(0) if fallback else ""
            else:
                code = pending.pop(message["tool_call_id"], "")
            if code:
                blocks.append(code)
    return "\n\n".join(blocks)


def _dedupe_lean_blocks(blocks):
    seen = set()
    kept = []
    for block in reversed([block for block in blocks if block.strip()]):
        names = set(LEAN_DECL_NAME_RE.findall(block))
        complete_names = {name for name in names if re.search(rf"\b{name}\b[\s\S]*:=", block)}
        if any(name in seen for name in complete_names):
            continue
        seen.update(complete_names)
        kept.append(block.strip())
    return list(reversed(kept))


def _prepend_persistent_lean(code, messages):
    prefix = _persistent_lean_prefix(messages)
    blocks = _dedupe_lean_blocks([prefix, code])
    return "\n\n".join(blocks)


def get_executed_lean_submission_parts(model_output, formal_statement=None, messages=None):
    lean_blocks = _extract_submission_blocks(model_output)
    persistent_prefix = _persistent_lean_prefix(messages)

    prefix_blocks = []
    if persistent_prefix:
        prefix_blocks.append(persistent_prefix)
    if len(lean_blocks) > 1:
        prefix_blocks.append(lean_blocks[0].strip())
    executed_prefix = "\n\n".join(_dedupe_lean_blocks(prefix_blocks))

    theorem_block = lean_blocks[-1].strip() if lean_blocks else ""
    if formal_statement is not None and theorem_block:
        normalized_formal_statement = _normalize_formal_statement(formal_statement)
        block_match = LEAN_BY_RE.search(theorem_block)
        formal_match = LEAN_BY_RE.search(normalized_formal_statement)
        if block_match is not None and formal_match is not None:
            theorem_block = (
                normalized_formal_statement[: formal_match.start()].rstrip() + "\n" + theorem_block[block_match.start() :]
            )

    return executed_prefix, theorem_block


def verify_lean(code, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None):
    feedback = get_lean_feedback_dict(_prepend_persistent_lean(code, messages), environment=environment)
    return _format_feedback(feedback)


def add_to_file(code, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None):
    feedback = get_lean_feedback_dict(_prepend_persistent_lean(code, messages), environment=environment)
    if feedback["okay"] and not feedback["errors"]:
        return f"{ADDED_TO_FILE_HEADER}\nTrue\n\nAdded to file."
    return f"{ADDED_TO_FILE_HEADER}\nFalse\n\n" + _format_feedback(feedback)


def _normalize_formal_statement(formal_statement):
    if "```lean" in formal_statement:
        lean_blocks = LEAN_CODE_BLOCK_RE.findall(formal_statement)
        if len(lean_blocks) > 0:
            return lean_blocks[-1].strip()
    return formal_statement


def _extract_submission_blocks(model_output):
    lean_blocks = [block.strip() for block in LEAN_CODE_BLOCK_RE.findall(model_output)]
    if len(lean_blocks) == 1:
        model_output = lean_blocks[0]
    elif len(lean_blocks) > 1:
        model_output = lean_blocks[-1]

    theorem_idx = model_output.rfind("\ntheorem")
    if theorem_idx == -1 and model_output.lstrip().startswith("theorem"):
        theorem_idx = model_output.find("theorem")
    if theorem_idx == -1:
        return []

    prefix = model_output[:theorem_idx].strip()
    theorem = model_output[theorem_idx:].strip()
    return [prefix, theorem] if prefix else [theorem]


def _run_comparator_check(model_output, formal_statement, messages=None):
    if not all(
        path.exists()
        for path in [COMPARATOR_BIN, LEAN4EXPORT_BIN, LANDRUN_BIN, COMPARATOR_PROJECT_DIR / "lakefile.lean"]
    ):
        return None

    theorem_name_match = LEAN_DECL_NAME_RE.search(formal_statement)
    if theorem_name_match is None:
        return "Comparator could not extract the theorem name from the formal statement."

    executed_prefix, theorem_block = get_executed_lean_submission_parts(
        model_output, formal_statement=formal_statement, messages=messages
    )
    solution_code = "\n\n".join(block for block in [executed_prefix, theorem_block] if block.strip())

    with COMPARATOR_LOCK:
        with tempfile.TemporaryDirectory(prefix="matharena-comparator-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            shutil.copy2(COMPARATOR_PROJECT_DIR / "lean-toolchain", tmpdir_path / "lean-toolchain")
            shutil.copy2(COMPARATOR_PROJECT_DIR / "lakefile.lean", tmpdir_path / "lakefile.lean")
            manifest_path = COMPARATOR_PROJECT_DIR / "lake-manifest.json"
            if manifest_path.exists():
                shutil.copy2(manifest_path, tmpdir_path / "lake-manifest.json")

            tmp_lake_dir = tmpdir_path / ".lake"
            tmp_lake_dir.mkdir(exist_ok=True)
            shared_packages = COMPARATOR_PROJECT_DIR / ".lake" / "packages"
            if shared_packages.exists():
                os.symlink(shared_packages, tmp_lake_dir / "packages", target_is_directory=True)

            (tmpdir_path / "Challenge.lean").write_text(f"import Mathlib\n\n{formal_statement}\n", encoding="utf-8")
            (tmpdir_path / "Solution.lean").write_text(f"import Mathlib\n\n{solution_code}\n", encoding="utf-8")
            (tmpdir_path / "comparator.json").write_text(
                json.dumps(
                    {
                        "challenge_module": "Challenge",
                        "solution_module": "Solution",
                        "theorem_names": [theorem_name_match.group(1)],
                        "permitted_axioms": COMPARATOR_AXIOMS,
                        "enable_nanoda": False,
                    }
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PATH"] = f"{LANDRUN_BIN.parent}:{LEAN4EXPORT_BIN.parent}:{env.get('PATH', '')}"
            try:
                result = subprocess.run(
                    ["lake", "env", str(COMPARATOR_BIN), "comparator.json"],
                    cwd=tmpdir_path,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            except subprocess.TimeoutExpired:
                return COMPARATOR_TIMEOUT_WARNING

    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "Comparator rejected the submission.").strip()


def get_lean_feedback_dict_with_formal_statement(
    model_output, formal_statement, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None, use_comparator=False
):
    formal_statement = _normalize_formal_statement(formal_statement)
    lean_blocks = _extract_submission_blocks(model_output)
    if len(lean_blocks) == 0:
        return {
            "okay": False,
            "errors": ["No Lean code block found in the model output."],
            "warnings": [],
            "infos": [],
        }

    block_match = LEAN_BY_RE.search(lean_blocks[-1])
    formal_match = LEAN_BY_RE.search(formal_statement)
    if block_match is None or formal_match is None:
        return {
            "okay": False,
            "errors": ["Could not align the final Lean code block with the formal statement."],
            "warnings": [],
            "infos": [],
        }

    lean_blocks[-1] = formal_statement[:formal_match.start()].rstrip() + "\n" + lean_blocks[-1][block_match.start():]
    feedback = get_lean_feedback_dict(_prepend_persistent_lean("\n\n".join(lean_blocks), messages), environment=environment)
    if not feedback["okay"] or feedback["errors"]:
        return feedback

    if use_comparator:
        comparator_error = _run_comparator_check(model_output, formal_statement, messages=messages)
        if comparator_error == COMPARATOR_TIMEOUT_WARNING:
            feedback["warnings"].append(COMPARATOR_TIMEOUT_WARNING)
            return feedback
        if comparator_error:
            feedback["okay"] = False
            feedback["errors"].append(comparator_error)
    return feedback


def verify_lean_with_formal_statement(model_output, formal_statement, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None):
    feedback = get_lean_feedback_dict_with_formal_statement(
        model_output, formal_statement, environment=environment, messages=messages
    )
    return _format_feedback(feedback)

def compiles_with_sorries(statement, environment=DEFAULT_LEAN_ENVIRONMENT):
    feedback = get_lean_feedback_dict(statement, environment=environment)
    return feedback["okay"]


def compiles_with_formal_statement(model_output, formal_statement, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None):
    feedback = get_lean_feedback_dict_with_formal_statement(
        model_output, formal_statement, environment=environment, messages=messages, use_comparator=True
    )
    if feedback["errors"]:
        return False
    return feedback["okay"]


def compiles(statement, environment=DEFAULT_LEAN_ENVIRONMENT):
    feedback = get_lean_feedback_dict(statement, environment=environment)
    if feedback["errors"]:
        return False
    return feedback["okay"]
