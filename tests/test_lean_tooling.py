from types import SimpleNamespace


def test_mcp_run_code_result_is_normalized_to_existing_verify_schema():
    from matharena.tools.lean_execution import _feedback_from_mcp_result

    result = SimpleNamespace(
        isError=False,
        structuredContent={
            "success": False,
            "timed_out": False,
            "diagnostics": [
                {"severity": "error", "message": "unknown constant"},
                {"severity": "warning", "message": "declaration uses sorry"},
            ],
        },
        content=[],
    )

    assert _feedback_from_mcp_result(result) == {
        "okay": False,
        "errors": ["unknown constant"],
        "warnings": ["declaration uses sorry"],
        "infos": [],
    }


def test_lsp_pool_restarts_failed_worker_once(monkeypatch, tmp_path):
    from matharena.tools import lean_execution

    attempts = []

    class FakeWorker:
        def __init__(self, project, command):
            self.number = len(attempts)
            attempts.append((project, command))

        def check(self, code):
            if self.number == 0:
                raise RuntimeError("worker died")
            return {"okay": True, "errors": [], "warnings": [], "infos": []}

        def close(self):
            pass

    monkeypatch.setattr(lean_execution, "_LeanLspMcpWorker", FakeWorker)
    pool = lean_execution._LeanLspMcpPool(tmp_path, 1, "lean-lsp-mcp")

    assert pool.check("example : True := by trivial")["okay"] is True
    assert len(attempts) == 2
