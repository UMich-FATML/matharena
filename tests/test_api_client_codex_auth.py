import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from openai import AuthenticationError

import matharena.api_client as api_client_module
from matharena.api_client import (
    APIClient,
    CODEX_RESPONSES_BASE_URL,
    CodexAuthenticationError,
    _CodexAuthSnapshot,
)


def _jwt(claims):
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"e30.{payload}.signature"


def _write_auth(path, *, access_token=None, account_id="account-test", expires_at=None):
    if access_token is None:
        access_token = _jwt({"exp": expires_at if expires_at is not None else time.time() + 3600})
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "id_token": _jwt(
                        {
                            "https://api.openai.com/auth": {
                                "chatgpt_account_id": account_id,
                            }
                        }
                    ),
                    "refresh_token": "unused-refresh-token",
                },
            }
        ),
        encoding="utf-8",
    )
    return access_token


def _codex_client(auth_path, **overrides):
    config = {
        "model": "gpt-5.6-sol--xhigh",
        "api": "openai",
        "codex_auth_token": str(auth_path),
        "use_openai_responses_api": True,
        "batch_processing": False,
        "background": False,
        "store": False,
        "sleep_after_request": 0,
    }
    config.update(overrides)
    return APIClient(**config)


@pytest.mark.parametrize("codex_config", [{}, {"codex_auth_token": None}], ids=["absent", "null"])
def test_api_key_is_used_when_codex_auth_token_is_not_set(monkeypatch, codex_config):
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-test")

    client = APIClient(model="gpt-test", api="openai", **codex_config)

    assert client.api_key == "api-key-test"
    assert client.codex_auth_token is None


def test_codex_auth_streams_directly_and_takes_precedence_over_api_key(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    access_token = _write_auth(auth_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    captured = {}

    class FakeResponse:
        usage = {"input_tokens": 2, "output_tokens": 1}
        output = [
            SimpleNamespace(
                type="message",
                id="message-1",
                content=[SimpleNamespace(type="output_text", text="ok")],
            )
        ]

        @staticmethod
        def model_dump():
            return {"id": "response-1"}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def get_final_response():
            return FakeResponse()

    class FakeResponses:
        @staticmethod
        def stream(**payload):
            captured["payload"] = payload
            return FakeStream()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            captured["closed"] = False
            self.responses = FakeResponses()

        @staticmethod
        def close():
            captured["closed"] = True

    monkeypatch.setattr(api_client_module, "OpenAI", FakeOpenAI)
    client = _codex_client(auth_path)

    result = client._openai_query_with_tools(0, [{"role": "user", "content": "Say ok"}])

    assert client.api_key is None
    assert captured["client"]["api_key"] == access_token
    assert captured["client"]["base_url"] == CODEX_RESPONSES_BASE_URL
    assert captured["client"]["default_headers"] == {
        "ChatGPT-Account-ID": "account-test",
        "originator": "codex_cli_rs",
        "x-openai-internal-codex-responses-lite": "true",
    }
    assert captured["payload"]["store"] is False
    assert captured["payload"]["reasoning"] == {"effort": "xhigh", "context": "all_turns"}
    assert captured["payload"]["parallel_tool_calls"] is False
    assert captured["payload"]["tool_choice"] == "auto"
    assert "tools" not in captured["payload"]
    assert captured["payload"]["input"][0] == {
        "type": "additional_tools",
        "role": "developer",
        "tools": [],
    }
    assert captured["closed"] is True
    assert result.conversation[-1]["content"] == "ok\n\n"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_openai_responses_api": False}, "requires use_openai_responses_api"),
        ({"api": "anthropic"}, "only supported when api is 'openai'"),
        ({"batch_processing": True}, "does not support batch_processing"),
        ({"background": True}, "does not support background"),
        ({"store": True}, "requires store: false"),
        ({"base_url": "https://example.test/v1"}, "cannot be combined with base_url"),
    ],
)
def test_codex_auth_rejects_conflicting_configuration(tmp_path, override, message):
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path)

    with pytest.raises(ValueError, match=message):
        _codex_client(auth_path, **override)


def test_codex_auth_rejects_empty_path():
    with pytest.raises(ValueError, match="non-empty path"):
        _codex_client("")


@pytest.mark.parametrize("contents", ["not-json", "{}", '{"auth_mode": "api_key"}'])
def test_codex_auth_rejects_malformed_files(tmp_path, contents):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(contents, encoding="utf-8")

    with pytest.raises(CodexAuthenticationError, match="codex login"):
        _codex_client(auth_path)


def test_codex_auth_rejects_missing_and_expired_files(tmp_path):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(CodexAuthenticationError, match="does not exist"):
        _codex_client(missing_path)

    expired_path = tmp_path / "expired.json"
    _write_auth(expired_path, expires_at=time.time() - 1)
    with pytest.raises(CodexAuthenticationError, match="expired"):
        _codex_client(expired_path)


def test_codex_auth_retries_once_when_external_login_replaces_credentials(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path)
    client = _codex_client(auth_path)
    old_auth = _CodexAuthSnapshot("old", "account-test", False)
    new_auth = _CodexAuthSnapshot("new", "account-test", False)
    unauthorized = AuthenticationError(
        "unauthorized",
        response=httpx.Response(401, request=httpx.Request("POST", CODEX_RESPONSES_BASE_URL)),
        body={},
    )
    load_auth = iter((old_auth, new_auth))
    monkeypatch.setattr(client, "_load_codex_auth", lambda: next(load_auth))
    attempts = []

    def stream_once(auth, _payload):
        attempts.append(auth)
        if len(attempts) == 1:
            raise unauthorized
        return "response"

    monkeypatch.setattr(client, "_stream_codex_response_once", stream_once)

    assert client._stream_codex_response({"model": "gpt-5.6-sol"}) == "response"
    assert attempts == [old_auth, new_auth]


def test_codex_auth_does_not_retry_or_fall_back_when_credentials_are_unchanged(tmp_path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    client = _codex_client(auth_path)
    auth = _CodexAuthSnapshot("unchanged", "account-test", False)
    unauthorized = AuthenticationError(
        "unauthorized",
        response=httpx.Response(401, request=httpx.Request("POST", CODEX_RESPONSES_BASE_URL)),
        body={},
    )
    monkeypatch.setattr(client, "_load_codex_auth", lambda: auth)
    attempts = []

    def reject(current_auth, _payload):
        attempts.append(current_auth)
        raise unauthorized

    monkeypatch.setattr(client, "_stream_codex_response_once", reject)

    with pytest.raises(CodexAuthenticationError, match="backend rejected"):
        client._stream_codex_response({"model": "gpt-5.6-sol"})
    assert attempts == [auth]
    assert client.api_key is None


@pytest.mark.parametrize(
    ("config_name", "reasoning_effort"),
    [
        ("gpt-56-sol-xhigh.yaml", "xhigh"),
        ("gpt-56-sol.yaml", "medium"),
    ],
)
def test_codex_model_config_is_safe_and_bounded(config_name, reasoning_effort):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "models"
        / "openai"
        / config_name
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model"] == f"gpt-5.6-sol--{reasoning_effort}"
    assert config["codex_auth_token"] == "~/.codex/auth.json"
    assert config["use_openai_responses_api"] is True
    assert config["batch_processing"] is False
    assert config["background"] is False
    assert config["store"] is False
    assert config["concurrent_requests"] == 6
    assert config["include"] == ["reasoning.encrypted_content"]
    assert "max_tokens" not in config
