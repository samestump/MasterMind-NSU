"""Model failure diagnostics and optional reporting against a token target."""
import asyncio
import hashlib
import time

from harness.files import Journal


def budget_balance(target, spent):
    return {
        "remaining_weighted_tokens": round(max(0.0, target - spent), 3),
        "weighted_tokens_over_budget": round(max(0.0, spent - target), 3),
    }


def _error_attribute(error, name):
    try:
        return getattr(error, name, None)
    except Exception:
        return None


def provider_error_details(error):
    """Allowlisted metadata only: exception text can contain credentials or prompts."""
    chain, seen, names, statuses, codes = [], set(), set(), [], []
    request_id_hash = None
    parsing = False
    usage_error = None
    while isinstance(error, BaseException) and id(error) not in seen and len(chain) < 8:
        seen.add(id(error))
        chain.append(type(error).__name__)
        classes = {cls.__name__ for cls in type(error).__mro__}
        names.update(classes)
        status = _error_attribute(error, "status_code")
        if type(status) is int and 100 <= status <= 599:
            statuses.append(status)
        code = _error_attribute(error, "code")
        if "FrameworkError" in classes and type(code) is int and 0 < code < 10_000_000:
            codes.append(code)
        request_id = _error_attribute(error, "request_id")
        if type(request_id) is str and 0 < len(request_id) <= 512:
            request_id_hash = hashlib.sha256(request_id.encode("utf-8", "replace")).hexdigest()
        if (type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str
                and error.args in (
                ("missing_usage",), ("invalid_usage",), ("inconsistent_reasoning_usage",))):
            usage_error = error.args[0]
        traceback = error.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            module = frame.f_globals.get("__name__")
            parsing |= (frame.f_code.co_name == "_parse_response" and type(module) is str
                        and module.startswith("openjiuwen."))
            traceback = traceback.tb_next
        error = error.__cause__ or error.__context__ or _error_attribute(error, "cause")
    status = statuses[-1] if statuses else None
    if status is not None:
        category = ("authentication_error" if status in (401, 403) else
                    "rate_limit" if status == 429 else
                    "server_error" if status >= 500 else "http_error")
    elif "SSLCertVerificationError" in names:
        category = "tls_error"
    elif "APITimeoutError" in names or "TimeoutException" in names:
        category = "request_timeout"
    elif "TimeoutError" in names:
        category = "deadline_timeout" if "CancelledError" in names else "request_timeout"
    elif names & {"APIConnectionError", "ConnectionError", "NetworkError", "ProtocolError"}:
        category = "connection_error"
    elif "JSONDecodeError" in names or parsing:
        category = "response_format_error"
    elif usage_error:
        category = "invalid_usage"
    else:
        category = "sdk_error" if "FrameworkError" in names else "unknown_error"
    result = {"category": category, "exception_chain": chain, "usage_known": False}
    if status is not None:
        result["http_status"] = status
    if codes:
        result["sdk_codes"] = codes
    if request_id_hash:
        result["request_id_sha256"] = request_id_hash
    if usage_error:
        result["usage_error"] = usage_error
    return result


class DiagnosticBackend:
    def __init__(self, backend, error_path, *, budget_target=None):
        self.backend = backend
        self.journal = Journal(error_path)
        self.calls = 0
        self.budget_target = budget_target
        self.weighted_tokens_spent = 0.0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    async def invoke(self, messages, max_tokens):
        self.calls += 1
        call_index = self.calls
        started = time.monotonic()
        try:
            response = await self.backend.invoke(messages, max_tokens)
        except (Exception, asyncio.CancelledError) as exc:
            try:
                details = provider_error_details(exc)
                if isinstance(exc, asyncio.CancelledError):
                    details['category'] = 'cancelled'
                self.journal.append({'event': 'model_error', 'call_index': call_index,
                                     'model': self.model_name, **details,
                                     'timestamp_unix': time.time(),
                                     'call_elapsed_seconds': round(time.monotonic() - started, 3)})
            except Exception:
                pass
            raise
        if self.budget_target is None:
            return response
        usage = response["usage"]
        self.weighted_tokens_spent += usage["output_tokens"] + usage["input_tokens"] / 10
        budget = {
            "token_budget": self.budget_target,
            "weighted_tokens_spent": round(self.weighted_tokens_spent, 3),
            **budget_balance(self.budget_target, self.weighted_tokens_spent),
        }
        return {**response, "usage": {**usage, "budget": budget}}
