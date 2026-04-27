#!/usr/bin/env python3
"""
eval_http_stability.py — live reproduction and verification of the
"server stops accepting after first request / model-switch hang" bug,
and stress of the single-model batching path.

All suites run against a locally running Osaurus server. No dependencies
outside the Python 3 standard library.

Usage:
    python3 scripts/eval_http_stability.py                     # run all suites
    python3 scripts/eval_http_stability.py --only S4           # just the repro
    python3 scripts/eval_http_stability.py --verbose
    python3 scripts/eval_http_stability.py --port 1337 \
        --model-a foundation --model-b qwen3.6-35b-a3b-mxfp4

Exit code is 0 only if every requested suite passes.

Design notes:
- We explicitly do NOT use `requests` / `httpx`. S4 needs to disconnect
  mid-SSE-stream at the TCP level and observe whether the server
  promptly releases its model lease; nothing short of raw sockets gives
  us that control.
- Every request sends `X-Persist: false` so re-running the script does
  not write test prompts into your chat history DB.
- Timeouts are per-request. A hang in the fix's failure mode manifests
  as a socket read timeout on suite S4 — that IS the signal.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import socket
import ssl  # noqa: F401 — reserved for future --tls support; keeps the imports list stable
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    host: str
    port: int
    model_a: str
    model_b: str
    concurrency: int
    timeout: float
    verbose: bool
    only: Optional[str]
    max_tokens: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def parse_args() -> Config:
    env_port = os.environ.get("OSAURUS_PORT")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_port) if env_port and env_port.isdigit() else 4242,
        help="Default 4242 (or $OSAURUS_PORT).",
    )
    parser.add_argument("--model-a", default="foundation")
    parser.add_argument("--model-b", default="qwen3.6-35b-a3b-mxfp4")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request wall-clock cap, seconds. Suite S4 uses this "
        "as the hang detector — if B takes this long, the fix isn't in.",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="Run a single suite id (e.g. S4). Default: run all.",
    )
    args = parser.parse_args()
    return Config(
        host=args.host,
        port=args.port,
        model_a=args.model_a,
        model_b=args.model_b,
        concurrency=args.concurrency,
        timeout=args.timeout,
        verbose=args.verbose,
        only=args.only,
        max_tokens=args.max_tokens,
    )


# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------


COLOR_PASS = "\033[32m"
COLOR_FAIL = "\033[31m"
COLOR_WARN = "\033[33m"
COLOR_DIM = "\033[2m"
COLOR_RESET = "\033[0m"


def use_color() -> bool:
    return sys.stdout.isatty()


def paint(text: str, code: str) -> str:
    return f"{code}{text}{COLOR_RESET}" if use_color() else text


@dataclass
class SuiteResult:
    suite_id: str
    name: str
    passed: bool
    detail: str
    elapsed: float
    notes: List[str] = field(default_factory=list)


def print_suite_result(r: SuiteResult) -> None:
    tag = paint("PASS", COLOR_PASS) if r.passed else paint("FAIL", COLOR_FAIL)
    print(f"[{r.suite_id}] {r.name:<44s} {tag}  ({r.elapsed:.2f}s) {r.detail}")
    for note in r.notes:
        print(f"      {paint(note, COLOR_DIM)}")


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib)
# ---------------------------------------------------------------------------


def _default_headers() -> List[Tuple[str, str]]:
    # X-Persist: false — keep the test prompts out of chat history.
    return [("X-Persist", "false"), ("User-Agent", "osaurus-stability/1.0")]


def http_get_json(cfg: Config, path: str) -> Tuple[int, dict]:
    """GET path, decode JSON body. Raises on non-200 or decode errors."""
    req = urllib.request.Request(cfg.base_url + path, method="GET")
    for k, v in _default_headers():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
        data = resp.read()
        return resp.status, json.loads(data.decode("utf-8"))


def chat_completions_payload(
    cfg: Config,
    model: str,
    prompt: str,
    *,
    stream: bool,
) -> bytes:
    body = {
        "model": model,
        "stream": stream,
        "max_tokens": cfg.max_tokens,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a terse assistant. Reply in one short sentence.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    return json.dumps(body).encode("utf-8")


def post_chat_json(
    cfg: Config,
    model: str,
    prompt: str,
    *,
    timeout: Optional[float] = None,
) -> Tuple[int, dict, float]:
    """Blocking non-streaming chat/completions. Returns (status, body, secs)."""
    url = f"{cfg.base_url}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=chat_completions_payload(cfg, model, prompt, stream=False),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    for k, v in _default_headers():
        req.add_header(k, v)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout or cfg.timeout) as resp:
        raw = resp.read()
        dt = time.monotonic() - t0
        return resp.status, json.loads(raw.decode("utf-8")), dt


def iter_sse_stream(
    cfg: Config,
    model: str,
    prompt: str,
    *,
    timeout: Optional[float] = None,
) -> Iterable[str]:
    """
    Yield SSE `data: ...` payloads (strings; not yet JSON-parsed) from
    /v1/chat/completions in streaming mode. Generator closes when the
    server writes `[DONE]` or the TCP stream ends.
    """
    url = f"{cfg.base_url}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=chat_completions_payload(cfg, model, prompt, stream=True),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    for k, v in _default_headers():
        req.add_header(k, v)
    resp = urllib.request.urlopen(req, timeout=timeout or cfg.timeout)
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("data: "):
                payload = line[len("data: ") :]
                if payload.strip() == "[DONE]":
                    return
                yield payload
    finally:
        resp.close()


# ---------------------------------------------------------------------------
# S4 requires a raw socket so we can disconnect mid-stream cleanly.
# ---------------------------------------------------------------------------


class RawSSEStream:
    """
    Raw HTTP/1.1 SSE client. Lets us:
      - start a streaming chat/completions request;
      - read the first N SSE data events to prove the stream is alive;
      - `shutdown` + `close` the socket to simulate a client disconnect
        mid-stream without waiting for EOS.

    This is intentionally spartan — we do not support chunked transfer
    decoding for the response because Osaurus's SSE writer does not use
    Transfer-Encoding: chunked. If it ever does, add a minimal chunked
    decoder here.
    """

    def __init__(self, cfg: Config, model: str, prompt: str, connect_timeout: float = 5.0):
        self.cfg = cfg
        self._sock = socket.create_connection(
            (cfg.host, cfg.port), timeout=connect_timeout
        )
        self._sock.settimeout(cfg.timeout)
        self._buf = b""
        self._sent_closed = False

        body = chat_completions_payload(cfg, model, prompt, stream=True)
        headers = [
            f"POST /v1/chat/completions HTTP/1.1",
            f"Host: {cfg.host}:{cfg.port}",
            "Accept: text/event-stream",
            "Content-Type: application/json",
            f"Content-Length: {len(body)}",
            "Connection: close",
        ]
        for k, v in _default_headers():
            headers.append(f"{k}: {v}")
        request_bytes = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + body
        self._sock.sendall(request_bytes)

        # Drain and discard response headers so the caller starts on body bytes.
        header_end = b"\r\n\r\n"
        while header_end not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed before sending response headers")
            self._buf += chunk
        head, _, rest = self._buf.partition(header_end)
        self._buf = rest
        status_line = head.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
        parts = status_line.split(" ", 2)
        self.status = int(parts[1]) if len(parts) >= 2 else 0
        if self.status != 200:
            raise ConnectionError(f"Non-200 status from server: {status_line!r}")

    def next_events(self, n: int) -> List[str]:
        """Block until we've received `n` `data:` frames (or the stream ends)."""
        out: List[str] = []
        while len(out) < n:
            # Pull complete SSE frames out of the buffer.
            while b"\n\n" in self._buf and len(out) < n:
                frame, _, rest = self._buf.partition(b"\n\n")
                self._buf = rest
                for line in frame.split(b"\n"):
                    if line.startswith(b"data: "):
                        payload = line[len(b"data: ") :].decode("utf-8", errors="replace")
                        if payload.strip() == "[DONE]":
                            return out
                        out.append(payload)
                        if len(out) >= n:
                            return out
            if len(out) >= n:
                break
            chunk = self._sock.recv(4096)
            if not chunk:
                break
            self._buf += chunk
        return out

    def abort(self) -> None:
        """Slam the socket shut. Server-side channelInactive should fire."""
        if self._sent_closed:
            return
        self._sent_closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


PROMPT = "Name one fruit."


def suite_S1_sanity(cfg: Config) -> SuiteResult:
    t0 = time.monotonic()
    notes: List[str] = []
    try:
        _, health = http_get_json(cfg, "/health")
        _, models = http_get_json(cfg, "/models")
    except Exception as exc:
        return SuiteResult(
            "S1",
            "sanity (/health + /models)",
            False,
            f"server unreachable: {exc!r}",
            time.monotonic() - t0,
        )

    ids = [m.get("id", "") for m in models.get("data", [])]
    notes.append(f"health keys: {sorted(health.keys())}")
    notes.append(f"{len(ids)} models registered")

    missing = [m for m in (cfg.model_a, cfg.model_b) if m not in ids]
    if missing:
        return SuiteResult(
            "S1",
            "sanity (/health + /models)",
            False,
            f"model(s) not installed: {missing}",
            time.monotonic() - t0,
            notes + [f"available: {ids[:10]}{'...' if len(ids) > 10 else ''}"],
        )
    return SuiteResult(
        "S1",
        "sanity (/health + /models)",
        True,
        f"both models present ({cfg.model_a}, {cfg.model_b})",
        time.monotonic() - t0,
        notes,
    )


def suite_S2_sequential_switch_nonstream(cfg: Config) -> SuiteResult:
    t0 = time.monotonic()
    try:
        status_a, body_a, dt_a = post_chat_json(cfg, cfg.model_a, PROMPT)
        status_b, body_b, dt_b = post_chat_json(cfg, cfg.model_b, PROMPT)
    except urllib.error.URLError as exc:
        return SuiteResult(
            "S2",
            "sequential switch, non-stream",
            False,
            f"request error: {exc}",
            time.monotonic() - t0,
        )
    except Exception as exc:
        return SuiteResult(
            "S2",
            "sequential switch, non-stream",
            False,
            f"unexpected error: {exc!r}",
            time.monotonic() - t0,
        )

    ok = status_a == 200 and status_b == 200 and body_a.get("choices") and body_b.get("choices")
    detail = f"A {dt_a:.2f}s, B {dt_b:.2f}s"
    notes = []
    if cfg.verbose:
        notes.append(f"A content: {body_a['choices'][0]['message'].get('content','')!r}")
        notes.append(f"B content: {body_b['choices'][0]['message'].get('content','')!r}")
    return SuiteResult(
        "S2", "sequential switch, non-stream", bool(ok), detail, time.monotonic() - t0, notes
    )


def _drain_sse(cfg: Config, model: str) -> Tuple[int, float]:
    """Consume a full SSE stream; return (event_count, seconds)."""
    t0 = time.monotonic()
    n = 0
    for _event in iter_sse_stream(cfg, model, PROMPT):
        n += 1
    return n, time.monotonic() - t0


def suite_S3_sequential_switch_sse(cfg: Config) -> SuiteResult:
    t0 = time.monotonic()
    try:
        n_a, dt_a = _drain_sse(cfg, cfg.model_a)
        n_b, dt_b = _drain_sse(cfg, cfg.model_b)
    except Exception as exc:
        return SuiteResult(
            "S3",
            "sequential switch, SSE",
            False,
            f"error: {exc!r}",
            time.monotonic() - t0,
        )
    ok = n_a > 0 and n_b > 0
    return SuiteResult(
        "S3",
        "sequential switch, SSE",
        ok,
        f"A {n_a} events / {dt_a:.2f}s, B {n_b} events / {dt_b:.2f}s",
        time.monotonic() - t0,
    )


def suite_S4_disconnect_then_switch(cfg: Config) -> SuiteResult:
    """
    The direct reproducer for the reported bug.

    1. Start streaming against model A.
    2. Read a couple of events (proves the model is generating).
    3. Slam the socket shut.
    4. Immediately POST a non-streaming request to model B.

    Pre-fix: step 4 hangs past --timeout because `ModelRuntime.unload`
    is waiting on a `ModelLease` that the abandoned producer task
    never releases.

    Post-fix: step 4 returns promptly (sub-second for Foundation +
    bounded by model-load for MLX).
    """
    suite_t0 = time.monotonic()
    notes: List[str] = []

    # Step 1-3 — raw socket dance against model A.
    try:
        stream = RawSSEStream(cfg, cfg.model_a, PROMPT)
    except Exception as exc:
        return SuiteResult(
            "S4",
            "mid-stream disconnect → switch",
            False,
            f"could not start A stream: {exc!r}",
            time.monotonic() - suite_t0,
        )

    a_events = stream.next_events(2)
    notes.append(f"A stream produced {len(a_events)} events before disconnect")
    if cfg.verbose and a_events:
        notes.append(f"first event: {a_events[0][:120]!r}")
    stream.abort()
    disconnect_mark = time.monotonic()
    a_cancel_at = disconnect_mark - suite_t0

    # Step 4 — model B request MUST complete within --timeout.
    try:
        status_b, body_b, dt_b = post_chat_json(
            cfg, cfg.model_b, PROMPT, timeout=cfg.timeout
        )
    except socket.timeout:
        return SuiteResult(
            "S4",
            "mid-stream disconnect → switch",
            False,
            f"model B timed out after {cfg.timeout:.1f}s — classic lease-stuck "
            f"signature (fix not deployed?)",
            time.monotonic() - suite_t0,
            notes,
        )
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, socket.timeout):
            return SuiteResult(
                "S4",
                "mid-stream disconnect → switch",
                False,
                f"model B timed out after {cfg.timeout:.1f}s — classic lease-stuck "
                f"signature (fix not deployed?)",
                time.monotonic() - suite_t0,
                notes,
            )
        return SuiteResult(
            "S4",
            "mid-stream disconnect → switch",
            False,
            f"model B raised URLError: {exc}",
            time.monotonic() - suite_t0,
            notes,
        )
    except Exception as exc:
        return SuiteResult(
            "S4",
            "mid-stream disconnect → switch",
            False,
            f"model B raised {exc!r}",
            time.monotonic() - suite_t0,
            notes,
        )

    ok = status_b == 200 and bool(body_b.get("choices"))
    return SuiteResult(
        "S4",
        "mid-stream disconnect → switch",
        ok,
        f"A cancelled at +{a_cancel_at:.2f}s, B {dt_b:.2f}s",
        time.monotonic() - suite_t0,
        notes,
    )


def _single_nonstream(cfg: Config, idx: int) -> Tuple[bool, float, str]:
    try:
        status, body, dt = post_chat_json(
            cfg, cfg.model_a, f"{PROMPT} (request {idx})"
        )
    except Exception as exc:
        return False, 0.0, f"exc: {exc!r}"
    content = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return (status == 200 and bool(content.strip())), dt, content[:64]


def _single_stream(cfg: Config, idx: int) -> Tuple[bool, float, int]:
    t0 = time.monotonic()
    try:
        events = 0
        for _e in iter_sse_stream(cfg, cfg.model_a, f"{PROMPT} (stream {idx})"):
            events += 1
        return events > 0, time.monotonic() - t0, events
    except Exception:
        return False, time.monotonic() - t0, 0


def suite_S5_single_model_batching(cfg: Config) -> SuiteResult:
    t0 = time.monotonic()
    notes: List[str] = []

    # Warmup + single-call latency reference.
    ok, single_dt, _ = _single_nonstream(cfg, 0)
    if not ok:
        return SuiteResult(
            "S5",
            "single-model batching",
            False,
            "warmup call failed; cannot benchmark concurrency",
            time.monotonic() - t0,
        )
    notes.append(f"warmup single-call latency: {single_dt:.2f}s")

    n = cfg.concurrency
    with cf.ThreadPoolExecutor(max_workers=n) as pool:
        t_par = time.monotonic()
        results = list(pool.map(lambda i: _single_nonstream(cfg, i + 1), range(n)))
        par_dt = time.monotonic() - t_par
    passed_each = [r[0] for r in results]
    latencies = [r[1] for r in results]
    notes.append(
        f"non-stream: {sum(passed_each)}/{n} ok; per-request "
        f"min={min(latencies):.2f}s max={max(latencies):.2f}s; "
        f"wallclock {par_dt:.2f}s vs sequential ~{n*single_dt:.2f}s"
    )

    with cf.ThreadPoolExecutor(max_workers=n) as pool:
        stream_results = list(pool.map(lambda i: _single_stream(cfg, i + 1), range(n)))
    stream_ok = [r[0] for r in stream_results]
    events_each = [r[2] for r in stream_results]
    notes.append(f"stream: {sum(stream_ok)}/{n} ok; events per request={events_each}")

    # Passing = every sub-request succeeded. We do NOT fail purely on the
    # batching speedup claim because a 3.5B model on an idle laptop is
    # already quick enough that the comparison gets noisy.
    passed = all(passed_each) and all(stream_ok)
    return SuiteResult(
        "S5",
        "single-model batching",
        passed,
        f"{n}× non-stream + {n}× stream concurrent",
        time.monotonic() - t0,
        notes,
    )


def _stream_cancel_after_first(cfg: Config) -> Tuple[bool, str]:
    """Start a stream and abort right after the first event."""
    try:
        stream = RawSSEStream(cfg, cfg.model_a, "Say hi briefly.")
    except Exception as exc:
        return False, f"connect: {exc!r}"
    stream.next_events(1)
    stream.abort()
    return True, "cancelled after first event"


def _stream_to_completion(cfg: Config, idx: int) -> Tuple[bool, int, str]:
    try:
        n = 0
        for _e in iter_sse_stream(cfg, cfg.model_a, f"Short fact #{idx}"):
            n += 1
        return n > 0, n, ""
    except Exception as exc:
        return False, 0, repr(exc)


def suite_S6_batching_with_cancels(cfg: Config) -> SuiteResult:
    t0 = time.monotonic()
    n = cfg.concurrency
    cancel_count = max(1, n // 2)
    survive_count = n - cancel_count
    if survive_count < 1:
        survive_count = 1
        cancel_count = n - 1

    # Run cancels + survivors in parallel so they genuinely overlap.
    survivors: List[Tuple[bool, int, str]] = []
    cancel_ok: List[Tuple[bool, str]] = []

    def do_survivor(i: int) -> None:
        survivors.append(_stream_to_completion(cfg, i + 1))

    def do_cancel() -> None:
        cancel_ok.append(_stream_cancel_after_first(cfg))

    threads: List[threading.Thread] = []
    for i in range(survive_count):
        threads.append(threading.Thread(target=do_survivor, args=(i,)))
    for _ in range(cancel_count):
        threads.append(threading.Thread(target=do_cancel))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=cfg.timeout)

    all_cancels_dispatched = len(cancel_ok) == cancel_count and all(r[0] for r in cancel_ok)
    all_survivors_completed = len(survivors) == survive_count and all(r[0] for r in survivors)

    notes = [
        f"{cancel_count} streams cancelled mid-stream, "
        f"{survive_count} streams expected to complete",
        f"cancel path: {sum(r[0] for r in cancel_ok)}/{cancel_count} clean",
        f"survivors: {sum(r[0] for r in survivors)}/{survive_count} reached EOS",
    ]
    for ok, n_events, err in survivors:
        if not ok:
            notes.append(f"  survivor failed: {err}")
    return SuiteResult(
        "S6",
        "batching + mid-stream cancel",
        all_cancels_dispatched and all_survivors_completed,
        f"{cancel_count} cancels + {survive_count} survivors",
        time.monotonic() - t0,
        notes,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


ALL_SUITES: List[Tuple[str, Callable[[Config], SuiteResult]]] = [
    ("S1", suite_S1_sanity),
    ("S2", suite_S2_sequential_switch_nonstream),
    ("S3", suite_S3_sequential_switch_sse),
    ("S4", suite_S4_disconnect_then_switch),
    ("S5", suite_S5_single_model_batching),
    ("S6", suite_S6_batching_with_cancels),
]


def main() -> int:
    cfg = parse_args()
    print(
        paint(
            f"osaurus stability eval → {cfg.base_url}  "
            f"A={cfg.model_a}  B={cfg.model_b}  "
            f"concurrency={cfg.concurrency}  timeout={cfg.timeout:.0f}s",
            COLOR_DIM,
        )
    )
    print()

    suites = ALL_SUITES
    if cfg.only:
        suites = [(sid, fn) for sid, fn in ALL_SUITES if sid.lower() == cfg.only.lower()]
        if not suites:
            print(f"unknown suite id: {cfg.only}", file=sys.stderr)
            return 2

    # S1 is a precondition — if it fails, abort so we don't waste time on
    # a server that doesn't even have the requested models.
    results: List[SuiteResult] = []
    for sid, fn in suites:
        try:
            r = fn(cfg)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            r = SuiteResult(sid, fn.__name__, False, f"crashed: {exc!r}", 0.0)
        results.append(r)
        print_suite_result(r)
        if sid == "S1" and not r.passed:
            print(paint("S1 failed — aborting remaining suites.", COLOR_FAIL))
            break

    print()
    total_pass = sum(1 for r in results if r.passed)
    total = len(results)
    summary = f"{total_pass}/{total} suites passed"
    if total_pass == total and total > 0:
        print(paint(summary, COLOR_PASS))
        return 0
    print(paint(summary, COLOR_FAIL))
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
