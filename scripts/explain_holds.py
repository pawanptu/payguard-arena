"""AI explanations for the ops console queue: eval/console/explanations.json (docs/CONSOLE_CONTRACT.md, section B).

  python -m scripts.explain_holds --plan                       no calls: cases, uncached cases, provider, model and,
                                                               for Sarvam, the number of paid calls a full run needs
  python -m scripts.explain_holds --run --provider gemini --cap N
                                                               at most N real requests this run (retries included),
                                                               top of the queue first; everything else comes from the
                                                               cache or the deterministic template
  python -m scripts.explain_holds --offline [--provider P]     cache + template only, no network, no key

Needs eval/console/queue.json (python -m scripts.build_console). Replies are cached in data/llm_cache_explain.jsonl
(env PAYGUARD_LLM_CACHE_EXPLAIN), so a rerun makes no new call. The LLM sees decision-time evidence only; a reply
whose numbers are not in the evidence is replaced by the template (source "template").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from payguard.config import optional
from payguard.console.explain import (GROUNDING_TEXT, PROMPT_VERSION, Explainer, default_cache_path,
                                      explain_cases)
from payguard.console.queue import now_iso
from payguard.llm.cache import LLMCache
from payguard.llm.errors import LLMConfigError

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "eval" / "console" / "queue.json"
OUT_PATH = ROOT / "eval" / "console" / "explanations.json"
PROVIDERS = ("gemini", "sarvam")
# Explanation JSON is about 200 tokens; Sarvam's default 300 can cut one off mid-object.
SARVAM_MAX_TOKENS = 700


def provider_model(provider: str) -> str:
    """Model name without building a client (needed for cache keys): env override, else the client default."""
    if provider == "gemini":
        from payguard.llm.gemini import DEFAULT_MODEL
        return optional("GEMINI_MODEL", DEFAULT_MODEL).removeprefix("models/")
    from payguard.llm.client import DEFAULT_MODEL
    return optional("SARVAM_MODEL", DEFAULT_MODEL)


def make_client(provider: str):
    if provider == "gemini":
        from payguard.llm.gemini import GeminiClient
        return GeminiClient()
    from payguard.llm.client import SarvamClient
    return SarvamClient(max_tokens=SARVAM_MAX_TOKENS)


def load_queue(path: Path = QUEUE_PATH) -> dict:
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run python -m scripts.build_console first")
    return json.loads(path.read_text(encoding="utf-8"))


def build_doc(queue: dict, provider: str, model: str, explainer: Explainer) -> dict:
    explanations = explain_cases(queue["cases"], explainer)
    n_llm = sum(1 for e in explanations.values() if e["source"] == "llm")
    return {"schema_version": 1, "provider": provider, "model": model, "prompt_version": PROMPT_VERSION,
            "generated_at": now_iso(), "grounding": GROUNDING_TEXT,
            "counts": {"llm": n_llm, "template": len(explanations) - n_llm}, "explanations": explanations}


def write_doc(doc: dict, out: Path, force: bool) -> bool:
    if out.is_file() and not force:
        try:
            old = json.loads(out.read_text(encoding="utf-8"))["counts"]["llm"]
        except (ValueError, KeyError, TypeError):
            old = 0
        if doc["counts"]["llm"] < old:
            print(f"REFUSED: {out.name} holds {old} LLM explanations, this run only has {doc['counts']['llm']} "
                  "(cache missing?). Pass --force to overwrite.", file=sys.stderr)
            return False
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:  # LF on every OS: same bytes everywhere
        fh.write(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes): {doc['counts']['llm']} llm, {doc['counts']['template']} template")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="count uncached cases, make no call")
    g.add_argument("--run", action="store_true", help="make real calls (needs --provider and --cap)")
    g.add_argument("--offline", action="store_true", help="cache and template only, no network")
    ap.add_argument("--provider", choices=PROVIDERS, default=None)
    ap.add_argument("--cap", type=int, default=None, help="most real requests in this run")
    ap.add_argument("--queue", type=Path, default=QUEUE_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--cache", type=Path, default=None, help="cache file (default data/llm_cache_explain.jsonl)")
    ap.add_argument("--force", action="store_true", help="overwrite explanations.json even with fewer LLM replies")
    args = ap.parse_args(argv)
    provider = args.provider or "gemini"
    if args.run and (args.provider is None or args.cap is None or args.cap < 1):
        ap.error("--run needs --provider gemini|sarvam and --cap N (N >= 1)")

    queue = load_queue(args.queue)
    cases, thresholds = queue["cases"], queue["thresholds"]
    model = provider_model(provider)
    cache_path = args.cache or default_cache_path()
    cache = LLMCache(cache_path)

    if args.plan:
        ex = Explainer(thresholds, model=model, cache=cache)
        uncached = sum(1 for c in cases if not ex.is_cached(c))
        print(f"provider {provider}, model {model}, prompt {PROMPT_VERSION}")
        print(f"cases: {len(cases)}; uncached: {uncached}; cached: {len(cases) - uncached} "
              f"(cache {cache_path}, {len(cache)} entries)")
        if provider == "sarvam":
            print(f"estimated paid Sarvam calls for a full run: {uncached} (one per uncached case); no call was made")
        else:
            print(f"a full Gemini run makes at most {uncached} requests, spaced by the client throttle; "
                  "no call was made")
        return 0

    client = None
    if args.run:
        try:
            client = make_client(provider)
        except LLMConfigError as exc:
            print(f"cannot start: {exc}", file=sys.stderr)
            return 1
        model = client.model
    ex = Explainer(thresholds, model=model, cache=cache, client=client, cap=args.cap or 0,
                   log=lambda s: print("  " + s, flush=True))
    t0 = time.time()
    try:
        doc = build_doc(queue, provider, model, ex)
    finally:
        if client is not None:
            client.close()
    print(f"{provider} / {model}: calls={ex.calls} errors={ex.errors} ungrounded={ex.ungrounded} "
          f"quota_stop={ex.quota_stop} ({time.time() - t0:.0f}s)")
    return 0 if write_doc(doc, args.out, args.force) else 2


if __name__ == "__main__":
    sys.exit(main())
