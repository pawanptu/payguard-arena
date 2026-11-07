# PayGuard Arena

A co-evolving fraud arms race on instant gig payouts. A swarm of defender agents decides whether to **release, hold or step-up** each payout through **PayPal sandbox Payouts**, while a population of attackers mutates its tactics. The defenders self-evolve through outcome-driven reputation updates and a champion-challenger gate.

> **Status: work in progress** (built for the PayPal AI Hackathon, Nov 2026). Nothing below claims results yet. Every number shown by this project is from a **simulated replay**, never real-world fraud data, and only sampled payouts go through the real PayPal sandbox.

## What is in the repo today

| Path | Purpose |
|---|---|
| `payguard/paypal_client.py` | Sandbox-only PayPal client: OAuth, create payout batch, get batch |
| `payguard/config.py` | Reads secrets from a local `.env` (never committed) |
| `scripts/smoke_paypal_payouts.py` | Day-0 gate: one real sandbox payout batch |
| `scripts/smoke_sarvam.py` | Day-0 gate: schema-valid JSON from the LLM defender |
| `tests/` | Offline tests (no network, no keys) |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest                           # offline, needs no keys
cp .env.example .env             # then fill in your own sandbox credentials
python -m scripts.smoke_paypal_payouts
python -m scripts.smoke_sarvam
```

## How PayPal and AI are used

To be completed as features land. Planned:

- **PayPal:** Payouts API (the action the swarm takes), batch polling, signed webhooks feeding the learning loop. Sandbox only.
- **AI:** one LLM defender (Sarvam, structured JSON output, used only on borderline cases and cached) plus the evolution loop that tunes the deterministic defenders.

## License

MIT, see [LICENSE](LICENSE).
