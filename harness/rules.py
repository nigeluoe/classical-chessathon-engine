"""Event constants. Canonical source: https://aichessathon.com/docs/rules.md and
https://aichessathon.com/docs/agent-contract.md.

Re-checked live 2026-09-10: INIT_BUDGET_S, PLY_CAP, and STDOUT_CAP had all changed since this
file was last written (60.0->90.0, 300->600, 4096->8192) and this file had not been updated to
match, so every local game was being judged against a stricter bar than the real platform
actually enforces. Updated on explicit instruction. STDOUT_CAP's real behaviour is now "first
4 KB plus last 4 KB kept" (a head+tail split); `harness/sandbox.py`'s capture still only keeps
the first STDOUT_CAP bytes (no tail half), so raising the constant to 8192 is a real
improvement (twice the budget) but does not fully replicate the tail-keeping behaviour -- a
known, smaller remaining gap, harmless for anything that matters within the first 8 KB.
Re-verify against both URLs before trusting these numbers again; they already moved once."""

INIT_BUDGET_S = 90.0
BASE_MS = 120_000
INCREMENT_MS = 500
PLY_CAP = 600
STDOUT_CAP = 8192
MAX_UNZIPPED_BYTES = 50_000_000
WATCHDOG_GRACE_MS = 500
