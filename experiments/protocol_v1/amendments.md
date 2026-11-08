# Amendments to preregister.yaml

| # | Date | Before any fit? | Change | Reason | New confirmation needed? |
|---|---|---|---|---|---|
| A0 | 2026-11-08 | yes | Background control mask: central 76% zeroed → only outer 8 px ring kept | 76% left real nodal lines near the plate edge visible; the control must see rim/markers/background only | no |
| A1 | 2026-11-08 | yes | Outer/inner splits: `StratifiedGroupKFold(seed=42)` → fixed round-robin of preparation blocks in capture order (block b → fold b mod K, per session) | With only 3 groups of 2241 Hz the stratifier produced a test fold without that class; protocol p.8 forbids trying seeds. The new rule uses only labels and groups, and passes the same leakage/coverage checks | no |
