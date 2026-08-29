import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    events = []
    for line in f:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue

# find last turn_start index
last_turn = 0
for i, ev in enumerate(events):
    if ev.get("type") == "turn_start":
        last_turn = i

buf = []
for ev in events[last_turn:]:
    if ev.get("type") == "message_update":
        ame = ev.get("assistantMessageEvent") or {}
        if ame.get("type") == "text_delta":
            buf.append(ame.get("delta", ""))

text = "".join(buf)
out = sys.argv[2] if len(sys.argv) > 2 else None
if out:
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {len(text)} chars to {out}")
else:
    print(text[-6000:])
