#!/usr/bin/env python3
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; out=ROOT/"outputs/phase4c_c_evidence_attribution"
arms={a:json.loads((out/f"baseline_reproduction_{a}.json").read_text()) for a in ("clip_reader","forensic_reader")}
status="PASS" if all(x["status"]=="PASS" for x in arms.values()) else "FAIL"
manifest=json.loads((out/"preflight_manifest.json").read_text()); manifest.update({"status":status,"baseline_reproduction":arms}); (out/"preflight_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
text=(out/"phase4c_c_preflight.md").read_text().replace("Status: **BASELINE_REPRODUCTION_PENDING**",f"Status: **{status}**")
text += "\n## Baseline reproduction\n\n"
for arm,value in arms.items():
    text += f"- {arm}: " + ", ".join(f"{q} error={r.get('absolute_error',0):.3g}" for q,r in value["queries"].items() if q!="phrase_repair") + ".\n"
(out/"phase4c_c_preflight.md").write_text(text)
print(json.dumps({"status":status},indent=2))
if status!="PASS":raise RuntimeError("baseline reproduction failed")
