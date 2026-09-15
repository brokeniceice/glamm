#!/usr/bin/env python3
"""Attach the preregistered I2 scratch control to the Phase6E.2 result."""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/phase6e2_c1_specific_r1"
MAIN = BASE / "phase6e2_c1_specific_r1.json"
I2 = BASE / "i2/phase6e2_i2_random_init.json"
DOC = ROOT / "docs/phase6e2_c1_specific_r1.md"

def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n"); os.replace(tmp, path)

def main():
    main = json.loads(MAIN.read_text()); i2 = json.loads(I2.read_text())
    main["supplemental_I2_random_utility_random_rectifier"] = {
        "arm_metrics": i2["arms"]["C1+I2_random_R1"], "paired_statistics": i2["paired_statistics"],
        "decisions": i2["decisions"], "initialization": i2["provenance"]["initialization"],
        "selector": i2["provenance"]["selector"], "result_path": str(I2.resolve())}
    main["status"] = "COMPLETE_WITH_I2"
    write_json(MAIN, main)
    m = i2["arms"]["C1+I2_random_R1"]
    appendix = ("\n## Supplemental I2 — random utility + random rectifier\n\n"
        "I1 was not run. I2 reused the identical frozen C1 G0 query cache and used the same 10-epoch recipe/validation selector.\n\n"
        "| Arm | Mean FG IoU | Mean FG F1 | Global FG IoU | Global FG F1 | SEG trigger |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"| C1+I2 random R1 | {m['mean_foreground_iou']:.6f} | {m['mean_foreground_f1']:.6f} | {m['global_foreground_iou']:.6f} | {m['global_foreground_f1']:.6f} | {m['seg_trigger_rate']:.6f} |\n")
    DOC.write_text(DOC.read_text() + appendix)

if __name__ == "__main__": main()
