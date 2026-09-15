# Extra OOD diagnostic: DFJ-Detect and FakeBench

Frozen C1/RINE replay only. Internal-TRAIN normalization and alpha=0.3 are unchanged; no OOD selection or tuning was performed.

| Dataset | Arm | N | Accuracy | ROC-AUC | Fake Recall | TNR | FPR | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dfj_detect | C1-center | 2000 | 0.638000 | 0.705207 | 0.717000 | 0.559000 | 0.441000 | 0.664504 |
| dfj_detect | D1-alpha | 2000 | 0.642500 | 0.708721 | 0.707000 | 0.578000 | 0.422000 | 0.664162 |
| fakebench | C1-center | 5999 | 0.928155 | 0.983064 | 0.858953 | 0.997333 | 0.002667 | 0.922801 |
| fakebench | D1-alpha | 5999 | 0.925988 | 0.995532 | 0.854618 | 0.997333 | 0.002667 | 0.920287 |
