#!/usr/bin/env python3
"""Small local web UI for reviewing sampled CLIP content labels.

The server is deliberately dependency-free.  It only edits
``manual_corrections_template.jsonl`` and treats every CLIP prediction file as
read-only provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTENT_LABELS = ("human", "animal", "object", "scene")
MANUAL_LABELS = frozenset((*CONTENT_LABELS, "ambiguous"))
REVIEW_STATUSES = frozenset(("pending", "reviewed", "rejected"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO_ROOT / "outputs/data_audits/content_labels_clip_v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


class ReviewStore:
    def __init__(self, labels_dir: Path):
        self.labels_dir = labels_dir.expanduser().resolve()
        self.candidates_path = self.labels_dir / "review_candidates.jsonl"
        self.corrections_path = self.labels_dir / "manual_corrections_template.jsonl"
        self.candidates = load_jsonl(self.candidates_path)
        self.corrections = load_jsonl(self.corrections_path)
        self.lock = threading.Lock()
        self._validate()

    def _validate(self) -> None:
        if len(self.candidates) != len(self.corrections):
            raise ValueError(
                f"Candidate/correction length mismatch: {len(self.candidates)} != {len(self.corrections)}"
            )
        candidate_ids = [row["sample_id"] for row in self.candidates]
        correction_ids = [row["sample_id"] for row in self.corrections]
        if candidate_ids != correction_ids:
            raise ValueError("Candidate and correction sample IDs are not in the same order")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Duplicate sample IDs in review candidates")
        for index, row in enumerate(self.corrections):
            label = row.get("manual_label")
            status = row.get("reviewer_status", "pending")
            if label is not None and label not in MANUAL_LABELS:
                raise ValueError(f"Invalid manual label at row {index + 1}: {label}")
            if status not in REVIEW_STATUSES:
                raise ValueError(f"Invalid reviewer status at row {index + 1}: {status}")

    def public_state(self) -> dict[str, Any]:
        rows = []
        for candidate, correction in zip(self.candidates, self.corrections):
            rows.append({
                "sample_id": candidate["sample_id"],
                "source": candidate["source"],
                "domain": candidate["domain"],
                "auto_label": correction["auto_label"],
                "manual_label": correction.get("manual_label"),
                "reviewer_status": correction.get("reviewer_status", "pending"),
                "review_reasons": candidate.get("review_reasons", []),
                "margin": candidate["top1_top2_margin"],
                "view_predictions": candidate.get("view_predictions", []),
                "view_agreement": candidate.get("view_agreement"),
                "embedding_index": candidate.get("embedding_index"),
            })
        return {"rows": rows, "content_labels": list(CONTENT_LABELS)}

    def image_path(self, index: int) -> Path:
        if index < 0 or index >= len(self.candidates):
            raise IndexError(index)
        path = Path(self.candidates[index]["image_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def update(self, index: int, manual_label: str | None, reviewer_status: str) -> dict[str, Any]:
        if index < 0 or index >= len(self.corrections):
            raise ValueError("Review index is out of range")
        if manual_label is not None and manual_label not in MANUAL_LABELS:
            raise ValueError(f"Invalid manual label: {manual_label}")
        if reviewer_status not in REVIEW_STATUSES:
            raise ValueError(f"Invalid reviewer status: {reviewer_status}")
        if reviewer_status == "reviewed" and manual_label is None:
            raise ValueError("A reviewed image requires a manual label")
        if reviewer_status == "rejected" and manual_label is not None:
            raise ValueError("A rejected image must have a null manual label")
        if reviewer_status == "pending" and manual_label is not None:
            raise ValueError("A pending image must have a null manual label")
        with self.lock:
            row = self.corrections[index]
            row["manual_label"] = manual_label
            row["reviewer_status"] = reviewer_status
            atomic_write_jsonl(self.corrections_path, self.corrections)
            return dict(row)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLIP 内容标签审核</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #101215; color: #eef1f5; }
header { height: 64px; padding: 12px 22px; display: flex; gap: 22px; align-items: center; background: #191d22; border-bottom: 1px solid #343a43; }
#progress { flex: 1; height: 12px; accent-color: #56b6c2; }
main { height: calc(100vh - 64px); display: grid; grid-template-columns: minmax(0, 1fr) 390px; }
.viewer { min-width: 0; display: grid; place-items: center; padding: 18px; background: #090b0d; }
#image { max-width: 100%; max-height: calc(100vh - 104px); object-fit: contain; box-shadow: 0 2px 24px #000; }
.panel { padding: 22px; overflow-y: auto; border-left: 1px solid #343a43; }
.sample { color: #aeb7c4; overflow-wrap: anywhere; font-size: 13px; }
.auto { margin: 18px 0; padding: 14px; background: #222831; border-radius: 8px; }
.auto strong { text-transform: uppercase; color: #ffd166; font-size: 21px; }
.buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
button { border: 1px solid #4b5563; background: #252b34; color: white; border-radius: 7px; padding: 13px 8px; font-size: 15px; cursor: pointer; }
button:hover { background: #343c48; }
button.selected { outline: 3px solid #56b6c2; background: #24434a; }
button.reject { color: #ff8d8d; }
.wide { grid-column: 1 / -1; }
.nav { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 18px; }
#save-state { min-height: 24px; margin: 12px 0; color: #86efac; }
.keys { margin-top: 20px; color: #aeb7c4; font-size: 13px; line-height: 1.75; }
kbd { background: #343a43; padding: 2px 6px; border-radius: 4px; color: white; }
@media (max-width: 800px) { main { grid-template-columns: 1fr; height: auto; } .viewer { height: 55vh; } .panel { border-left: 0; } }
</style>
</head>
<body>
<header>
  <strong id="position">加载中…</strong>
  <progress id="progress" value="0" max="1"></progress>
  <span id="done"></span>
</header>
<main>
  <section class="viewer"><img id="image" alt="待审核图片"></section>
  <aside class="panel">
    <div class="sample" id="sample"></div>
    <div class="auto">
      自动标签：<strong id="auto"></strong><br>
      margin：<span id="margin"></span><br>
      双视图：<span id="views"></span><br>
      复核原因：<span id="reasons"></span>
    </div>
    <div class="buttons">
      <button data-label="human">1 · Human</button>
      <button data-label="animal">2 · Animal</button>
      <button data-label="object">3 · Object</button>
      <button data-label="scene">4 · Scene</button>
      <button data-label="ambiguous">5 · Ambiguous</button>
      <button id="accept">A · 接受自动标签</button>
      <button id="reject" class="reject wide">X · Reject</button>
    </div>
    <div id="save-state"></div>
    <div class="nav">
      <button id="previous">← 上一张</button>
      <button id="next">下一张 →</button>
      <button id="next-pending" class="wide">跳到下一张未审核</button>
    </div>
    <div class="keys">
      快捷键：<kbd>1</kbd>–<kbd>4</kbd> 四类；<kbd>5</kbd> 模糊；
      <kbd>A</kbd> 接受自动标签；<kbd>X</kbd> 剔除；方向键前后切换。
      选择后会立即保存并跳到下一张未审核图片。
    </div>
  </aside>
</main>
<script>
let rows = [];
let index = 0;
let saving = false;
const $ = id => document.getElementById(id);

function reviewedCount() { return rows.filter(r => r.reviewer_status !== 'pending').length; }
function nextPending(start) {
  for (let offset = 1; offset <= rows.length; offset++) {
    const candidate = (start + offset) % rows.length;
    if (rows[candidate].reviewer_status === 'pending') return candidate;
  }
  return Math.min(start + 1, rows.length - 1);
}
function render() {
  const row = rows[index];
  $('position').textContent = `${index + 1} / ${rows.length}`;
  $('done').textContent = `已完成 ${reviewedCount()} / ${rows.length}`;
  $('progress').max = rows.length;
  $('progress').value = reviewedCount();
  $('image').src = `/api/image/${index}`;
  $('sample').textContent = `${row.sample_id}\n来源：${row.source} · ${row.domain}`;
  $('sample').style.whiteSpace = 'pre-line';
  $('auto').textContent = row.auto_label;
  $('margin').textContent = row.margin.toFixed(5);
  $('views').textContent = row.view_predictions.join(' / ') || '未知';
  $('reasons').textContent = row.review_reasons.join(', ') || '随机质检样本';
  document.querySelectorAll('[data-label]').forEach(button => {
    button.classList.toggle('selected', row.reviewer_status === 'reviewed' && button.dataset.label === row.manual_label);
  });
  $('reject').classList.toggle('selected', row.reviewer_status === 'rejected');
  $('save-state').textContent = row.reviewer_status === 'pending' ? '尚未审核' :
    row.reviewer_status === 'rejected' ? '已剔除' : `已保存：${row.manual_label}`;
}
async function save(manualLabel, status) {
  if (saving) return;
  saving = true;
  $('save-state').textContent = '保存中…';
  try {
    const response = await fetch('/api/review', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({index, manual_label: manualLabel, reviewer_status: status})
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '保存失败');
    rows[index].manual_label = payload.manual_label;
    rows[index].reviewer_status = payload.reviewer_status;
    index = nextPending(index);
    render();
  } catch (error) {
    $('save-state').textContent = `错误：${error.message}`;
  } finally { saving = false; }
}
document.querySelectorAll('[data-label]').forEach(button => button.onclick = () => save(button.dataset.label, 'reviewed'));
$('accept').onclick = () => save(rows[index].auto_label, 'reviewed');
$('reject').onclick = () => save(null, 'rejected');
$('previous').onclick = () => { index = Math.max(0, index - 1); render(); };
$('next').onclick = () => { index = Math.min(rows.length - 1, index + 1); render(); };
$('next-pending').onclick = () => { index = nextPending(index); render(); };
document.addEventListener('keydown', event => {
  if (event.repeat || saving) return;
  const labels = {'1':'human', '2':'animal', '3':'object', '4':'scene', '5':'ambiguous'};
  if (labels[event.key]) save(labels[event.key], 'reviewed');
  else if (event.key.toLowerCase() === 'a') save(rows[index].auto_label, 'reviewed');
  else if (event.key.toLowerCase() === 'x') save(null, 'rejected');
  else if (event.key === 'ArrowLeft') { index = Math.max(0, index - 1); render(); }
  else if (event.key === 'ArrowRight') { index = Math.min(rows.length - 1, index + 1); render(); }
});
fetch('/api/state').then(response => response.json()).then(payload => {
  rows = payload.rows;
  const pending = rows.findIndex(row => row.reviewer_status === 'pending');
  index = pending < 0 ? 0 : pending;
  render();
}).catch(error => { document.body.textContent = `加载失败：${error}`; });
</script>
</body>
</html>
"""


def make_handler(store: ReviewStore) -> type[BaseHTTPRequestHandler]:
    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format_string: str, *args: Any) -> None:
            print(f"{self.client_address[0]} - {format_string % args}", flush=True)

        def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self.send_json(store.public_state())
                return
            if path.startswith("/api/image/"):
                try:
                    index = int(path.rsplit("/", 1)[-1])
                    image_path = store.image_path(index)
                    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
                    self.send_bytes(image_path.read_bytes(), mime_type)
                except (ValueError, IndexError, FileNotFoundError) as error:
                    self.send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/review":
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 4096:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                updated = store.update(
                    int(payload["index"]), payload.get("manual_label"), str(payload["reviewer_status"])
                )
                self.send_json(updated)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return ReviewHandler


def main() -> None:
    args = parse_args()
    store = ReviewStore(args.labels_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}"
    completed = sum(row.get("reviewer_status") != "pending" for row in store.corrections)
    print(f"Loaded {len(store.candidates)} review samples; completed {completed}", flush=True)
    print(f"Open {url} in a browser; press Ctrl+C here to stop", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
