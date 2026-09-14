#!/usr/bin/env python3
"""
Small browser UI for the face-recognition baseline.

Start from the project root:
  python scripts/web_app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, render_template_string, request
from werkzeug.utils import secure_filename

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from face_demo import analyze_image, build_app, cosine_similarity  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLD = 0.50

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
face_app = None


PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>人脸识别系统</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f3f5f7; color: #20252b; }
    main { max-width: 920px; margin: 0 auto; padding: 36px 18px 60px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    .intro { color: #5c6670; margin: 0 0 26px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    section { background: white; border: 1px solid #dce1e5; border-radius: 8px; padding: 22px; }
    h2 { margin: 0 0 18px; font-size: 20px; }
    label { display: block; margin: 14px 0 7px; font-weight: 600; }
    input[type=text], input[type=number], input[type=file] { width: 100%; box-sizing: border-box; font: inherit; }
    input[type=text], input[type=number] { border: 1px solid #cbd2d8; border-radius: 5px; padding: 10px; }
    input[type=file] { padding: 12px 0; }
    button { border: 0; border-radius: 5px; background: #1769aa; color: white; padding: 11px 16px; font: inherit; font-weight: 600; cursor: pointer; margin-top: 16px; }
    button:hover { background: #125384; }
    .hint { color: #69737d; font-size: 14px; line-height: 1.5; }
    .result { margin-top: 18px; border-radius: 6px; padding: 13px 14px; line-height: 1.65; background: #eef6ff; border: 1px solid #c8e0f7; }
    .result.error { background: #fff4f2; border-color: #f0c8c1; }
    .result strong { font-size: 18px; }
    .footer { color: #69737d; font-size: 13px; margin-top: 22px; }
    @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } main { padding-top: 24px; } }
  </style>
</head>
<body>
<main>
  <h1>人脸识别系统</h1>
  <p class="intro">先登记，再上传照片识别。照片会保存到服务器的项目目录中。</p>
  <div class="grid">
    <section>
      <h2>登记新人员</h2>
      <form method="post" action="/register" enctype="multipart/form-data">
        <label for="name">姓名</label>
        <input id="name" name="name" type="text" placeholder="例如：张三" required>
        <label for="images">照片</label>
        <input id="images" name="images" type="file" accept="image/*" multiple required>
        <p class="hint">可以一次选择多张照片。建议每人 3 到 5 张，角度和光线稍有变化，但每张最好只有一张清楚的人脸。</p>
        <button type="submit">登记</button>
      </form>
      {% if register_result %}<div class="result {% if register_result.error %}error{% endif %}">{{ register_result.html|safe }}</div>{% endif %}
    </section>

    <section>
      <h2>识别照片</h2>
      <form method="post" action="/recognize" enctype="multipart/form-data">
        <label for="image">待识别照片</label>
        <input id="image" name="image" type="file" accept="image/*" required>
        <label for="threshold">判断阈值</label>
        <input id="threshold" name="threshold" type="number" min="0" max="1" step="0.01" value="0.50">
        <p class="hint">当前建议使用 0.50。分数越高越像；低于阈值时会显示“未知人员”。</p>
        <button type="submit">开始识别</button>
      </form>
      {% if recognize_result %}<div class="result {% if recognize_result.error %}error{% endif %}">{{ recognize_result.html|safe }}</div>{% endif %}
    </section>
  </div>
  <p class="footer">模型：InsightFace buffalo_l · 当前识别的是最大的人脸 · 这是学习和内部测试版本</p>
</main>
</body>
</html>
"""


def registry_path() -> Path:
    return PROJECT_DIR / "data" / "registry.json"


def load_registry() -> dict:
    path = registry_path()
    if not path.exists():
        return {"people": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(registry: dict) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def next_person_id(registry: dict) -> str:
    largest = 0
    for person_id in registry.get("people", {}):
        if person_id.startswith("person_"):
            try:
                largest = max(largest, int(person_id.split("_", 1)[1]))
            except ValueError:
                pass
    return f"person_{largest + 1:03d}"


def get_face_app():
    global face_app
    if face_app is None:
        face_app = build_app()
    return face_app


def safe_uploaded_name(filename: str, index: int) -> str:
    cleaned = secure_filename(filename)
    suffix = Path(cleaned).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"第 {index} 张文件不是支持的图片格式")
    stem = Path(cleaned).stem or "photo"
    return f"upload_{index:03d}_{stem}{suffix}"


def register_images(name: str, files) -> dict:
    name = name.strip()
    if not name:
        raise ValueError("请先填写姓名")

    usable_files = [item for item in files if item and item.filename]
    if not usable_files:
        raise ValueError("请至少选择一张照片")

    registry = load_registry()
    person_id = next_person_id(registry)
    person_dir = PROJECT_DIR / "data" / "people" / person_id
    raw_dir = person_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    valid_embeddings = []
    saved_photos = []
    rejected = []

    for index, uploaded in enumerate(usable_files, start=1):
        try:
            filename = safe_uploaded_name(uploaded.filename, index)
            target = raw_dir / filename
            uploaded.save(target)
            result = analyze_image(get_face_app(), str(target))
            if result["face"] is None:
                rejected.append(f"{uploaded.filename}：没有检测到人脸")
                target.unlink(missing_ok=True)
                continue
            if result["face_count"] != 1:
                rejected.append(f"{uploaded.filename}：检测到 {result['face_count']} 张脸，请单独上传一个人的照片")
                target.unlink(missing_ok=True)
                continue
            valid_embeddings.append(result["face"]["embedding"].astype(np.float32))
            saved_photos.append(str(target.relative_to(PROJECT_DIR)))
        except ValueError as exc:
            rejected.append(str(exc))

    if not valid_embeddings:
        shutil.rmtree(person_dir, ignore_errors=True)
        raise ValueError("没有可登记的照片。" + ("；".join(rejected) if rejected else ""))

    embedding_path = person_dir / "embeddings.npy"
    np.save(embedding_path, np.stack(valid_embeddings))
    registry.setdefault("people", {})[person_id] = {
        "name": name,
        "folder": str(person_dir.relative_to(PROJECT_DIR)),
        "photos": saved_photos,
        "embedding_file": str(embedding_path.relative_to(PROJECT_DIR)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "insightface/buffalo_l",
    }
    save_registry(registry)
    return {"person_id": person_id, "saved_count": len(valid_embeddings), "rejected": rejected}


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def recognize_image(uploaded, threshold: float) -> dict:
    if not uploaded or not uploaded.filename:
        raise ValueError("请先选择待识别照片")
    suffix = Path(secure_filename(uploaded.filename)).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError("待识别文件不是支持的图片格式")

    temp_dir = PROJECT_DIR / "data" / ".web_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / f"query{suffix}"
    uploaded.save(target)
    try:
        result = analyze_image(get_face_app(), str(target))
    finally:
        target.unlink(missing_ok=True)

    if result["face"] is None:
        raise ValueError("没有检测到人脸，请上传正脸或稍微转回来的照片")

    registry = load_registry()
    people = registry.get("people", {})
    if not people:
        raise ValueError("系统还没有登记任何人")

    query_embedding = result["face"]["embedding"].astype(np.float32)
    best = None
    for person_id, person in people.items():
        embedding_file = resolve_project_path(person["embedding_file"])
        embeddings = np.load(embedding_file)
        similarity = max(cosine_similarity(query_embedding, item) for item in embeddings)
        candidate = {"person_id": person_id, "name": person.get("name", ""), "similarity": similarity}
        if best is None or candidate["similarity"] > best["similarity"]:
            best = candidate

    assert best is not None
    best["face_count"] = result["face_count"]
    best["det_score"] = result["face"]["det_score"]
    best["threshold"] = threshold
    best["decision"] = "matched" if best["similarity"] >= threshold else "unknown_person"
    return best


def render_page(register_result=None, recognize_result=None):
    return render_template_string(PAGE, register_result=register_result, recognize_result=recognize_result)


@app.get("/")
def index():
    return render_page()


@app.post("/register")
def register_route():
    try:
        outcome = register_images(request.form.get("name", ""), request.files.getlist("images"))
        message = f"<strong>登记成功</strong><br>编号：{outcome['person_id']}<br>姓名：{request.form.get('name', '').strip()}<br>已保存：{outcome['saved_count']} 张照片"
        if outcome["rejected"]:
            message += "<br>跳过：" + "；".join(outcome["rejected"])
        return render_page(register_result={"html": message})
    except Exception as exc:
        return render_page(register_result={"html": f"<strong>登记未完成</strong><br>{exc}", "error": True}), 400


@app.post("/recognize")
def recognize_route():
    try:
        threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
        if not 0 <= threshold <= 1:
            raise ValueError("阈值必须在 0 到 1 之间")
        outcome = recognize_image(request.files.get("image"), threshold)
        if outcome["decision"] == "matched":
            message = f"<strong>识别结果：{outcome['name']}</strong><br>相似度：{outcome['similarity']:.4f}<br>阈值：{threshold:.2f}"
        else:
            message = f"<strong>未知人员</strong><br>最接近：{outcome['name']}<br>相似度：{outcome['similarity']:.4f}，低于阈值 {threshold:.2f}"
        if outcome["face_count"] > 1:
            message += f"<br>提醒：照片中检测到 {outcome['face_count']} 张脸，系统比较了最大的一张"
        return render_page(recognize_result={"html": message})
    except Exception as exc:
        return render_page(recognize_result={"html": f"<strong>识别未完成</strong><br>{exc}", "error": True}), 400


@app.errorhandler(413)
def too_large(_error):
    return render_page(recognize_result={"html": "<strong>图片太大</strong><br>请上传小于 16 MB 的照片", "error": True}), 413


if __name__ == "__main__":
    port = int(os.environ.get("FACE_WEB_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
