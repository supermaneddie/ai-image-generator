import base64
import gc
import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
from diffusers import DiffusionPipeline
from flask import Flask, jsonify, render_template, request, send_file, url_for
from huggingface_hub.utils import enable_progress_bars
from PIL import Image, UnidentifiedImageError

from config import (
    ASPECT_RATIOS,
    AVAILABLE_MODELS,
    AVAILABLE_STYLES,
    DEFAULT_MODEL,
    GPU_MEMORY_FRACTION,
    MODEL_IDLE_TIMEOUT,
    MODEL_TYPES,
    PROMPT_ENHANCEMENT_LLM,
    STYLE_PROMPTS,
)

REQUEST_ID = re.compile(r"^request_\d{8}T\d{6}_?[0-9a-f]{8}$")
REQUEST_ASSET = re.compile(r"^(?:image|reference)\d+\.png$")
DEFAULT_IMAGE_COUNT = 2
MAX_IMAGE_COUNT = 4
MAX_PROMPT_LENGTH = 10000
MAX_ARCHIVE_FILES = int(os.getenv("MAX_ARCHIVE_FILES", "100"))
MAX_ARCHIVE_BYTES = int(os.getenv("MAX_ARCHIVE_SIZE_MB", "500")) * 1024 * 1024
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_MB", "100")) * 1024 * 1024
app.logger.setLevel(LOG_LEVEL)
print(
    "ai-image-generator configuration: "
    f"MODEL_IDLE_TIMEOUT={MODEL_IDLE_TIMEOUT}, "
    f"GPU_MEMORY_FRACTION={GPU_MEMORY_FRACTION}, "
    f"LOG_LEVEL={logging.getLevelName(app.logger.level)}",
    flush=True,
)

_pipeline = None
_pipeline_model = None
_pipeline_lock = threading.Lock()
_idle_timer = None
_idle_generation = 0
_cuda_memory_limit_configured = False


def _load_model_pipeline(model_id: str):
    """Load a model while keeping long Hub downloads visible in container logs."""
    app.logger.info("Resolving/downloading and loading model: %s", model_id)
    enable_progress_bars()
    finished = threading.Event()
    started = time.monotonic()

    def log_download_heartbeat() -> None:
        while not finished.wait(15):
            app.logger.info(
                "Model download/load still in progress: %s (elapsed %.0f seconds)",
                model_id,
                time.monotonic() - started,
            )

    heartbeat = threading.Thread(target=log_download_heartbeat, daemon=True)
    heartbeat.start()
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
    finally:
        finished.set()
        heartbeat.join(timeout=1)

    app.logger.info(
        "Model pipeline ready: %s (%.1f seconds)",
        model_id,
        time.monotonic() - started,
    )
    return pipeline


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError(f"Unsafe archive path: {member_name}")
    return destination


def _extract_archive(source: Path, output: Path) -> list[Path] | None:
    """Extract a ZIP or TAR archive, or return None when source is not an archive."""
    extracted = []
    total_size = 0

    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) > MAX_ARCHIVE_FILES:
                raise ValueError("Archive contains too many files")
            for member in members:
                total_size += member.file_size
                if total_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive expands beyond the configured size limit")
                destination = _safe_destination(output, member.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_file, destination.open("wb") as target:
                    while chunk := source_file.read(1024 * 1024):
                        target.write(chunk)
                extracted.append(destination)
        return extracted

    if tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            if len(members) > MAX_ARCHIVE_FILES:
                raise ValueError("Archive contains too many files")
            for member in members:
                total_size += member.size
                if total_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive expands beyond the configured size limit")
                destination = _safe_destination(output, member.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_file = archive.extractfile(member)
                if source_file is None:
                    continue
                with source_file, destination.open("wb") as target:
                    while chunk := source_file.read(1024 * 1024):
                        target.write(chunk)
                extracted.append(destination)
        return extracted

    return None


def _open_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _collect_images(uploaded_files, root: Path) -> list[Image.Image]:
    images = []
    for index, upload in enumerate(uploaded_files):
        if not upload.filename:
            continue
        source = root / f"upload-{index}"
        upload.save(source)

        archive_dir = root / f"archive-{index}"
        archive_dir.mkdir()
        extracted = _extract_archive(source, archive_dir)
        candidates = extracted if extracted is not None else [source]
        for candidate in candidates:
            image = _open_image(candidate)
            if image is not None:
                images.append(image)
    return images


def _unload_pipeline_locked() -> None:
    global _pipeline, _pipeline_model
    if _pipeline is None:
        return

    _pipeline = None
    _pipeline_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    app.logger.info("Unloaded idle model pipeline and released CUDA cache")


def _cancel_idle_unload_locked() -> None:
    global _idle_timer, _idle_generation
    _idle_generation += 1
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _idle_unload(timer_generation: int) -> None:
    global _idle_timer
    with _pipeline_lock:
        if timer_generation != _idle_generation:
            return
        _idle_timer = None
        _unload_pipeline_locked()


def _schedule_idle_unload_locked() -> None:
    global _idle_timer, _idle_generation
    if MODEL_IDLE_TIMEOUT <= 0 or _pipeline is None:
        return

    _idle_generation += 1
    timer_generation = _idle_generation
    _idle_timer = threading.Timer(
        MODEL_IDLE_TIMEOUT,
        _idle_unload,
        args=(timer_generation,),
    )
    _idle_timer.daemon = True
    _idle_timer.start()


def _configure_cuda_memory_limit_locked() -> None:
    global _cuda_memory_limit_configured
    if _cuda_memory_limit_configured:
        return
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    _cuda_memory_limit_configured = True
    app.logger.info(
        "Configured CUDA allocator limit to %.1f%% per process",
        GPU_MEMORY_FRACTION * 100,
    )


def _get_pipeline(model_id: str):
    global _pipeline, _pipeline_model
    if _pipeline is not None and _pipeline_model == model_id:
        return _pipeline

    if _pipeline is not None:
        _unload_pipeline_locked()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this container")

    _configure_cuda_memory_limit_locked()
    _pipeline = _load_model_pipeline(model_id)
    _pipeline_model = model_id
    return _pipeline


def _output_dir() -> Path:
    return Path(os.getenv("OUTPUT_DIR", "/var/ai-image-generator")).resolve()


def _request_dir(request_id: str) -> Path | None:
    if not REQUEST_ID.fullmatch(request_id):
        return None
    path = _output_dir() / request_id
    if path.is_symlink() or not path.is_dir() or path.resolve().parent != _output_dir():
        return None
    return path


def _request_asset(request_id: str, filename: str) -> Path | None:
    directory = _request_dir(request_id)
    if directory is None or not REQUEST_ASSET.fullmatch(filename):
        return None
    path = directory / filename
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        return None
    return path


def _request_record(directory: Path) -> dict | None:
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    request_id = directory.name
    generated = sorted(
        (path for path in directory.glob("image*.png") if re.fullmatch(r"image\d+\.png", path.name)),
        key=lambda path: int(path.stem[5:]),
    )
    references = sorted(
        (path for path in directory.glob("reference*.png") if re.fullmatch(r"reference\d+\.png", path.name)),
        key=lambda path: int(path.stem[9:]),
    )
    if not generated:
        return None
    return {
        **metadata,
        "id": request_id,
        "image_count": len(generated),
        "thumbnail_url": url_for("request_asset", request_id=request_id, filename=generated[0].name),
        "images": [
            {"name": path.name, "url": url_for("request_asset", request_id=request_id, filename=path.name)}
            for path in generated
        ],
        "references": [
            {"name": path.name, "url": url_for("request_asset", request_id=request_id, filename=path.name)}
            for path in references
        ],
        "detail_url": url_for("request_detail", request_id=request_id),
    }


def _store_request(metadata: dict, references: list[Image.Image], results: list[Image.Image]) -> dict:
    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"request_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    directory = output_dir / request_id
    directory.mkdir()
    try:
        (directory / "prompt.txt").write_text(metadata["prompt"], encoding="utf-8")
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for index, image in enumerate(references, 1):
            image.save(directory / f"reference{index}.png", format="PNG")
        for index, image in enumerate(results, 1):
            image.save(directory / f"image{index}.png", format="PNG")
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return _request_record(directory)


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model=_pipeline_model,
        model_idle_timeout=MODEL_IDLE_TIMEOUT,
        gpu_memory_fraction=GPU_MEMORY_FRACTION,
    )


@app.get("/models")
def models():
    return jsonify(
        models=list(AVAILABLE_MODELS),
        model_types=MODEL_TYPES,
        default_model=DEFAULT_MODEL,
    )


@app.get("/")
def portal():
    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        model_types=MODEL_TYPES,
        styles=AVAILABLE_STYLES,
        aspect_ratios=ASPECT_RATIOS,
        default_model=DEFAULT_MODEL,
        prompt_enhancement_enabled=PROMPT_ENHANCEMENT_LLM is not None,
    )


@app.post("/api/enhance-prompt")
def enhance_prompt():
    if PROMPT_ENHANCEMENT_LLM is None:
        return jsonify(error="Prompt enhancement is not configured"), 503

    body = request.get_json(silent=True) if request.is_json else None
    if body is not None and not isinstance(body, dict):
        return jsonify(error="Request body must be a JSON object"), 400
    prompt = request.form.get("prompt") or (body or {}).get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify(error="Provide a prompt to enhance"), 400
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify(error=f"Prompt must be at most {MAX_PROMPT_LENGTH} characters"), 400

    uploads = request.files.getlist("images") + request.files.getlist("images[]")
    try:
        with tempfile.TemporaryDirectory(prefix="prompt-enhancement-") as temp_dir:
            images = _collect_images(uploads, Path(temp_dir))

        user_content = [{"type": "text", "text": prompt}]
        for image in images:
            encoded_image = io.BytesIO()
            image.save(encoded_image, format="JPEG", quality=90)
            image_data = base64.b64encode(encoded_image.getvalue()).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            )
    except ValueError as error:
        return jsonify(error=str(error)), 400

    payload = json.dumps(
        {
            "model": PROMPT_ENHANCEMENT_LLM["model"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Enhance the user's prompt for high-quality image generation. Use any "
                        "reference images to understand the desired subjects, composition, "
                        "style, colors, and visual details. Preserve the user's intent while "
                        "adding concrete visual detail. Return only the enhanced prompt, with "
                        "no explanation, heading, or quotation marks."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.7,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if PROMPT_ENHANCEMENT_LLM["api_key"]:
        headers["Authorization"] = f"Bearer {PROMPT_ENHANCEMENT_LLM['api_key']}"
    upstream_request = urllib.request.Request(
        f"{PROMPT_ENHANCEMENT_LLM['url']}/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(upstream_request, timeout=60) as response:
            result = json.load(response)
        enhanced = result["choices"][0]["message"]["content"].strip()
        if not enhanced:
            raise ValueError("The LLM returned an empty prompt")
        return jsonify(prompt=enhanced)
    except urllib.error.HTTPError as error:
        app.logger.warning("Prompt enhancement LLM returned HTTP %s", error.code)
        return jsonify(error=f"Prompt enhancement service returned HTTP {error.code}"), 502
    except (urllib.error.URLError, TimeoutError):
        app.logger.exception("Could not reach prompt enhancement LLM")
        return jsonify(error="Could not reach the prompt enhancement service"), 502
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        app.logger.exception("Prompt enhancement LLM returned an invalid response")
        return jsonify(error="Prompt enhancement service returned an invalid response"), 502


@app.get("/api/history")
def history():
    output_dir = _output_dir()
    if not output_dir.exists():
        return jsonify(requests=[])

    requests = []
    for path in output_dir.iterdir():
        safe_path = _request_dir(path.name)
        if safe_path is None:
            continue
        record = _request_record(safe_path)
        if record is not None:
            requests.append(record)
    requests.sort(key=lambda item: item["created_at"], reverse=True)
    return jsonify(requests=requests)


@app.get("/api/history/<request_id>")
def request_detail(request_id: str):
    directory = _request_dir(request_id)
    record = _request_record(directory) if directory else None
    if record is None:
        return jsonify(error="Generation request not found"), 404
    return jsonify(request=record)


@app.get("/history/<request_id>/<filename>")
def request_asset(request_id: str, filename: str):
    path = _request_asset(request_id, filename)
    if path is None:
        return jsonify(error="Request image not found"), 404
    return send_file(path, mimetype="image/png", conditional=True)


@app.delete("/api/history/<request_id>")
def delete_history_request(request_id: str):
    directory = _request_dir(request_id)
    if directory is None:
        return jsonify(error="Generation request not found"), 404
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        return jsonify(error="Generation request not found"), 404
    return "", 204


@app.post("/generate")
def generate():
    json_body = request.get_json(silent=True) if request.is_json else None
    prompt = (
        request.form.get("prompt")
        or (json_body or {}).get("prompt")
        or ""
    ).strip()
    model_id = (
        request.form.get("pretrained_model")
        or request.form.get("model")
        or (json_body or {}).get("pretrained_model")
        or (json_body or {}).get("model")
        or DEFAULT_MODEL
    ).strip()
    style = (
        request.form.get("style")
        or (json_body or {}).get("style")
        or ""
    ).strip()
    aspect_ratio = (
        request.form.get("aspect_ratio")
        or (json_body or {}).get("aspect_ratio")
        or ""
    ).strip()
    image_count_value = (
        request.form.get("image_count")
        or (json_body or {}).get("image_count")
        or DEFAULT_IMAGE_COUNT
    )

    try:
        image_count = int(image_count_value)
    except (TypeError, ValueError):
        return jsonify(error="image_count must be an integer"), 400
    if not 1 <= image_count <= MAX_IMAGE_COUNT:
        return jsonify(error=f"image_count must be between 1 and {MAX_IMAGE_COUNT}"), 400

    if model_id not in AVAILABLE_MODELS:
        return jsonify(error="Unsupported model", models=list(AVAILABLE_MODELS)), 400
    if style and style not in STYLE_PROMPTS:
        return jsonify(error="Unsupported style", styles=list(AVAILABLE_STYLES)), 400
    if aspect_ratio and aspect_ratio not in ASPECT_RATIOS:
        return jsonify(
            error="Unsupported aspect ratio",
            aspect_ratios=list(ASPECT_RATIOS),
        ), 400

    uploads = request.files.getlist("images") + request.files.getlist("images[]")
    if MODEL_TYPES[model_id] == "text" and any(upload.filename for upload in uploads):
        return jsonify(error=f"Model '{model_id}' does not support reference images"), 400

    try:
        with tempfile.TemporaryDirectory(prefix="ai-image-generator-") as temp_dir:
            images = _collect_images(uploads, Path(temp_dir))

        if not prompt and not images:
            return jsonify(error="Provide a prompt, one or more images, or both"), 400

        source_prompt = prompt
        effective_prompt = f"{prompt} {STYLE_PROMPTS[style]}".strip() if style else prompt

        inputs = {"prompt": effective_prompt, "num_images_per_prompt": image_count}
        if aspect_ratio:
            inputs["width"], inputs["height"] = ASPECT_RATIOS[aspect_ratio]
        if images:
            inputs["image"] = images

        # Model loading and inference share a lock because both mutate GPU state.
        with _pipeline_lock:
            _cancel_idle_unload_locked()
            try:
                results = _get_pipeline(model_id)(**inputs).images
            finally:
                _schedule_idle_unload_locked()

        if len(results) != image_count:
            raise RuntimeError(f"Model returned {len(results)} images; expected {image_count}")
        created_at = datetime.now().astimezone().isoformat()
        record = _store_request(
            {
                "prompt": source_prompt,
                "effective_prompt": effective_prompt,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "model": model_id,
                "created_at": created_at,
            },
            images,
            results,
        )
        return jsonify(request=record), 201
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        app.logger.exception("Image generation failed")
        return jsonify(error="Image generation failed", detail=str(error)), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="Request exceeds MAX_CONTENT_LENGTH_MB"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "9000")), threaded=True)
