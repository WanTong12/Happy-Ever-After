import hashlib
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"

WAVESPEED_BASE_URL = "https://api.wavespeed.ai/api/v3"
WAVESPEED_API_KEY = os.getenv("WAVESPEED_API_KEY", "")
PORT = int(os.getenv("PORT", "3000"))

ALLOWED_STYLES = [
    "Watercolor Dream",
    "Cartoon Pop",
    "Paper Cutout",
    "Crayon Sketch",
    "Bedtime Glow",
]

MIN_AGE = 2
MAX_AGE = 8
MAX_PAGES = 15

jobs: dict[str, dict[str, Any]] = {}
audio_cache: dict[str, str] = {}
audio_blob_cache: dict[str, dict[str, Any]] = {}
state_lock = threading.Lock()

app = FastAPI(title="Happy Ever Storybook")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_api_key() -> None:
    if not WAVESPEED_API_KEY:
        raise ValueError("Missing WAVESPEED_API_KEY. Add it in your .env file.")


def is_http_url(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith(("http://", "https://"))


def collect_strings(input_value: Any, bucket: list[str] | None = None) -> list[str]:
    if bucket is None:
        bucket = []

    if input_value is None:
        return bucket

    if isinstance(input_value, str):
        bucket.append(input_value)
        return bucket

    if isinstance(input_value, list):
        for item in input_value:
            collect_strings(item, bucket)
        return bucket

    if isinstance(input_value, dict):
        for value in input_value.values():
            collect_strings(value, bucket)

    return bucket


def extract_task_id(payload: dict[str, Any]) -> str | None:
    return (
        payload.get("data", {}).get("id")
        or payload.get("id")
        or payload.get("requestId")
        or payload.get("request_id")
    )


def extract_status(payload: dict[str, Any]) -> str:
    return (
        payload.get("data", {}).get("status")
        or payload.get("status")
        or payload.get("data", {}).get("state")
        or payload.get("state")
        or "unknown"
    )


def extract_urls(payload: dict[str, Any]) -> list[str]:
    return [text for text in collect_strings(payload, []) if is_http_url(text)]


def extract_outputs(payload: dict[str, Any]) -> list[Any]:
    outputs: list[Any] = []

    for container in (payload.get("data"), payload):
        if not isinstance(container, dict):
            continue

        value = container.get("outputs")
        if isinstance(value, list):
            outputs.extend(value)
        elif value is not None:
            outputs.append(value)

    return outputs


def extract_output_urls(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    for output in extract_outputs(payload):
        if isinstance(output, str) and is_http_url(output):
            urls.append(output)
            continue

        if isinstance(output, dict):
            prioritized = [
                output.get("url"),
                output.get("audio_url"),
                output.get("image_url"),
                output.get("file"),
                output.get("href"),
            ]
            for item in prioritized:
                if is_http_url(item):
                    urls.append(str(item))

            urls.extend(
                text for text in collect_strings(output, []) if is_http_url(text)
            )

    # Preserve order and uniqueness.
    return list(dict.fromkeys(urls))


def extract_text(payload: dict[str, Any]) -> str:
    candidates = [
        text.strip()
        for text in collect_strings(payload, [])
        if isinstance(text, str) and not is_http_url(text) and text.strip()
    ]
    candidates.sort(key=len, reverse=True)
    return candidates[0] if candidates else ""


def extract_json_from_text(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Model returned an empty story payload.")

    fenced_match = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced_match and fenced_match.group(1):
        return json.loads(fenced_match.group(1))

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        return json.loads(text[first_brace : last_brace + 1])

    return json.loads(text)


def contains_unsafe_words(text: str) -> bool:
    blocked = [
        "blood",
        "kill",
        "weapon",
        "gore",
        "hate",
        "curse",
        "drugs",
        "alcohol",
        "sex",
    ]
    lowered = (text or "").lower()
    return any(word in lowered for word in blocked)


def clamp_words(text: str, max_words: int = 40) -> str:
    words = [word for word in str(text or "").strip().split() if word]
    if len(words) <= max_words:
        return " ".join(words)
    return f"{' '.join(words[:max_words])}."


def normalize_story_blueprint(raw: dict[str, Any], requested_pages: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Story blueprint is not a valid object.")

    pages = raw.get("pages")
    if not isinstance(pages, list):
        pages = []

    if len(pages) != requested_pages:
        raise ValueError(
            f"Story must contain exactly {requested_pages} pages, received {len(pages)}."
        )

    ids: set[str] = set()
    normalized_pages: list[dict[str, Any]] = []

    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"Page at index {index} is not valid.")

        page_id = str(page.get("id") or f"p{index + 1}")
        if page_id in ids:
            raise ValueError(f"Duplicate page id found: {page_id}")
        ids.add(page_id)

        text = clamp_words(str(page.get("text") or ""), 40)
        if not text:
            raise ValueError(f"Page {page_id} is missing narration text.")
        if contains_unsafe_words(text):
            raise ValueError(f"Unsafe words detected in page {page_id}.")

        image_prompts_raw = page.get("imagePrompts")
        image_prompts = (
            [str(item).strip() for item in image_prompts_raw if str(item).strip()]
            if isinstance(image_prompts_raw, list)
            else []
        )

        choices_raw = page.get("choices")
        choices: list[dict[str, str]] = []
        if isinstance(choices_raw, list):
            for idx, choice in enumerate(choices_raw[:2]):
                if not isinstance(choice, dict):
                    continue
                choices.append(
                    {
                        "label": "Option A" if idx == 0 else "Option B",
                        "text": str(choice.get("text") or f"Choose path {idx + 1}").strip(),
                        "nextPageId": str(choice.get("nextPageId") or "").strip(),
                    }
                )

        normalized_pages.append(
            {
                "id": page_id,
                "text": text,
                "textPosition": "top" if page.get("textPosition") == "top" else "bottom",
                "imagePrompts": image_prompts[:3]
                if image_prompts
                else [f"Main story illustration for page {index + 1}"],
                "choices": choices,
            }
        )

    branch_count = 0
    for page in normalized_pages:
        if len(page["choices"]) == 2:
            branch_count += 1

        for choice in page["choices"]:
            next_page_id = choice["nextPageId"]
            if not next_page_id or next_page_id not in ids:
                raise ValueError(
                    f"Page {page['id']} has an invalid branch target: {next_page_id or '(empty)'}"
                )

    if branch_count < 1:
        raise ValueError("Story must contain at least one Option A / Option B branch point.")

    start_page_id = str(raw.get("startPageId") or normalized_pages[0]["id"])
    if start_page_id not in ids:
        raise ValueError("Invalid startPageId in generated story.")

    return {
        "title": str(raw.get("title") or "My Interactive Storybook")[:80],
        "summary": str(raw.get("summary") or "")[:200],
        "startPageId": start_page_id,
        "pages": normalized_pages,
    }


def submit_wavespeed_task(endpoint: str, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    assert_api_key()

    response = requests.post(
        f"{WAVESPEED_BASE_URL}{endpoint}",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {WAVESPEED_API_KEY}",
        },
        json=body,
        timeout=120,
    )

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if not response.ok:
        raise RuntimeError(payload.get("message") or f"WaveSpeed submit failed ({response.status_code}).")

    task_id = extract_task_id(payload)
    if not task_id:
        raise RuntimeError("WaveSpeed did not return a task id.")

    return task_id, payload


def fetch_wavespeed_result(task_id: str) -> dict[str, Any]:
    urls = [
        f"{WAVESPEED_BASE_URL}/predictions/{task_id}/result",
        f"{WAVESPEED_BASE_URL}/predictions/{task_id}",
    ]

    last_error: Exception | None = None

    for url in urls:
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {WAVESPEED_API_KEY}"},
                timeout=120,
            )
            if not response.ok:
                last_error = RuntimeError(f"WaveSpeed poll failed ({response.status_code})")
                continue
            return response.json()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(str(last_error or "Unable to fetch WaveSpeed prediction result."))


def wait_for_wavespeed_task(task_id: str, timeout_s: int = 240) -> dict[str, Any]:
    started_at = time.time()

    while time.time() - started_at < timeout_s:
        payload = fetch_wavespeed_result(task_id)
        status = str(extract_status(payload)).lower()

        if status in {"completed", "succeeded", "success"}:
            return payload

        if status in {"failed", "error", "canceled"}:
            message = extract_text(payload) or payload.get("message") or "WaveSpeed task failed."
            raise RuntimeError(message)

        time.sleep(1.8)

    raise RuntimeError("WaveSpeed task timed out.")


def run_any_llm(prompt: str) -> dict[str, Any]:
    task_id, _ = submit_wavespeed_task(
        "/wavespeed-ai/any-llm",
        {
            "prompt": prompt,
            "model": "google/gemini-2.5-flash",
            "reasoning": False,
            "priority": "latency",
            "temperature": 0.5,
            "max_tokens": 3500,
            "enable_sync_mode": False,
        },
    )
    return wait_for_wavespeed_task(task_id, timeout_s=180)


def run_flux_image(prompt: str) -> str:
    task_id, _ = submit_wavespeed_task(
        "/wavespeed-ai/flux-dev",
        {
            "prompt": prompt,
            "size": "1024*1024",
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "num_images": 1,
            "output_format": "jpeg",
            "enable_sync_mode": False,
        },
    )

    result = wait_for_wavespeed_task(task_id, timeout_s=240)
    urls = extract_output_urls(result) or extract_urls(result)

    image_url = next((url for url in urls if re.search(r"\.(png|jpe?g|webp)(\?|$)", url, flags=re.I)), None)
    if not image_url:
        image_url = urls[0] if urls else None

    if not image_url:
        raise RuntimeError("Image generation completed but no image URL was returned.")

    return image_url


def run_qwen_tts(text: str) -> str:
    task_id, _ = submit_wavespeed_task(
        "/wavespeed-ai/qwen3-tts/text-to-speech",
        {
            "text": text,
            "language": "auto",
            "voice": "Vivian",
            "style_instruction": "Warm and friendly, not too fast, expressive like a bedtime storyteller.",
        },
    )

    result = wait_for_wavespeed_task(task_id, timeout_s=180)
    urls = extract_output_urls(result) or extract_urls(result)

    audio_url = next((url for url in urls if re.search(r"\.(mp3|wav|ogg|m4a)(\?|$)", url, flags=re.I)), None)
    if not audio_url:
        audio_url = urls[0] if urls else None

    if not audio_url:
        raise RuntimeError("Audio generation completed but no audio URL was returned.")

    return audio_url


def download_audio_blob(source_url: str) -> tuple[bytes, str]:
    attempts = [
        {"Authorization": f"Bearer {WAVESPEED_API_KEY}"},
        {},
    ]

    last_error: Exception | None = None
    for headers in attempts:
        try:
            response = requests.get(source_url, headers=headers, timeout=120)
            if not response.ok:
                last_error = RuntimeError(
                    f"Audio download failed ({response.status_code})."
                )
                continue

            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if not content_type.startswith("audio/"):
                guessed_type = mimetypes.guess_type(source_url)[0]
                content_type = guessed_type if guessed_type and guessed_type.startswith("audio/") else "audio/mpeg"

            return response.content, content_type
        except Exception as exc:
            last_error = exc

    raise RuntimeError(str(last_error or "Failed to download generated narration audio."))


def build_story_prompt(prompt: str, style: str, page_count: int, age: int) -> str:
    return (
        f"You are writing a children storybook for age {age}.\n\n"
        "Create a fully kid-safe interactive branching story in strict JSON.\n\n"
        "Constraints:\n"
        f"- Exactly {page_count} pages in total.\n"
        f"- Reading level suitable for age {age}.\n"
        "- Every page text must be one short paragraph (max 40 words).\n"
        "- No scary, violent, hateful, romantic, or inappropriate content.\n"
        "- At least one branch point where there are exactly two choices: Option A and Option B.\n"
        "- Each choice must point to a valid page id in nextPageId.\n"
        "- Keep language warm, simple, friendly, and imaginative.\n"
        "- imagePrompts must be short, clear, and kid friendly with no distortion and no text in image.\n"
        "- Mostly 1 image prompt per page, occasionally 2-3 if needed (max 3).\n"
        "- textPosition must be either top or bottom.\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "title": "string",\n'
        '  "summary": "string",\n'
        '  "startPageId": "p1",\n'
        '  "pages": [\n'
        "    {\n"
        '      "id": "p1",\n'
        '      "text": "string",\n'
        '      "textPosition": "top or bottom",\n'
        '      "imagePrompts": ["string"],\n'
        '      "choices": [\n'
        '        { "text": "string", "nextPageId": "p2" },\n'
        '        { "text": "string", "nextPageId": "p3" }\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Parent/child story request: {prompt}\n"
        f"Visual style: {style}"
    )


def get_job(job_id: str) -> dict[str, Any] | None:
    with state_lock:
        return jobs.get(job_id)


def update_job(job_id: str, **updates: Any) -> None:
    with state_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updatedAt"] = now_iso()


def generate_story(job_id: str, input_data: dict[str, Any]) -> None:
    if not get_job(job_id):
        return

    try:
        update_job(
            job_id,
            status="generating_story",
            progress=8,
            message="Writing story and branching choices...",
        )

        llm_result = run_any_llm(
            build_story_prompt(
                prompt=input_data["prompt"],
                style=input_data["style"],
                page_count=input_data["pageCount"],
                age=input_data["age"],
            )
        )

        story_text = extract_text(llm_result)
        blueprint = normalize_story_blueprint(
            extract_json_from_text(story_text), input_data["pageCount"]
        )

        update_job(
            job_id,
            status="generating_images",
            progress=18,
            message="Painting storybook images...",
        )

        pages: list[dict[str, Any]] = []

        for index, page in enumerate(blueprint["pages"]):
            page_images: list[str] = []

            desired_image_count = (
                3
                if len(page["imagePrompts"]) >= 3 and index % 9 == 0
                else 2
                if len(page["imagePrompts"]) >= 2 and index % 5 == 0
                else 1
            )

            prompts = page["imagePrompts"][:desired_image_count]
            for prompt_idx, prompt in enumerate(prompts):
                illustration_prompt = " ".join(
                    [
                        f"Children's storybook illustration in {input_data['style']} style.",
                        f"For kids age {input_data['age']}.",
                        f"Scene: {prompt}.",
                        "Bright, kind, playful, clean anatomy, non-distorted, no scary content, no text in image.",
                        f"Image {prompt_idx + 1} of {len(prompts)} for page {index + 1}.",
                    ]
                )
                page_images.append(run_flux_image(illustration_prompt))

            with_images = {**page, "imageUrls": page_images}
            pages.append(with_images)

            progress = min(96, 18 + round(((index + 1) / len(blueprint["pages"])) * 78))
            update_job(
                job_id,
                progress=progress,
                message=f"Generated page {index + 1} of {len(blueprint['pages'])}",
            )

        story = {
            "id": job_id,
            "title": blueprint["title"],
            "summary": blueprint["summary"],
            "style": input_data["style"],
            "age": input_data["age"],
            "pageCount": input_data["pageCount"],
            "startPageId": blueprint["startPageId"],
            "pages": pages,
            "createdAt": now_iso(),
        }

        update_job(
            job_id,
            status="completed",
            progress=100,
            message="Storybook is ready!",
            story=story,
            error=None,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            message="Could not generate storybook.",
            error=str(exc),
        )


def validate_story_request(body: dict[str, Any]) -> dict[str, Any]:
    prompt = str(body.get("prompt") or "").strip()
    style = str(body.get("style") or "").strip()

    try:
        page_count = int(body.get("pageCount"))
    except (TypeError, ValueError):
        page_count = -1

    try:
        age = int(body.get("age"))
    except (TypeError, ValueError):
        age = -1

    if not prompt or len(prompt) < 8:
        raise ValueError("Please provide a more detailed story prompt.")

    if style not in ALLOWED_STYLES:
        raise ValueError("Please choose one of the 5 supported styles.")

    if page_count < 3 or page_count > MAX_PAGES:
        raise ValueError(f"Page count must be between 3 and {MAX_PAGES}.")

    if age < MIN_AGE or age > MAX_AGE:
        raise ValueError(f"Age must be between {MIN_AGE} and {MAX_AGE}.")

    return {
        "prompt": prompt,
        "style": style,
        "pageCount": page_count,
        "age": age,
    }


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    return {
        "styles": ALLOWED_STYLES,
        "minAge": MIN_AGE,
        "maxAge": MAX_AGE,
        "maxPages": MAX_PAGES,
    }


@app.post("/api/story/jobs", status_code=202)
def create_story_job(body: dict[str, Any]) -> dict[str, str]:
    try:
        assert_api_key()
        input_data = validate_story_request(body or {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    now = now_iso()

    with state_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued for generation...",
            "story": None,
            "error": None,
            "createdAt": now,
            "updatedAt": now,
        }

    thread = threading.Thread(target=generate_story, args=(job_id, input_data), daemon=True)
    thread.start()

    return {"jobId": job_id}


@app.get("/api/story/jobs/{job_id}")
def get_story_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error"),
        "story": job.get("story"),
        "updatedAt": job["updatedAt"],
    }


@app.post("/api/story/audio")
def create_story_audio(body: dict[str, Any]) -> dict[str, Any]:
    try:
        assert_api_key()

        story_id = str(body.get("storyId") or "").strip()
        page_id = str(body.get("pageId") or "").strip()
        text = str(body.get("text") or "").strip()

        if not story_id or not page_id or not text:
            raise ValueError("storyId, pageId and text are required.")

        if contains_unsafe_words(text):
            raise ValueError("Narration blocked due to unsafe words.")

        cache_key = f"{story_id}:{page_id}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"

        with state_lock:
            cached = audio_cache.get(cache_key)

        if cached:
            return {"audioUrl": cached, "cached": True}

        prefixed = f"Narrate this for children slowly, warmly, and clearly: {text}"
        source_audio_url = run_qwen_tts(prefixed)
        audio_bytes, content_type = download_audio_blob(source_audio_url)

        audio_id = str(uuid.uuid4())
        playback_url = f"/api/story/audio/files/{audio_id}"

        with state_lock:
            audio_blob_cache[audio_id] = {
                "bytes": audio_bytes,
                "contentType": content_type,
                "createdAt": now_iso(),
            }
            audio_cache[cache_key] = playback_url

        return {"audioUrl": playback_url, "cached": False}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/story/audio/files/{audio_id}")
def get_story_audio_file(audio_id: str) -> Response:
    with state_lock:
        item = audio_blob_cache.get(audio_id)

    if not item:
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return Response(
        content=item["bytes"],
        media_type=item["contentType"],
        headers={"Cache-Control": "public, max-age=86400"},
    )


app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
