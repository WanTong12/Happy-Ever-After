import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import binascii

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


def normalize_audio_media_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.split(";", 1)[0].strip().lower()
    return candidate if candidate.startswith("audio/") else None


def get_audio_media_type_hint(node: Any) -> str | None:
    if not isinstance(node, dict):
        return None

    for key in ("mime_type", "mimeType", "content_type", "contentType", "type", "format"):
        media_type = normalize_audio_media_type(node.get(key))
        if media_type:
            return media_type
    return None


def decode_audio_data_url(value: str) -> tuple[bytes, str] | None:
    data_url_match = re.match(
        r"^data:(audio/[\w.+-]+);base64,(.+)$",
        value.strip(),
        flags=re.IGNORECASE,
    )
    if not data_url_match:
        return None

    media_type = data_url_match.group(1).lower()
    encoded = re.sub(r"\s+", "", data_url_match.group(2))
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return None

    return decoded, media_type


def decode_base64_audio(value: str) -> bytes | None:
    encoded = re.sub(r"\s+", "", value or "")
    if len(encoded) < 128:
        return None

    if not re.fullmatch(r"[A-Za-z0-9+/=]+", encoded):
        return None

    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return None

    return decoded if len(decoded) >= 256 else None


def extract_embedded_audio_blob(payload: dict[str, Any]) -> tuple[bytes, str] | None:
    audio_key_hints = (
        "audio",
        "speech",
        "voice",
        "tts",
        "wav",
        "mp3",
        "ogg",
        "m4a",
    )
    base64_value_keys = {
        "audio",
        "audio_base64",
        "audiobase64",
        "audio_data",
        "audiodata",
        "base64",
        "b64",
        "b64_json",
        "data",
        "content",
    }

    def walk(node: Any, media_hint: str | None = None) -> tuple[bytes, str] | None:
        if isinstance(node, str):
            parsed_data_url = decode_audio_data_url(node)
            if parsed_data_url:
                return parsed_data_url
            return None

        if isinstance(node, list):
            for item in node:
                found = walk(item, media_hint)
                if found:
                    return found
            return None

        if not isinstance(node, dict):
            return None

        local_media_hint = media_hint or get_audio_media_type_hint(node)

        for raw_key, value in node.items():
            key = str(raw_key).lower()

            if isinstance(value, str):
                parsed_data_url = decode_audio_data_url(value)
                if parsed_data_url:
                    return parsed_data_url

                should_try_base64 = (
                    key in base64_value_keys
                    or any(hint in key for hint in audio_key_hints)
                    or bool(local_media_hint)
                )
                if should_try_base64:
                    decoded = decode_base64_audio(value)
                    if decoded:
                        return decoded, (local_media_hint or "audio/mpeg")

            found = walk(value, local_media_hint)
            if found:
                return found

        return None

    roots = extract_outputs(payload)
    if not roots:
        roots = [payload]

    for root in roots:
        found = walk(root)
        if found:
            return found

    return None


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


def min_branch_count_for_pages(page_count: int) -> int:
    if page_count <= 5:
        return 1
    if page_count <= 10:
        return 2
    return 3


def random_branch_targets(page_count: int) -> list[str]:
    target_count = min_branch_count_for_pages(page_count)

    # Prefer interior pages so story can introduce/explore before branching.
    eligible = list(range(3, page_count)) if page_count >= 5 else list(range(2, page_count))
    if len(eligible) < target_count:
        eligible = list(range(1, page_count))

    random.shuffle(eligible)
    selected = sorted(eligible[:target_count])
    return [f"p{index}" for index in selected]


def normalize_story_blueprint(
    raw: dict[str, Any], requested_pages: int, branch_targets: list[str] | None = None
) -> dict[str, Any]:
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
    branch_positions: list[int] = []
    for page_index, page in enumerate(normalized_pages):
        if len(page["choices"]) == 2:
            branch_count += 1
            branch_positions.append(page_index + 1)

        for choice in page["choices"]:
            next_page_id = choice["nextPageId"]
            if not next_page_id or next_page_id not in ids:
                raise ValueError(
                    f"Page {page['id']} has an invalid branch target: {next_page_id or '(empty)'}"
                )

    if normalized_pages and len(normalized_pages[0]["choices"]) == 2:
        raise ValueError("The first page must introduce the story and cannot be a branch point.")

    required_branches = min_branch_count_for_pages(requested_pages)
    if branch_count < required_branches:
        raise ValueError(
            f"Story must contain at least {required_branches} Option A / Option B branch points."
        )

    if requested_pages >= 7 and required_branches >= 2:
        has_early_branch = any(position <= requested_pages // 2 for position in branch_positions)
        has_late_branch = any(position > requested_pages // 2 for position in branch_positions)
        if not (has_early_branch and has_late_branch):
            raise ValueError(
                "Branch points should be spread across the story (early and later pages)."
            )

    required_branch_targets = [target for target in (branch_targets or []) if target in ids]
    missing_targets = [
        page_id
        for page_id in required_branch_targets
        if len(next(page["choices"] for page in normalized_pages if page["id"] == page_id)) != 2
    ]
    if missing_targets:
        raise ValueError(
            f"Story missed required random branch pages: {', '.join(missing_targets)}"
        )

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


def build_visual_consistency_guide(
    parent_prompt: str, style: str, age: int, blueprint: dict[str, Any]
) -> str:
    page_samples = "\n".join(
        [
            f"- {page.get('id')}: {clamp_words(str(page.get('text') or ''), 18)}"
            for page in (blueprint.get("pages") or [])[:6]
            if isinstance(page, dict)
        ]
    )

    prompt = (
        "Create a concise visual consistency guide for a children's storybook.\n"
        "Return strict JSON only with keys: character_anchor, setting_anchor, palette_anchor, style_anchor.\n"
        "Each value must be one short sentence (max 22 words).\n"
        "Keep this kid-friendly and specific.\n\n"
        f"Story request: {parent_prompt}\n"
        f"Visual style: {style}\n"
        f"Target age: {age}\n"
        f"Story title: {blueprint.get('title', '')}\n"
        f"Story summary: {blueprint.get('summary', '')}\n"
        f"Page excerpts:\n{page_samples}"
    )

    try:
        payload = run_any_llm(prompt)
        parsed = extract_json_from_text(extract_text(payload))

        character_anchor = clamp_words(str(parsed.get("character_anchor") or ""), 22)
        setting_anchor = clamp_words(str(parsed.get("setting_anchor") or ""), 22)
        palette_anchor = clamp_words(str(parsed.get("palette_anchor") or ""), 22)
        style_anchor = clamp_words(str(parsed.get("style_anchor") or ""), 22)

        if character_anchor and setting_anchor and palette_anchor and style_anchor:
            return " ".join(
                [
                    f"Character anchor: {character_anchor}",
                    f"Setting anchor: {setting_anchor}",
                    f"Palette anchor: {palette_anchor}",
                    f"Style anchor: {style_anchor}",
                ]
            )
    except Exception:
        pass

    return (
        "Keep the same main character identity, body shape, colors, outfit, and accessories across every page. "
        "Keep the same world design language, lighting mood, and color palette across all illustrations."
    )


def generate_page_images(
    page: dict[str, Any],
    page_index: int,
    style: str,
    age: int,
    visual_consistency_guide: str,
) -> list[str]:
    image_prompts = page.get("imagePrompts")
    prompts_for_page = image_prompts if isinstance(image_prompts, list) else []

    desired_image_count = (
        3
        if len(prompts_for_page) >= 3 and page_index % 9 == 0
        else 2
        if len(prompts_for_page) >= 2 and page_index % 5 == 0
        else 1
    )

    prompts = prompts_for_page[:desired_image_count]
    page_images: list[str] = []

    for prompt_idx, prompt in enumerate(prompts):
        illustration_prompt = " ".join(
            [
                f"Children's storybook illustration in {style} style.",
                f"For kids age {age}.",
                f"Scene: {prompt}.",
                f"Global consistency guide: {visual_consistency_guide}",
                "Keep the same protagonist design and world look from previous pages.",
                "Bright, kind, playful, clean anatomy, non-distorted, no scary content.",
                "No words, letters, numbers, logos, captions, signs, speech bubbles, or watermarks anywhere.",
                f"Image {prompt_idx + 1} of {len(prompts)} for page {page_index + 1}.",
            ]
        )
        page_images.append(run_flux_image(illustration_prompt))

    return page_images


def run_gemini_tts(text: str) -> tuple[bytes, str]:
    script_text = f"Narrator: {text}"

    task_id, _ = submit_wavespeed_task(
        "/google/gemini-2.5-flash/text-to-speech",
        {
            "text": script_text,
            "language": "English (United States)",
            "speakers": [
                {
                    "speaker": "Narrator",
                    "voice": "Achernar",
                }
            ],
            "enable_sync_mode": False,
        },
    )

    result = wait_for_wavespeed_task(task_id, timeout_s=180)
    urls = extract_output_urls(result) or extract_urls(result)

    audio_url = next((url for url in urls if re.search(r"\.(mp3|wav|ogg|m4a)(\?|$)", url, flags=re.I)), None)
    if not audio_url:
        audio_url = urls[0] if urls else None

    if audio_url:
        return download_audio_blob(audio_url)

    embedded_audio = extract_embedded_audio_blob(result)
    if embedded_audio:
        return embedded_audio

    raise RuntimeError("Audio generation completed but no playable audio output was returned.")


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


def build_audio_cache_key(story_id: str, page_id: str, text: str) -> str:
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{story_id}:{page_id}:{text_hash}"


def get_or_create_story_audio_url(story_id: str, page_id: str, text: str) -> tuple[str, bool]:
    cache_key = build_audio_cache_key(story_id, page_id, text)

    with state_lock:
        cached = audio_cache.get(cache_key)

    if cached:
        return cached, True

    prefixed = f"Narrate this for children slowly, warmly, and clearly: {text}"
    last_error: Exception | None = None
    audio_bytes: bytes | None = None
    content_type = "audio/mpeg"

    for attempt in range(3):
        try:
            audio_bytes, content_type = run_gemini_tts(prefixed)
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.2 * (attempt + 1))

    if audio_bytes is None:
        raise RuntimeError(
            f"Narration generation failed after retries: {last_error or 'unknown error'}"
        )

    audio_id = str(uuid.uuid4())
    playback_url = f"/api/story/audio/files/{audio_id}"

    with state_lock:
        audio_blob_cache[audio_id] = {
            "bytes": audio_bytes,
            "contentType": content_type,
            "createdAt": now_iso(),
        }
        audio_cache[cache_key] = playback_url

    return playback_url, False


def build_story_prompt(
    prompt: str,
    style: str,
    page_count: int,
    age: int,
    branch_targets: list[str],
) -> str:
    required_branches = min_branch_count_for_pages(page_count)
    branch_target_text = ", ".join(branch_targets)

    return (
        f"You are writing a children storybook for age {age}.\n\n"
        "Create a fully kid-safe interactive branching story in strict JSON.\n\n"
        "Constraints:\n"
        f"- Exactly {page_count} pages in total.\n"
        f"- Use page IDs in this exact sequence: p1 to p{page_count}.\n"
        f"- Reading level suitable for age {age}.\n"
        "- Every page text must be one short paragraph (max 40 words).\n"
        "- No scary, violent, hateful, romantic, or inappropriate content.\n"
        "- p1 must introduce the world/character and must NOT include branching choices.\n"
        "- Story arc must feel engaging: magical hook, playful challenge, turning point, and heartwarming resolution.\n"
        "- Keep each page vivid and active with curiosity, emotion, and momentum for children.\n"
        f"- Create at least {required_branches} branch points where there are exactly two choices: Option A and Option B.\n"
        f"- Required branch anchor pages for this run: {branch_target_text}.\n"
        "- Branches should be naturally spread around the story where they make sense (not all clustered together).\n"
        "- Choice A and Choice B should lead to meaningfully different scenes.\n"
        "- Each choice must point to a valid page id in nextPageId.\n"
        "- Keep language warm, simple, friendly, and imaginative.\n"
        "- imagePrompts must be short, clear, and kid friendly with no distortion.\n"
        "- Absolutely no words, letters, numbers, logos, captions, signs, or watermarks in illustrations.\n"
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
            message="Writing an engaging story and branching choices...",
        )

        branch_targets = random_branch_targets(input_data["pageCount"])
        attempts = 3
        blueprint: dict[str, Any] | None = None
        last_validation_error = ""

        for attempt_index in range(attempts):
            llm_result = run_any_llm(
                build_story_prompt(
                    prompt=input_data["prompt"],
                    style=input_data["style"],
                    page_count=input_data["pageCount"],
                    age=input_data["age"],
                    branch_targets=branch_targets,
                )
            )

            story_text = extract_text(llm_result)
            try:
                blueprint = normalize_story_blueprint(
                    extract_json_from_text(story_text),
                    input_data["pageCount"],
                    branch_targets=branch_targets,
                )
                break
            except Exception as exc:
                last_validation_error = str(exc)
                if attempt_index < attempts - 1:
                    update_job(
                        job_id,
                        progress=10,
                        message="Refining story structure and branch timing...",
                    )

        if blueprint is None:
            raise ValueError(last_validation_error or "Story format validation failed.")

        update_job(
            job_id,
            status="generating_images",
            progress=18,
            message="Painting storybook images...",
        )

        pages: list[dict[str, Any]] = [{**page} for page in blueprint["pages"]]
        page_count = len(pages)

        update_job(
            job_id,
            progress=20,
            message="Preparing a consistent visual style guide...",
        )
        visual_consistency_guide = build_visual_consistency_guide(
            parent_prompt=input_data["prompt"],
            style=input_data["style"],
            age=input_data["age"],
            blueprint=blueprint,
        )

        for index, page in enumerate(pages):
            pages[index]["imageUrls"] = generate_page_images(
                page,
                index,
                input_data["style"],
                input_data["age"],
                visual_consistency_guide,
            )

            progress = min(70, 18 + round(((index + 1) / page_count) * 52))
            update_job(
                job_id,
                progress=progress,
                message=f"Generated page {index + 1} of {page_count}",
            )

        update_job(
            job_id,
            status="generating_audio",
            progress=72,
            message="Recording narration for every page...",
        )

        for index, page in enumerate(pages):
            audio_url, _ = get_or_create_story_audio_url(job_id, page["id"], page["text"])
            pages[index]["audioUrl"] = audio_url

            progress = min(96, 72 + round(((index + 1) / page_count) * 24))
            update_job(
                job_id,
                progress=progress,
                message=f"Narrated page {index + 1} of {page_count}",
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

        audio_url, cached = get_or_create_story_audio_url(story_id, page_id, text)
        return {"audioUrl": audio_url, "cached": cached}
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
