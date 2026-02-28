import crypto from "crypto";
import path from "path";
import { fileURLToPath } from "url";

import dotenv from "dotenv";
import express from "express";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const WAVESPEED_BASE_URL = "https://api.wavespeed.ai/api/v3";
const WAVESPEED_API_KEY = process.env.WAVESPEED_API_KEY;

const ALLOWED_STYLES = [
  "Watercolor Dream",
  "Cartoon Pop",
  "Paper Cutout",
  "Crayon Sketch",
  "Bedtime Glow"
];

const MIN_AGE = 2;
const MAX_AGE = 8;
const MAX_PAGES = 15;

const jobs = new Map();
const audioCache = new Map();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

function assertApiKey() {
  if (!WAVESPEED_API_KEY) {
    throw new Error("Missing WAVESPEED_API_KEY. Add it in your .env file.");
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isHttpUrl(value) {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

function collectStrings(input, bucket = []) {
  if (input == null) return bucket;

  if (typeof input === "string") {
    bucket.push(input);
    return bucket;
  }

  if (Array.isArray(input)) {
    for (const item of input) collectStrings(item, bucket);
    return bucket;
  }

  if (typeof input === "object") {
    for (const value of Object.values(input)) collectStrings(value, bucket);
  }

  return bucket;
}

function extractTaskId(payload) {
  return (
    payload?.data?.id ||
    payload?.id ||
    payload?.requestId ||
    payload?.request_id ||
    null
  );
}

function extractStatus(payload) {
  return (
    payload?.data?.status ||
    payload?.status ||
    payload?.data?.state ||
    payload?.state ||
    "unknown"
  );
}

function extractUrls(payload) {
  const raw = collectStrings(payload, []);
  return raw.filter((item) => isHttpUrl(item));
}

function extractText(payload) {
  const strings = collectStrings(payload, []);
  const candidates = strings
    .filter((s) => typeof s === "string")
    .filter((s) => !isHttpUrl(s))
    .map((s) => s.trim())
    .filter(Boolean)
    .sort((a, b) => b.length - a.length);

  return candidates[0] || "";
}

function extractJsonFromText(text) {
  if (!text) throw new Error("Model returned an empty story payload.");

  const fenced = text.match(/```json\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    return JSON.parse(fenced[1]);
  }

  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    return JSON.parse(text.slice(firstBrace, lastBrace + 1));
  }

  return JSON.parse(text);
}

function containsUnsafeWords(text) {
  const blocked = [
    "blood",
    "kill",
    "weapon",
    "gore",
    "hate",
    "curse",
    "drugs",
    "alcohol",
    "sex"
  ];

  const lower = (text || "").toLowerCase();
  return blocked.some((word) => lower.includes(word));
}

function clampWords(text, maxWords = 40) {
  const words = String(text || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (words.length <= maxWords) {
    return words.join(" ");
  }

  return `${words.slice(0, maxWords).join(" ")}.`;
}

function normalizeStoryBlueprint(raw, requestedPages) {
  if (!raw || typeof raw !== "object") {
    throw new Error("Story blueprint is not a valid object.");
  }

  const pages = Array.isArray(raw.pages) ? raw.pages : [];
  if (pages.length !== requestedPages) {
    throw new Error(
      `Story must contain exactly ${requestedPages} pages, received ${pages.length}.`
    );
  }

  const ids = new Set();
  const normalizedPages = pages.map((page, index) => {
    const id = String(page.id || `p${index + 1}`);
    if (ids.has(id)) {
      throw new Error(`Duplicate page id found: ${id}`);
    }
    ids.add(id);

    const text = clampWords(page.text, 40);
    if (!text) {
      throw new Error(`Page ${id} is missing narration text.`);
    }
    if (containsUnsafeWords(text)) {
      throw new Error(`Unsafe words detected in page ${id}.`);
    }

    const imagePrompts = Array.isArray(page.imagePrompts)
      ? page.imagePrompts.map((p) => String(p).trim()).filter(Boolean)
      : [];

    const choices = Array.isArray(page.choices)
      ? page.choices.slice(0, 2).map((choice, idx) => ({
          label: idx === 0 ? "Option A" : "Option B",
          text: String(choice.text || "").trim() || `Choose path ${idx + 1}`,
          nextPageId: String(choice.nextPageId || "").trim()
        }))
      : [];

    return {
      id,
      text,
      textPosition: page.textPosition === "top" ? "top" : "bottom",
      imagePrompts: imagePrompts.length > 0 ? imagePrompts.slice(0, 3) : [
        `Main story illustration for page ${index + 1}`
      ],
      choices
    };
  });

  let branchCount = 0;
  for (const page of normalizedPages) {
    if (page.choices.length === 2) branchCount += 1;

    for (const choice of page.choices) {
      if (!choice.nextPageId || !ids.has(choice.nextPageId)) {
        throw new Error(
          `Page ${page.id} has an invalid branch target: ${choice.nextPageId || "(empty)"}`
        );
      }
    }
  }

  if (branchCount < 1) {
    throw new Error("Story must contain at least one Option A / Option B branch point.");
  }

  const startPageId = String(raw.startPageId || normalizedPages[0]?.id || "p1");
  if (!ids.has(startPageId)) {
    throw new Error("Invalid startPageId in generated story.");
  }

  return {
    title: String(raw.title || "My Interactive Storybook").slice(0, 80),
    summary: String(raw.summary || "").slice(0, 200),
    startPageId,
    pages: normalizedPages
  };
}

async function submitWaveSpeedTask(endpoint, body) {
  assertApiKey();

  const response = await fetch(`${WAVESPEED_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${WAVESPEED_API_KEY}`
    },
    body: JSON.stringify(body)
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload?.message || `WaveSpeed submit failed (${response.status}).`);
  }

  const taskId = extractTaskId(payload);
  if (!taskId) {
    throw new Error("WaveSpeed did not return a task id.");
  }

  return { taskId, payload };
}

async function fetchWaveSpeedResult(taskId) {
  const urls = [
    `${WAVESPEED_BASE_URL}/predictions/${taskId}/result`,
    `${WAVESPEED_BASE_URL}/predictions/${taskId}`
  ];

  let lastError;

  for (const url of urls) {
    try {
      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${WAVESPEED_API_KEY}`
        }
      });

      if (!response.ok) {
        lastError = new Error(`WaveSpeed poll failed (${response.status})`);
        continue;
      }

      return await response.json();
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error("Unable to fetch WaveSpeed prediction result.");
}

async function waitForWaveSpeedTask(taskId, timeoutMs = 240000) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    const payload = await fetchWaveSpeedResult(taskId);
    const status = extractStatus(payload);

    if (["completed", "succeeded", "success"].includes(String(status).toLowerCase())) {
      return payload;
    }

    if (["failed", "error", "canceled"].includes(String(status).toLowerCase())) {
      const message = extractText(payload) || payload?.message || "WaveSpeed task failed.";
      throw new Error(message);
    }

    await delay(1800);
  }

  throw new Error("WaveSpeed task timed out.");
}

async function runAnyLlm(prompt) {
  const { taskId } = await submitWaveSpeedTask("/wavespeed-ai/any-llm", {
    prompt,
    model: "google/gemini-2.5-flash",
    reasoning: false,
    priority: "latency",
    temperature: 0.5,
    max_tokens: 3500,
    enable_sync_mode: false
  });

  return waitForWaveSpeedTask(taskId, 180000);
}

async function runFluxImage(prompt) {
  const { taskId } = await submitWaveSpeedTask("/wavespeed-ai/flux-dev", {
    prompt,
    size: "1024*1024",
    num_inference_steps: 28,
    guidance_scale: 3.5,
    num_images: 1,
    output_format: "jpeg",
    enable_sync_mode: false
  });

  const result = await waitForWaveSpeedTask(taskId, 240000);
  const urls = extractUrls(result);

  const imageUrl = urls.find((url) => /\.(png|jpe?g|webp)(\?|$)/i.test(url)) || urls[0];
  if (!imageUrl) {
    throw new Error("Image generation completed but no image URL was returned.");
  }

  return imageUrl;
}

async function runQwenTts(text) {
  const { taskId } = await submitWaveSpeedTask(
    "/wavespeed-ai/qwen3-tts/text-to-speech",
    {
      text,
      language: "auto",
      voice: "Vivian",
      style_instruction: "Warm and friendly, not too fast, expressive like a bedtime storyteller."
    }
  );

  const result = await waitForWaveSpeedTask(taskId, 180000);
  const urls = extractUrls(result);
  const audioUrl =
    urls.find((url) => /\.(mp3|wav|ogg|m4a)(\?|$)/i.test(url)) || urls[0];

  if (!audioUrl) {
    throw new Error("Audio generation completed but no audio URL was returned.");
  }

  return audioUrl;
}

function buildStoryPrompt({ prompt, style, pageCount, age }) {
  return `You are writing a children storybook for age ${age}.\n\nCreate a fully kid-safe interactive branching story in strict JSON.\n\nConstraints:\n- Exactly ${pageCount} pages in total.\n- Reading level suitable for age ${age}.\n- Every page text must be one short paragraph (max 40 words).\n- No scary, violent, hateful, romantic, or inappropriate content.\n- At least one branch point where there are exactly two choices: Option A and Option B.\n- Each choice must point to a valid page id in nextPageId.\n- Keep language warm, simple, friendly, and imaginative.\n- imagePrompts must be short, clear, and kid friendly with no distortion and no text in image.\n- Mostly 1 image prompt per page, occasionally 2-3 if needed (max 3).\n- textPosition must be either top or bottom.\n\nOutput JSON schema:\n{\n  "title": "string",\n  "summary": "string",\n  "startPageId": "p1",\n  "pages": [\n    {\n      "id": "p1",\n      "text": "string",\n      "textPosition": "top or bottom",\n      "imagePrompts": ["string"],\n      "choices": [\n        { "text": "string", "nextPageId": "p2" },\n        { "text": "string", "nextPageId": "p3" }\n      ]\n    }\n  ]\n}\n\nParent/child story request: ${prompt}\nVisual style: ${style}`;
}

async function generateStory(jobId, input) {
  const job = jobs.get(jobId);
  if (!job) return;

  try {
    job.status = "generating_story";
    job.progress = 8;
    job.message = "Writing story and branching choices...";

    const llmResult = await runAnyLlm(
      buildStoryPrompt({
        prompt: input.prompt,
        style: input.style,
        pageCount: input.pageCount,
        age: input.age
      })
    );

    const storyText = extractText(llmResult);
    const blueprint = normalizeStoryBlueprint(
      extractJsonFromText(storyText),
      input.pageCount
    );

    job.status = "generating_images";
    job.progress = 18;
    job.message = "Painting storybook images...";

    const pages = [];

    for (let i = 0; i < blueprint.pages.length; i += 1) {
      const page = blueprint.pages[i];
      const pageImages = [];

      const desiredImageCount =
        page.imagePrompts.length >= 3 && i % 9 === 0
          ? 3
          : page.imagePrompts.length >= 2 && i % 5 === 0
            ? 2
            : 1;

      const prompts = page.imagePrompts.slice(0, desiredImageCount);
      for (const [idx, prompt] of prompts.entries()) {
        const illustrationPrompt = [
          `Children's storybook illustration in ${input.style} style.`,
          `For kids age ${input.age}.`,
          `Scene: ${prompt}.`,
          `Bright, kind, playful, clean anatomy, non-distorted, no scary content, no text in image.`,
          `Image ${idx + 1} of ${prompts.length} for page ${i + 1}.`
        ].join(" ");

        const imageUrl = await runFluxImage(illustrationPrompt);
        pageImages.push(imageUrl);
      }

      pages.push({
        ...page,
        imageUrls: pageImages
      });

      const progressBase = 18;
      const span = 78;
      job.progress = Math.min(
        96,
        progressBase + Math.round(((i + 1) / blueprint.pages.length) * span)
      );
      job.message = `Generated page ${i + 1} of ${blueprint.pages.length}`;
    }

    job.status = "completed";
    job.progress = 100;
    job.message = "Storybook is ready!";
    job.story = {
      id: jobId,
      title: blueprint.title,
      summary: blueprint.summary,
      style: input.style,
      age: input.age,
      pageCount: input.pageCount,
      startPageId: blueprint.startPageId,
      pages,
      createdAt: new Date().toISOString()
    };
  } catch (error) {
    job.status = "failed";
    job.message = "Could not generate storybook.";
    job.error = error instanceof Error ? error.message : String(error);
  } finally {
    job.updatedAt = new Date().toISOString();
  }
}

function validateStoryRequest(body) {
  const prompt = String(body.prompt || "").trim();
  const style = String(body.style || "").trim();
  const pageCount = Number(body.pageCount);
  const age = Number(body.age);

  if (!prompt || prompt.length < 8) {
    throw new Error("Please provide a more detailed story prompt.");
  }

  if (!ALLOWED_STYLES.includes(style)) {
    throw new Error("Please choose one of the 5 supported styles.");
  }

  if (!Number.isInteger(pageCount) || pageCount < 3 || pageCount > MAX_PAGES) {
    throw new Error(`Page count must be between 3 and ${MAX_PAGES}.`);
  }

  if (!Number.isInteger(age) || age < MIN_AGE || age > MAX_AGE) {
    throw new Error(`Age must be between ${MIN_AGE} and ${MAX_AGE}.`);
  }

  return { prompt, style, pageCount, age };
}

app.get("/api/config", (_req, res) => {
  res.json({
    styles: ALLOWED_STYLES,
    minAge: MIN_AGE,
    maxAge: MAX_AGE,
    maxPages: MAX_PAGES
  });
});

app.post("/api/story/jobs", async (req, res) => {
  try {
    assertApiKey();
    const input = validateStoryRequest(req.body || {});

    const jobId = crypto.randomUUID();
    const now = new Date().toISOString();

    jobs.set(jobId, {
      id: jobId,
      status: "queued",
      progress: 0,
      message: "Queued for generation...",
      story: null,
      error: null,
      createdAt: now,
      updatedAt: now
    });

    generateStory(jobId, input);

    return res.status(202).json({
      jobId
    });
  } catch (error) {
    return res.status(400).json({
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get("/api/story/jobs/:jobId", (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: "Job not found." });
  }

  return res.json({
    id: job.id,
    status: job.status,
    progress: job.progress,
    message: job.message,
    error: job.error,
    story: job.story || null,
    updatedAt: job.updatedAt
  });
});

app.post("/api/story/audio", async (req, res) => {
  try {
    assertApiKey();

    const storyId = String(req.body.storyId || "").trim();
    const pageId = String(req.body.pageId || "").trim();
    const text = String(req.body.text || "").trim();

    if (!storyId || !pageId || !text) {
      throw new Error("storyId, pageId and text are required.");
    }

    if (containsUnsafeWords(text)) {
      throw new Error("Narration blocked due to unsafe words.");
    }

    const cacheKey = `${storyId}:${pageId}:${crypto
      .createHash("sha1")
      .update(text)
      .digest("hex")}`;

    if (audioCache.has(cacheKey)) {
      return res.json({ audioUrl: audioCache.get(cacheKey), cached: true });
    }

    const prefixed = `Narrate this for children slowly, warmly, and clearly: ${text}`;
    const audioUrl = await runQwenTts(prefixed);
    audioCache.set(cacheKey, audioUrl);

    return res.json({ audioUrl, cached: false });
  } catch (error) {
    return res.status(400).json({
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`Happy Ever running on http://localhost:${PORT}`);
});
