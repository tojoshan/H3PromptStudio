import os
import re
import json
from functools import lru_cache
from pathlib import Path

import torch
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForImageTextToText

from scene import SceneContent

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "Qwen3-VL-4B-Instruct"
MODEL_ID = str(MODEL_DIR)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SYSTEM_PROMPT = """You are the semantic scene interpreter for a MiniMax H3 video pipeline.

The user may write in Spanish or any other language.

Your job is to understand the request in its original language, but ALWAYS return
all SceneContent semantic fields in concise, natural, canonical English.

Do NOT perform a literal word-for-word translation.
Translate meaning, relationships, modifiers, actions, camera intent, style, and
continuity constraints into a clean English scene representation.

Proper names must never be translated.

STRICT RULES:
- Preserve explicit facts exactly in meaning.
- Do not invent appearance, lighting, lenses, locations, props, people, animals, actions, backstory, or events.
- Unknown information must remain empty.
- Separate subject, environment, primary action, secondary action, camera, style, lighting, preservation constraints, and exclusions.
- Explicit negatives such as "no people" belong in must_avoid.
- Do not add generic exclusions that the user did not request.
- Do not decide duration, FPS, aspect ratio, resolution, seed, video mode, or assets.
- Do not write the final MiniMax H3 prompt.
- Avoid unresolved pronouns like "él", "ella", "eso", "it", "he", "she", or "they" when the referenced entity can be named explicitly.
- Prefer explicit nouns instead of pronouns in action and secondary_action.
- If another entity participates in the same action, include that entity explicitly inside ACTION.
- Use SECONDARY_ACTION only when there is a second distinct action with its own verb.
- Descriptive fragments without a verb must NOT go into SECONDARY_ACTION.
- If a phrase describes an entity, attach it to that entity in ACTION or put the stable visual trait in PRESERVE.
- Preserve explicit adverbs and motion modifiers such as slowly, quickly, gently, cautiously, or energetically.
- Put stable visual identity traits into PRESERVE when they matter for continuity, such as color, age, breed, size, clothing, fur, markings, or other distinctive details.
- Keep ACTION short, concrete, and visually executable.
- Keep SECONDARY_ACTION short, concrete, and visually executable.
- Return ONLY one valid JSON object matching the supplied schema.
- Do not wrap the JSON in markdown fences.
"""

class SceneIntent(BaseModel):
    content: SceneContent = Field(default_factory=SceneContent)


def model_files_present() -> bool:
    required = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ]
    return MODEL_DIR.is_dir() and all((MODEL_DIR / name).exists() for name in required)


@lru_cache(maxsize=1)
def get_qwen_components():
    if not model_files_present():
        raise FileNotFoundError(
            "Missing Qwen3-VL-4B-Instruct. Copy the complete repository into "
            f"{MODEL_DIR}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot detect CUDA. Check the CUDA/PyTorch installation."
        )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    return processor, model


def is_qwen_loaded() -> bool:
    return get_qwen_components.cache_info().currsize > 0


_VERB_PATTERNS = [
    r"\b(walks?|walking|runs?|running|plays?|playing|looks?|looking|jumps?|jumping|flies?|flying|appears?|appearing|sits?|sitting|barks?|barking|meows?|meowing|chases?|chasing|observes?|observing|rests?|resting|sleeps?|sleeping|trots?|trotting|moves?|moving|turns?|turning|interacts?|interacting|stops?|stopping|approaches?|approaching|follows?|following|crosses?|crossing|enters?|entering|leaves?|leaving)\b",
]


def _has_verb(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(re.search(pattern, t) for pattern in _VERB_PATTERNS)


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip(" ,;")


def _append_unique(items: list[str], value: str):
    value = _clean(value)
    if value and value not in items:
        items.append(value)


def _normalize_content(content: SceneContent) -> SceneContent:
    content.action = _clean(content.action)
    content.secondary_action = _clean(content.secondary_action)

    # If Qwen returns a descriptive fragment as secondary_action, merge it back
    # into the main action and preserve it as a continuity descriptor.
    if content.secondary_action and not _has_verb(content.secondary_action):
        descriptor = content.secondary_action

        if content.action and not content.action.endswith(descriptor):
            content.action = _clean(f"{content.action} {descriptor}")

        _append_unique(content.must_preserve, descriptor)
        content.secondary_action = ""

    # Stable identity traits of the primary subject are continuity constraints.
    subject = content.subject
    for trait in [subject.appearance, *subject.identity_constraints]:
        _append_unique(content.must_preserve, trait)

    # Remove duplicates while preserving order.
    deduped_preserve = []
    for item in content.must_preserve:
        _append_unique(deduped_preserve, item)
    content.must_preserve = deduped_preserve

    deduped_avoid = []
    for item in content.must_avoid:
        _append_unique(deduped_avoid, item)
    content.must_avoid = deduped_avoid

    return content


def interpret_scene(text: str) -> SceneContent:
    text = (text or "").strip()
    if not text:
        raise ValueError("Write a scene description.")

    import time

    t0 = time.perf_counter()
    processor, model = get_qwen_components()
    print(f"[Qwen] model ready in {time.perf_counter() - t0:.1f}s")

    request = {
        "request": text,
        "json_schema": SceneIntent.model_json_schema(),
    }

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(request, ensure_ascii=False),
                }
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        return_tensors="pt",
    )

    target_device = next(model.parameters()).device
    inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    t1 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )
    print(f"[Qwen] generation finished in {time.perf_counter() - t1:.1f}s")

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]

    generated = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    start = generated.find("{")
    end = generated.rfind("}")

    if start < 0 or end < start:
        raise ValueError(
            f"Qwen did not return valid JSON: {generated[:500]}"
        )

    content = SceneIntent.model_validate_json(
        generated[start:end + 1]
    ).content

    return _normalize_content(content)
