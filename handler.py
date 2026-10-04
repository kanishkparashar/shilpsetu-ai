import os
import base64
from pathlib import Path

import torch
import runpod
from diffusers import FluxPipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

PROMPT_DIR = Path(__file__).parent / "prompts"

PROMPT_FILES = {
    "model_shot": "model_shot.txt",
    "lifestyle": "lifestyle.txt",
    "catalog": "catalog.txt",
    "white_background": "white_background.txt",
    "luxury": "luxury.txt",
}


# ---------------------------------------------------------
# Load master prompts
# ---------------------------------------------------------

def load_master_prompts():
    prompts = {}

    for prompt_type, filename in PROMPT_FILES.items():
        path = PROMPT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        prompts[prompt_type] = path.read_text(
            encoding="utf-8"
        ).strip()

    return prompts


MASTER_PROMPTS = load_master_prompts()


# ---------------------------------------------------------
# Load FLUX model
# ---------------------------------------------------------

print("Loading FLUX.1 Schnell...")

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()

print("FLUX.1 Schnell loaded successfully.")


# ---------------------------------------------------------
# Generate image
# ---------------------------------------------------------

def generate_image(prompt):

    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
    ).images[0]

    return image


# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------

def handler(job):

    job_input = job.get("input", {})

    prompt_type = job_input.get("prompt_type")

    if prompt_type not in MASTER_PROMPTS:
        return {
            "error": (
                "Invalid prompt_type. Choose one of: "
                + ", ".join(MASTER_PROMPTS.keys())
            )
        }

    master_prompt = MASTER_PROMPTS[prompt_type]

    # Optional additional prompt from the caller
    additional_prompt = job_input.get("prompt", "").strip()

    if additional_prompt:
        final_prompt = f"{master_prompt}\n\n{additional_prompt}"
    else:
        final_prompt = master_prompt

    print(f"Using prompt type: {prompt_type}")
    print(f"Generating image...")

    image = generate_image(final_prompt)

    # Convert image to PNG bytes
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "prompt_type": prompt_type,
        "image_base64": image_base64
    }


# ---------------------------------------------------------
# Start RunPod Serverless worker
# ---------------------------------------------------------

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })