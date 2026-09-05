from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from urllib.parse import quote
import httpx
import random
import asyncio
import copy
import os
import secrets

app = FastAPI(
    title="Maudy AI Image API",
    description="AI Image Generator menggunakan Gemma + FLUX.2 Klein",
    version="2.0.0"
)

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://ollama-api-vo6ttxy3vu51hlhih4ld4p82:11434"
OLLAMA_MODEL = "gemma4:e2b"

# Internal Docker URL - digunakan FastAPI untuk komunikasi dengan ComfyUI
COMFYUI_URL = "http://comfyui-m3eyisjgrhmjvcuzqt0ea6t0:8188"

# Public URL - digunakan browser untuk mengambil hasil gambar
COMFYUI_PUBLIC_URL = "https://comfyui.maudynetwork.id"

# Maksimum waktu menunggu FLUX selesai
COMFY_TIMEOUT = 300

# API
AI_IMAGE_API_KEY = os.getenv("AI_IMAGE_API_KEY")

# ============================================================
# GENERATION QUEUE / GPU LOCK
# ============================================================

generation_lock = asyncio.Lock()

# ============================================================
# REQUEST MODEL
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str


# ============================================================
# COMFYUI WORKFLOW
# ============================================================

WORKFLOW = {
    "9": {
        "inputs": {
            "filename_prefix": "Flux2-Klein",
            "images": ["75:65", 0]
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"}
    },

    "76": {
        "inputs": {
            "value": ""
        },
        "class_type": "PrimitiveStringMultiline",
        "_meta": {"title": "Prompt"}
    },

    "75:61": {
        "inputs": {
            "sampler_name": "euler"
        },
        "class_type": "KSamplerSelect",
        "_meta": {"title": "KSamplerSelect"}
    },

    "75:62": {
        "inputs": {
            "steps": 20,
            "width": ["75:68", 0],
            "height": ["75:69", 0]
        },
        "class_type": "Flux2Scheduler",
        "_meta": {"title": "Flux2Scheduler"}
    },

    "75:63": {
        "inputs": {
            "cfg": 5,
            "model": ["75:70", 0],
            "positive": ["75:74", 0],
            "negative": ["75:67", 0]
        },
        "class_type": "CFGGuider",
        "_meta": {"title": "CFG Guider"}
    },

    "75:64": {
        "inputs": {
            "noise": ["75:73", 0],
            "guider": ["75:63", 0],
            "sampler": ["75:61", 0],
            "sigmas": ["75:62", 0],
            "latent_image": ["75:66", 0]
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "SamplerCustomAdvanced"}
    },

    "75:65": {
        "inputs": {
            "samples": ["75:64", 0],
            "vae": ["75:72", 0]
        },
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"}
    },

    "75:66": {
        "inputs": {
            "width": ["75:68", 0],
            "height": ["75:69", 0],
            "batch_size": 1
        },
        "class_type": "EmptyFlux2LatentImage",
        "_meta": {"title": "Empty Flux 2 Latent"}
    },

    "75:67": {
        "inputs": {
            "text": "",
            "clip": ["75:71", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
    },

    "75:68": {
        "inputs": {
            "value": 1024
        },
        "class_type": "PrimitiveInt",
        "_meta": {"title": "Width"}
    },

    "75:69": {
        "inputs": {
            "value": 1024
        },
        "class_type": "PrimitiveInt",
        "_meta": {"title": "Height"}
    },

    "75:73": {
        "inputs": {
            "noise_seed": 1
        },
        "class_type": "RandomNoise",
        "_meta": {"title": "RandomNoise"}
    },

    "75:70": {
        "inputs": {
            "unet_name": "flux-2-klein-4b.safetensors",
            "weight_dtype": "default"
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"}
    },

    "75:71": {
        "inputs": {
            "clip_name": "qwen_3_4b.safetensors",
            "type": "flux2",
            "device": "default"
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"}
    },

    "75:72": {
        "inputs": {
            "vae_name": "flux2-vae.safetensors"
        },
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"}
    },

    "75:74": {
        "inputs": {
            "text": ["76", 0],
            "clip": ["75:71", 0]
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
    }
}


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Maudy AI Image API",
        "version": "2.0.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ollama_model": OLLAMA_MODEL,
        "flux_model": "flux-2-klein-4b.safetensors",
        "resolution": "1024x1024",
        "api_key_configured": bool(AI_IMAGE_API_KEY)
    }


# ============================================================
# GENERATE
# ============================================================

@app.post("/generate")
async def generate(
    request: GenerateRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if not AI_IMAGE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API key belum dikonfigurasi pada server."
        )

    if (
        x_api_key is None
        or not secrets.compare_digest(x_api_key, AI_IMAGE_API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt tidak boleh kosong."
        )

    # Hanya satu proses Gemma -> FLUX yang boleh memakai GPU pada satu waktu.
    async with generation_lock:
        return await process_generation(request)


async def process_generation(request: GenerateRequest):
    # --------------------------------------------------------
    # STEP 1
    # GEMMA PROMPT ENHANCEMENT
    # --------------------------------------------------------

    enhancement_instruction = f"""
Convert this Indonesian image request into one concise English prompt
for FLUX image generation.

Rules:
- Preserve the user's original intent.
- Add relevant visual details.
- Add appropriate lighting.
- Add composition and camera perspective.
- Add environment and atmosphere when relevant.
- Mention materials and textures when relevant.
- Keep the result concise and effective.
- Maximum 100 words.
- Do not explain anything.
- Do not add headings.
- Do not use bullet points.
- Output only the final English image prompt.

Request:
{request.prompt}
"""

    try:

        async with httpx.AsyncClient(timeout=300) as client:

            ollama_response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": enhancement_instruction,
                    "stream": False,

                    # Penting:
                    # unload Gemma setelah selesai
                    "keep_alive": 0
                }
            )

            ollama_response.raise_for_status()

            ollama_result = ollama_response.json()

            enhanced_prompt = (
                ollama_result
                .get("response", "")
                .strip()
            )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Gemma timeout."
        )

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Gemma error: {str(e)}"
        )

    if not enhanced_prompt:
        raise HTTPException(
            status_code=500,
            detail="Gemma tidak menghasilkan prompt."
        )

    # --------------------------------------------------------
    # STEP 2
    # BERI WAKTU OLLAMA MELEPAS VRAM
    # --------------------------------------------------------

    await asyncio.sleep(2)

    # --------------------------------------------------------
    # STEP 3
    # PREPARE FLUX WORKFLOW
    # --------------------------------------------------------

    workflow = copy.deepcopy(WORKFLOW)

    seed = random.randint(
        1,
        1125899906842624
    )

    workflow["76"]["inputs"]["value"] = enhanced_prompt

    workflow["75:73"]["inputs"]["noise_seed"] = seed

    # --------------------------------------------------------
    # STEP 4
    # SUBMIT WORKFLOW KE COMFYUI
    # --------------------------------------------------------

    try:

        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True
        ) as client:

            comfy_response = await client.post(
                f"{COMFYUI_URL}/prompt",
                json={
                    "prompt": workflow
                }
            )

            comfy_response.raise_for_status()

            comfy_result = comfy_response.json()

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI submit error: {str(e)}"
        )

    prompt_id = comfy_result.get("prompt_id")

    if not prompt_id:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "ComfyUI tidak memberikan prompt_id.",
                "response": comfy_result
            }
        )

    # --------------------------------------------------------
    # STEP 5
    # WAIT FOR COMFYUI
    # --------------------------------------------------------

    elapsed = 0

    while elapsed < COMFY_TIMEOUT:

        await asyncio.sleep(2)

        elapsed += 2

        try:

            async with httpx.AsyncClient(
                timeout=30,
                follow_redirects=True
            ) as client:

                history_response = await client.get(
                    f"{COMFYUI_URL}/history/{prompt_id}"
                )

                history_response.raise_for_status()

                history = history_response.json()

        except Exception:
            continue

        if prompt_id not in history:
            continue

        job = history[prompt_id]

        outputs = job.get("outputs", {})

        # SaveImage adalah node 9
        save_output = outputs.get("9", {})

        images = save_output.get("images", [])

        if not images:
            continue

        image = images[0]

        filename = image.get("filename", "")
        subfolder = image.get("subfolder", "")
        image_type = image.get("type", "output")

        # ----------------------------------------------------
        # STEP 6
        # BUILD IMAGE URL
        # ----------------------------------------------------

        image_url = (
    f"{COMFYUI_PUBLIC_URL}/view"
    f"?filename={quote(filename)}"
    f"&subfolder={quote(subfolder)}"
    f"&type={quote(image_type)}"
)

        return {
            "status": "success",
            "original_prompt": request.prompt,
            "enhanced_prompt": enhanced_prompt,
            "model_prompt": OLLAMA_MODEL,
            "model_image": "flux-2-klein-4b.safetensors",
            "width": 1024,
            "height": 1024,
            "seed": seed,
            "prompt_id": prompt_id,
            "filename": filename,
            "image_url": image_url
        }

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    raise HTTPException(
        status_code=504,
        detail={
            "message": "FLUX generation timeout.",
            "prompt_id": prompt_id
        }
    )
