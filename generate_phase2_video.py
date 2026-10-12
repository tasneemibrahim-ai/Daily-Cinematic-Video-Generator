"""
Compact Phase 2 Video Generation Script.
Daily Cinematic Video Generator: Human Motion in Dynamic Environments.

Pipeline:
1. Stable Diffusion XL (SDXL) Base 1.0 for high-resolution text-to-image synthesis.
2. Stable Video Diffusion (SVD-XT) for temporal latent video animation.
3. Temporal smoothing and export.
"""

from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

PROMPT = (
    "A person walking in a busy street with moving cars and background elements, "
    "daytime, dynamic motion, cinematic tracking shot, coherent human anatomy, "
    "consistent face and clothes across frames, natural walking gait, stable camera motion"
)
NEGATIVE_PROMPT = (
    "flicker, jitter, distorted body, deformed hands, extra limbs, duplicate person, "
    "warped face, unstable background, text, watermark, low quality, blurry"
)


def main(output_dir: str = "phase2_final_video", seed: int = 4096):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if device == "cuda":
        kwargs["variant"] = "fp16"
    
    print(f"[*] Initializing SDXL text-to-image pipeline on {device} ({dtype})...")
    t2i = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", **kwargs
    )
    
    i2v_kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        i2v_kwargs["variant"] = "fp16"
    
    print(f"[*] Initializing SVD-XT image-to-video pipeline on {device} ({dtype})...")
    i2v = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", **i2v_kwargs
    )
    
    if device == "cuda":
        t2i.enable_model_cpu_offload()
        i2v.enable_model_cpu_offload()
    else:
        t2i.to(device)
        i2v.to(device)
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    print("[*] Generating cinematic keyframe...")
    image = t2i(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=576,
        width=1024,
        num_inference_steps=30,
        guidance_scale=10.0,
        generator=generator,
    ).images[0]
    
    keyframe_path = out_path / "phase2_initial_image.png"
    image.save(keyframe_path)
    print(f"[+] Saved keyframe to: {keyframe_path}")
    
    print("[*] Generating video frames via SVD-XT (motion_bucket_id=80, noise_aug=0.04)...")
    frames = i2v(
        image,
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=80,
        noise_aug_strength=0.04,
        num_inference_steps=25,
    ).frames[0]
    
    video_path = out_path / "phase2_final_video.mp4"
    export_to_video(frames, str(video_path), fps=6)
    print(f"[+] Successfully exported video ({len(frames)} frames) to: {video_path}")


if __name__ == "__main__":
    main()
