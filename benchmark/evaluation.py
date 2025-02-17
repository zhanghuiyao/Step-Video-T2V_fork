from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed
from stepvideo.parallel import initialize_parall_group, get_parallel_group


def load_bmk_prompt(path):
    prompts = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            prompts.append(line.strip()) 
    return prompts


if __name__ == "__main__":
    args = parse_args()
    initialize_parall_group(ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree)
    
    local_rank = get_parallel_group().local_rank
    device = torch.device(f"cuda:{local_rank}")
    
    setup_seed(args.seed)
        
    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(dtype=torch.bfloat16, device=device)
    pipeline.setup_api(
        vae_url = args.vae_url,
        caption_url = args.caption_url,
    )
    
    prompts = load_bmk_prompt('benchmark/Step-Video-T2V-Eval')
    
    for prompt in prompts:
        videos = pipeline(
            prompt=prompt, 
            num_frames=args.num_frames, 
            height=args.height, 
            width=args.width,
            num_inference_steps = args.infer_steps,
            guidance_scale=args.cfg_scale,
            time_shift=args.time_shift,
            pos_magic=args.pos_magic,
            neg_magic=args.neg_magic,
            output_file_name=prompt[:50]
        )
    
    dist.destroy_process_group()