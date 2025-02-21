import numpy as np

import torch

from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed
from stepvideo.parallel import initialize_parall_group, get_parallel_group



# for test
import os
from api.call_remote_server import CaptionPipeline, StepVaePipeline


if __name__ == "__main__":

    args = parse_args()
    # initialize_parall_group(ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree)
    
    # local_rank = get_parallel_group().local_rank
    device = torch.device("cuda:0")
    
    setup_seed(args.seed)
        
    # pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(dtype=torch.bfloat16, device=device)
    # pipeline.setup_api(
    #     vae_url = args.vae_url,
    #     caption_url = args.caption_url,
    # )


    vae_pipeline = StepVaePipeline(
        vae_dir=os.path.join(args.model_dir, "vae")
    )

    def decode_vae(samples: torch.Tensor):
        # () -> (b, 128, 128, 16)
        # samples = np.random.randn(2, 128, 128, 16)

        print(f"decode_vae input shape: {samples.shape}")

        samples = vae_pipeline.decode(samples)

        print(f"decode_vae output shape: {samples.shape}")

        return samples

    
    # latent = np.random.randn(1, 36, 64, 34, 62)
    # np.save("results/vae_input_numpy.npy", latent)
    # print(f"save success, results/vae_input_numpy.npy, {latent.shape=}")
    latent = np.load("results/vae_input_numpy.npy")

    x = torch.Tensor(latent)
    out = decode_vae(x)

    np.save("results/vae_output_numpy.npy", out.detach().to(torch.float32).cpu().numpy())
    print(f"save success, results/test_video_numpy.npy, {out.shape=}")
    
    # save video
    from stepvideo.utils.video_process import VideoProcessor
    video_processor = VideoProcessor("./results", "")
    video_processor.postprocess_video(out, output_file_name="test_video", output_type="mp4")

    print(f"svae test_video success.")
