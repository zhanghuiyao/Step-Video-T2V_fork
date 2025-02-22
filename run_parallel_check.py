import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch.distributed as dist
import torch
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed
from stepvideo.parallel import initialize_parall_group, get_parallel_group
from stepvideo.diffusion.video_pipeline import StepVideoPipelineOutput



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
    
    

    @torch.inference_mode()
    def new_call_fn(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        time_shift: float = 13.0,
        neg_magic: str = "",
        pos_magic: str = "",
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "mp4",
        output_file_name: Optional[str] = "",
        return_dict: bool = True,
    ):
        
        # 1. Check inputs. Raise error if not correct
        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_2, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            neg_magic=neg_magic,
            pos_magic=pos_magic,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        prompt_embeds_2 = prompt_embeds_2.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            time_shift=time_shift,
            device=device
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.bfloat16,
            device,
            generator,
            latents,
        )

        ################# 1 #############################################################
        np.save(f"latents_rank{torch.distributed.get_rank()}.npy", latents.detach().cpu().to(torch.float32).numpy())
        np.save(f"prompt_embeds_rank{torch.distributed.get_rank()}.npy", prompt_embeds.detach().cpu().to(torch.float32).numpy())
        np.save(f"prompt_attention_mask_rank{torch.distributed.get_rank()}.npy", prompt_attention_mask.detach().cpu().to(torch.float32).numpy())
        np.save(f"prompt_embeds_2_rank{torch.distributed.get_rank()}.npy", prompt_embeds_2.detach().cpu().to(torch.float32).numpy())
        print("self.transformer input save success.")
        ################################################################################

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(transformer_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_hidden_states_2=prompt_embeds_2,
                    return_dict=False,
                )

                ################# 2 #############################################################
                np.save(f"noise_pred_step{i}_rank{torch.distributed.get_rank()}.npy", noise_pred.detach().cpu().to(torch.float32).numpy())
                print("noise_pred save success.")
                ################################################################################

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents
                )
                
                ################# 3 #############################################################
                np.save(f"latents_step{i}_rank{torch.distributed.get_rank()}.npy", latents.detach().cpu().to(torch.float32).numpy())
                print("latents save success.")
                ################################################################################

                progress_bar.update()

        if not torch.distributed.is_initialized() or int(torch.distributed.get_rank())==0:
            if not output_type == "latent":
                video = self.decode_vae(latents)

                ################# 4 #############################################################
                np.save(f"video_rank{torch.distributed.get_rank()}.npy", video.detach().cpu().to(torch.float32).numpy())
                print("video_tensor save success.")
                ################################################################################

                video = self.video_processor.postprocess_video(video, output_file_name=output_file_name, output_type=output_type)
            else:
                video = latents

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (video, )

            return StepVideoPipelineOutput(video=video)
        

    import types
    pipeline.new_call = types.MethodType(new_call_fn, pipeline)

    prompt = args.prompt
    videos = pipeline.new_call(
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