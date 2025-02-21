import numpy as np

import torch

from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed
from stepvideo.parallel import initialize_parall_group


# for test
import numpy as np
from typing import Optional, Union, List


if __name__ == "__main__":
    args = parse_args()
    
    # initialize_parall_group(args, ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree)
    
    device = torch.device("cuda:0")

    setup_seed(args.seed)
    
    import time
    print("building pipeline...")
    _t = time.time()
    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(torch.bfloat16)
    print(f"build pipeline success, time cost: {(time.time()-_t)/60:.2f} min")

    pipeline.setup_api(
        vae_url = args.vae_url,
        caption_url = args.caption_url,
    )

    def new_call_fn(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        num_inference_steps: int = 5,
        guidance_scale: float = 9.0,
        time_shift: float = 13.0,
        neg_magic: str = "",
        pos_magic: str = "",
        num_videos_per_prompt: Optional[int] = 1,
        # TODO: add numpy generator
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "mp4",
        output_file_name: Optional[str] = "",
        return_dict: bool = True,
    ):
        # 1. Check inputs. Raise error if not correct

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # prompt_embeds, prompt_embeds_2, prompt_attention_mask = self.encode_prompt(
        #     prompt=prompt,
        #     neg_magic=neg_magic,
        #     pos_magic=pos_magic,
        # )
        # transformer_dtype = self.transformer.dtype
        # prompt_embeds = prompt_embeds.to(transformer_dtype)
        # prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        # prompt_embeds_2 = prompt_embeds_2.to(transformer_dtype)
        # print("="* 100 + "\n" + f"Step1. get encode_prompt from server success.")
        # print(f"{prompt_embeds.shape=}")
        # print(f"{prompt_embeds_2.shape=}")
        # print(f"{prompt_attention_mask.shape=}")

        transformer_dtype = self.transformer.dtype
        prompt_embeds = torch.Tensor(np.random.randn(2, 320, 6144), transformer_dtype)
        prompt_attention_mask = torch.Tensor(np.random.randn(2, 397), transformer_dtype)
        prompt_embeds_2 = torch.Tensor(np.random.randn(2, 77, 1024), transformer_dtype)


        # 4. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            time_shift=time_shift
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
            latents,
        )
        print("="* 100 + "\n" + f"Step2. get latent success.")
        print(f"{latents.shape=}")

        # 7. Denoising loop
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        #     for i, t in enumerate(self.scheduler.timesteps):
        #         latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        #         latent_model_input = latent_model_input.to(transformer_dtype)
        #         # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                
        #         timestep = t.broadcast_to((latent_model_input.shape[0],)).to(latent_model_input.dtype)

        #         noise_pred = self.transformer(
        #             hidden_states=latent_model_input,
        #             timestep=timestep,
        #             encoder_hidden_states=prompt_embeds,
        #             encoder_attention_mask=prompt_attention_mask,
        #             encoder_hidden_states_2=prompt_embeds_2,
        #             return_dict=False,
        #         )
        #         # perform guidance
        #         if do_classifier_free_guidance:
        #             noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        #             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        #         # compute the previous noisy sample x_t -> x_t-1
        #         latents = self.scheduler.step(
        #             model_output=noise_pred,
        #             timestep=t,
        #             sample=latents
        #         )
                
        #         progress_bar.update()


        # 7.1 run onece
        t = self.scheduler.timesteps[0]
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = latent_model_input.to(transformer_dtype)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.broadcast_to((latent_model_input.shape[0],)).to(latent_model_input.dtype)
        
        # numpy save
        np.save("./latent_model_input.npy", latent_model_input.detach().cpu().to(torch.float32).numpy())
        np.save("./timestep.npy", timestep.detach().cpu().to(torch.float32).numpy())
        np.save("./prompt_embeds.npy", prompt_embeds.detach().cpu().to(torch.float32).numpy())
        np.save("./prompt_attention_mask.npy", prompt_attention_mask.detach().cpu().to(torch.float32).numpy())
        np.save("./prompt_embeds_2.npy", prompt_embeds_2.detach().cpu().to(torch.float32).numpy())
        print(f"tensors original dtype is: {[x.dtype for x in (latent_model_input, timestep, prompt_embeds, prompt_attention_mask, prompt_embeds_2)]}")
        return

        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            encoder_hidden_states_2=prompt_embeds_2,
            return_dict=False,
        )
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


        # video = self.decode_vae(latents)
        # print("="* 100 + "\n" + f"Step3. get video from vae server success.")
        # print(f"{video.shape=}")

        # video = self.video_processor.postprocess_video(video, output_file_name=output_file_name, output_type=output_type)
        # print("="* 100 + "\n" + f"Step4. save video success.")
        # print(f"{video.shape=}")


    import types
    pipeline.new_call = types.MethodType(new_call_fn, pipeline)


    args.infer_steps = 5  # for test
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
