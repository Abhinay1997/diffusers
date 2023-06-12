from diffusers import DiffusionPipeline, DDIMInverseScheduler, UNet2DModel, UNet2DConditionModel, UnCLIPScheduler
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from diffusers.pipelines.unclip import UnCLIPTextProjModel
from diffusers.utils import logging, randn_tensor, is_accelerate_available, is_accelerate_version
from typing import List, Union, Optional, Tuple
from torch.nn import functional as F

import torch
unclip_checkpoint = "kakaobrain/karlo-v1-alpha"
unclip_image_checkpoint = "kakaobrain/karlo-v1-alpha-image-variations"

def slerp(val, low, high):
    """
    Find the interpolation point between the 'low' and 'high' values for the given 'val'. See https://en.wikipedia.org/wiki/Slerp for more details on the topic.
    """
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos((low_norm * high_norm))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res

class UnCLIPTextDiffInterpolationPipeline(DiffusionPipeline):

    decoder: UNet2DConditionModel
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    image_encoder: CLIPVisionModelWithProjection
    feature_extractor: CLIPImageProcessor
    tokenizer: CLIPTokenizer
    super_res_first: UNet2DModel
    super_res_last: UNet2DModel

    decoder_scheduler: UnCLIPScheduler
    inverse_scheduler: DDIMInverseScheduler
    super_res_scheduler: UnCLIPScheduler

    def __init__(
        self,
        decoder: UNet2DConditionModel,
        text_encoder: CLIPTextModelWithProjection,
        image_encoder: CLIPVisionModelWithProjection,
        feature_extractor: CLIPImageProcessor,
        tokenizer: CLIPTokenizer,
        text_proj: UnCLIPTextProjModel,
        super_res_first: UNet2DModel,
        super_res_last: UNet2DModel,
        inverse_scheduler: DDIMInverseScheduler,
        decoder_scheduler: UnCLIPScheduler,
        super_res_scheduler: UnCLIPScheduler,
    ):  
        super().__init__()

        self.register_modules(
            decoder=decoder,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            text_proj=text_proj,
            super_res_first=super_res_first,
            super_res_last=super_res_last,
            inverse_scheduler=inverse_scheduler,
            decoder_scheduler=decoder_scheduler,
            super_res_scheduler=super_res_scheduler
        )
    
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_image(self, image, device, num_images_per_prompt, image_embeddings: Optional[torch.Tensor] = None):
        dtype = next(self.image_encoder.parameters()).dtype

        if image_embeddings is None:
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeddings = self.image_encoder(image).image_embeds

        image_embeddings = image_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        return image_embeddings
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        if text_model_output is None:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_mask = text_inputs.attention_mask.bool().to(device)

            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

            text_encoder_output = self.text_encoder(text_input_ids.to(device))

            prompt_embeds = text_encoder_output.text_embeds
            text_encoder_hidden_states = text_encoder_output.last_hidden_state

        else:
            batch_size = text_model_output[0].shape[0]
            prompt_embeds, text_encoder_hidden_states = text_model_output[0], text_model_output[1]
            text_mask = text_attention_mask

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, text_encoder_hidden_states, text_mask
        

    def __call__(self, image, base_prompt, target_prompt, num_steps):
        image_embeds = image_encoder.preprocess(image)
        
        print("Generating image embeddings completed")
        base_prompt_embeds = encode_prompt(base_prompt)
        target_prompt_embeds = encode_prompt(target_prompt)

        text_diff = norm_diff(base_prompt_embeds, target_prompt_embeds)
        
        for val in linspace(0.25, 0.50, num_steps):
            image
    
    def invert(self, image_embeds):
        decoder_latents = prepare_latents(image_embeds)

        original_noise = decoder.decode(decoder_latents)
    
    # Copied from unclip_pipeline.__call__
    def decode(self, prompt, image_embeddings, do_classifier_free_guidance, decoder_num_inference_steps = 30, invert = False):
        
        device = self._execution_device
        generator = None
        decoder_guidance_scale = 7.5

        if invert:
            scheduler = self.inverse_scheduler
        else:
            scheduler = self.decoder_scheduler
            
        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeddings,
            prompt_embeds=prompt_embeds,
            text_encoder_hidden_states=text_encoder_hidden_states,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        
        if self.device.type == "mps":
            # HACK: MPS: There is a panic when padding bool tensors,
            # so cast to int tensor for the pad and back to bool afterwards
            text_mask = text_mask.type(torch.int)
            decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=1)
            decoder_text_mask = decoder_text_mask.type(torch.bool)
        else:
            decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=True)

        scheduler.set_timesteps(decoder_num_inference_steps, device=device)
        decoder_timesteps_tensor = scheduler.timesteps
        print(decoder_timesteps_tensor)

        num_channels_latents = self.decoder.config.in_channels
        height = self.decoder.config.sample_size
        width = self.decoder.config.sample_size
        
        #
        batch_size = 1
        decoder_latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            text_encoder_hidden_states.dtype,
            device,
            generator,
            None,
            self.decoder_scheduler,
        )
        
        for i, t in enumerate(self.progress_bar(decoder_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([decoder_latents] * 2) if do_classifier_free_guidance else decoder_latents

            noise_pred = self.decoder(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask,
            ).sample
            print(noise_pred.shape)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(latent_model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(latent_model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + decoder_guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            if i + 1 == decoder_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = decoder_timesteps_tensor[i + 1]

            # compute the previous noisy sample x_t -> x_t-1
            print(f"{noise_pred.shape} and {decoder_latents.shape}")
            if isinstance(scheduler,DDIMInverseScheduler):
                noise_pred, _ = torch.split(noise_pred, decoder_latents.shape[1], dim=1)
                decoder_latents = scheduler.step(
                    noise_pred, t, decoder_latents
                ).prev_sample
            else:
                decoder_latents = scheduler.step(
                    noise_pred, t, decoder_latents, prev_timestep=prev_timestep, generator=generator
                ).prev_sample

        decoder_latents = decoder_latents.clamp(-1, 1)

        image_small = decoder_latents
        
        return image_small
    
    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)
        
        models = [
            self.decoder,
            self.text_proj,
            self.text_encoder,
            self.image_encoder,
            self.super_res_first,
            self.super_res_last,
        ]
        for cpu_offloaded_model in models:
            cpu_offload(cpu_offloaded_model, device)
            
    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.decoder, "_hf_hook"):
            return self.device
        for module in self.decoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device
