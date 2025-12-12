
from diffusers import AutoencoderKL,  UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
import torch 
import torch.nn as nn
import os 
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import io
from torchvision import transforms
from train_loop.utils.visualization import show_tensor

class TextImageDataset(Dataset):
    def __init__(self, dataset_name, text_prepend="" , split="train", image_col="image", text_col="text"):
        """
        Load a dataset from Hugging Face with image and text columns.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            split: Dataset split to use (default: "train")
            image_col: Name of the image column (default: "image")
            text_col: Name of the text column (default: "text")
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.image_col = image_col
        self.text_col = text_col
        self.text_prepend = text_prepend

        self.train_transforms = transforms.Compose(
                [
                    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )


    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample[self.image_col]
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image)) if isinstance(image, bytes) else image 
        image = self.train_transforms(image)
        
        text = sample[self.text_col]
        
        return {
            "image": image,
            "text": self.text_prepend + text
        }

class StableDiffusionModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, lora_rank=4 ):
        super().__init__()

        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer" 
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder" 
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae" 
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet" 
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        for param in self.unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        self.vae_dtype = torch.float16
        self.unet_dtype = torch.float16
        self.text_encoder_dtype = torch.float16

        self.unet.to(self.unet_dtype)
        self.vae.to(self.vae_dtype)
        self.text_encoder.to(self.text_encoder_dtype)

        self.unet.add_adapter(unet_lora_config)
        for param in self.unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

        self.lora_params_dict = {}
        for name, param in self.unet.named_parameters():
            if param.requires_grad:
                self.lora_params_dict[name] = param
        self.lora_params = nn.ParameterList(self.lora_params_dict.values())
        self.pipeline = None

        

    def forward(self, batch):
        image = batch["image"]

        latents = self.vae.encode(image.to(dtype=self.vae_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # print("noisy_latents shape:", noisy_latents.shape )

        if "input_ids" in batch:
            input_ids = batch["input_ids"]
        else:
            texts = batch["text"]
            input_ids = self.tokenizer(
                texts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(latents.device)

        encoder_hidden_states = self.text_encoder(input_ids)[0]
        # print("encoder_hidden_states shape:", encoder_hidden_states.shape )

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return {
            "predicted_noise": model_pred,
            "actual_noise": noise,
            "target_noise": target,
            "timesteps": timesteps,
        }
    
    def state_dict(self, *args, **kwargs):
        return self.lora_params.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        return self.lora_params.load_state_dict(state_dict, strict=strict)
    
    def generate_image_with_pipe(self, prompt, num_inference_steps=25):
        
        if self.pipeline is None:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                                self.pretrained_model_name_or_path,
                                torch_dtype=self.unet_dtype,
                                unet=self.unet,
                                text_encoder=self.text_encoder,
                                vae=self.vae
                            )
            self.pipeline = self.pipeline.to(self.unet.device)

        generator = torch.Generator(device=self.unet.device)
        generator.manual_seed(1)
        im = self.pipeline(prompt, num_inference_steps=num_inference_steps, generator=generator).images[0]
        return im

    def generate_image(self, prompt, num_inference_steps=25, guidance_scale=7.5, negative_prompt="", seed=1):

        device = self.unet.device
        use_cfg = guidance_scale > 1.0
        
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None
        
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
        scheduler.set_timesteps(num_inference_steps)
        
        latents_shape = (1, self.unet.config.in_channels, 512 // 8, 512 // 8)  # 64x64 latents for 512x512 image
        
        if generator is not None:
            latents = torch.randn(latents_shape, generator=generator, device=device, dtype=self.unet_dtype)
        else:
            latents = torch.randn(latents_shape, device=device, dtype=self.unet_dtype)
        
        latents = latents * scheduler.init_noise_sigma
        
        prompt_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        
        prompt_embeds = self.text_encoder(prompt_input_ids)[0].to(dtype=self.text_encoder_dtype)
        
        if use_cfg:
            negative_input_ids = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            
            negative_embeds = self.text_encoder(negative_input_ids)[0].to(dtype=self.text_encoder_dtype)
        
        with torch.no_grad():
            for t in scheduler.timesteps:
                latent_model_input = scheduler.scale_model_input(latents, timestep=t)
                
                if use_cfg:
                    noise_pred_uncond = self.unet(
                        latent_model_input.to(self.unet_dtype), 
                        t, 
                        negative_embeds
                    ).sample
                    
                    noise_pred_text = self.unet(
                        latent_model_input.to(self.unet_dtype), 
                        t, 
                        prompt_embeds
                    ).sample
                    
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = self.unet(
                        latent_model_input.to(self.unet_dtype), 
                        t, 
                        prompt_embeds
                    ).sample
                
                latents = scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
            
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents.to(self.vae_dtype)).sample
        
        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        
        return Image.fromarray(image[0])



class GenerateBasicPreview:
    def __init__(self, show_imgcat=False):
        self.show_imgcat = show_imgcat
        pass 


    def run(self, model, dataset, out_dir, iter):
        

        prompts = [
            "sks style taj mahal",
            "sks style a beautiful landscape with mountains and a river",
            "sks style a futuristic city with flying cars",
            "sks style a serene beach at sunset",
        ]
        imgs = []
        for prompt in prompts:
            im = model.generate_image(prompt )
            imgs.append(im)
        img_cat = Image.new('RGB', (imgs[0].width * len(imgs), imgs[0].height))
        for i, img in enumerate(imgs):
            img_cat.paste(img, (i * imgs[0].width, 0))
        out_path = os.path.join(out_dir, f"preview_{iter:08d}.png")
        img_cat.save(out_path)
        if self.show_imgcat:
            show_tensor(img_cat)