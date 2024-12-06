import torch
from diffusers import StableDiffusionImg2ImgPipeline
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import os
from PIL import Image

class StableDiffusionFineTuner:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-base", 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_id = model_id
        self.pipeline = None
        self.setup_model()
        
    def setup_model(self):
        """Initialize model components"""
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
    def prepare_data(self, dataset_name="cifar10", batch_size=4):
        """Prepare dataset for training"""
        dataset = load_dataset(dataset_name)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        def transform_images(examples):
            examples["source_images"] = [
                transform(image.convert("RGB")) 
                for image in examples["img"]
            ]
            examples["target_images"] = [
                transform(image.convert("RGB")) 
                for image in examples["img"]
            ]
            return examples
            
        dataset.set_transform(transform_images)
        return DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
        
    def train(self, train_dataloader, num_epochs=10, learning_rate=1e-5):
        """Fine-tune the model"""
        optimizer = torch.optim.AdamW(self.pipeline.unet.parameters(), lr=learning_rate)
        wandb.init(project="stable-diffusion-finetuning")
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                # Get images
                source_images = batch["source_images"].to(self.device)
                target_images = batch["target_images"].to(self.device)
                
                # Sample noise and add to images
                noise = torch.randn_like(target_images)
                timesteps = torch.randint(
                    0, self.pipeline.scheduler.num_train_timesteps, (target_images.shape[0],)
                ).long().to(self.device)
                noisy_images = self.pipeline.scheduler.add_noise(target_images, noise, timesteps)
                
                # Get model prediction
                noise_pred = self.pipeline.unet(noisy_images, timesteps).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                total_loss += loss.item()
                
                # Backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(train_dataloader)
            wandb.log({"epoch": epoch, "loss": avg_loss})
            print(f"Epoch {epoch}: Average Loss = {avg_loss}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}")
                
    def save_checkpoint(self, checkpoint_dir):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.pipeline.save_pretrained(checkpoint_dir)
        
    def generate_sample(self, input_image, strength=0.75, guidance_scale=7.5):
        """Generate sample images using fine-tuned model"""
        images = self.pipeline(
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale
        ).images
        return images

def main():
    # Initialize fine-tuner
    fine_tuner = StableDiffusionFineTuner()
    
    # Prepare data
    train_dataloader = fine_tuner.prepare_data()
    
    # Train model
    fine_tuner.train(train_dataloader)
    
    # Generate test samples
    input_image = Image.open("path_to_input_image.jpg")
    samples = fine_tuner.generate_sample(input_image)
    
    # Save samples
    for i, image in enumerate(samples):
        image.save(f"sample_{i}.png")

if __name__ == "__main__":
    main()