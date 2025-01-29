import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import requests
from io import BytesIO

def load_image(image_path, target_size=(768, 768)):
    """
    Load and preprocess the input image
    """
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # Resize and preserve aspect ratio
    image = image.convert("RGB")
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create new image with padding to reach target size
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    new_image.paste(image, ((target_size[0] - image.size[0]) // 2,
                           (target_size[1] - image.size[1]) // 2))
    
    return new_image

def generate_image_variation(
    input_image_path,
    prompt,
    model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    num_images=1,
    strength=0.75,
    guidance_scale=7.5,
    seed=None
):
    """
    Generate variations of an input image using a specified prompt
    
    Parameters:
    - input_image_path: Path or URL to the input image
    - prompt: Text prompt to guide the image generation
    - model_id: Hugging Face model ID
    - num_images: Number of variations to generate
    - strength: How much to transform the input image (0-1)
    - guidance_scale: How closely to follow the prompt
    - seed: Random seed for reproducibility
    
    Returns:
    - List of generated images
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # Load and preprocess the input image
    init_image = load_image(input_image_path)
    
    # Generate images
    result = pipe(
        prompt=prompt,
        image=init_image,
        num_images_per_prompt=num_images,
        strength=strength,
        guidance_scale=guidance_scale
    )
    
    return result.images

def save_generated_images(images, output_prefix="generated"):
    """
    Save the generated images with sequential numbering
    """
    for i, image in enumerate(images):
        image.save(f"images-out/{output_prefix}_{i}.png")

# Example usage
if __name__ == "__main__":
    # Example parameters
    input_image = "images-in/Your_image.jpg"  # or URL
    prompt = "Draw the image in modern art style, photorealistic and detailed."
    
    # Generate variations
    generated_images = generate_image_variation(
        input_image,
        prompt,
        num_images=3,
        strength=0.75,
        seed=42  # Optional: for reproducibility
    )
    
    # Save the results
    save_generated_images(generated_images)