# Image Variations with Stable Diffusion

This project generates variations of an input image using the Stable Diffusion model. It leverages the `diffusers` library from Hugging Face.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Example Usage

You can generate image variations by running the script with the following example parameters:

```python
# Example usage
if __name__ == "__main__":
    # Example parameters
    input_image = "images-in/Alex.jpg"  # or URL
    prompt = "Draw this person as a powerful king, photorealistic and detailed, in a medieval setting."
    
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
```

### Functions

#### `load_image(image_path, target_size=(768, 768))`

Loads and preprocesses the input image.

#### `generate_image_variation(input_image_path, prompt, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", num_images=1, strength=0.75, guidance_scale=7.5, seed=None)`

Generates variations of an input image using a specified prompt.

#### `save_generated_images(images, output_prefix="generated")`

Saves the generated images with sequential numbering.

## Parameters

- **input_image_path**: Path or URL to the input image.
- **prompt**: Text prompt to guide the image generation.
- **model_id**: Hugging Face model ID.
- **num_images**: Number of variations to generate.
- **strength**: How much to transform the input image (0-1).
- **guidance_scale**: How closely to follow the prompt.
- **seed**: Random seed for reproducibility.

## License

This project is licensed under the MIT License.
