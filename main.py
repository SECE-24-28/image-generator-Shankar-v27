import os
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# Create result folder
# =========================
os.makedirs("result", exist_ok=True)

# =========================
# Check Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing Device: {device}")

# =========================
# Load Stable Diffusion Model
# =========================
print("\nLoading Stable Diffusion Model...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

print("Model Loaded Successfully!\n")

# =========================
# User Prompt
# =========================
prompt = input("Enter image description: ")

# =========================
# Generate Image
# =========================
print("\nGenerating Image...")

if device == "cuda":
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
else:
    image = pipe(prompt).images[0]

print("Image Generated Successfully!")

# =========================
# Show Image
# =========================
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")
plt.show()

# =========================
# Save Image
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

filename = f"result/generated_{timestamp}.png"

image.save(filename)

print(f"\nImage Saved Successfully!")
print(f"Saved at: {filename}")