from image_generator_comfyui import ComfyUIImageGenerator
from dotenv import load_dotenv
import os

load_dotenv()

print("="*60)
print("TESTING IMAGE GENERATION WITH COMFYUI")
print("="*60)

# Test with your existing story
story_file = 'stories/story_20251019_184004.json'

print(f"\nGenerating images for: {story_file}")
print("This will take about 10-15 minutes for all 10 pages")
print("You'll see progress for each page...\n")

# Create generator instance
generator = ComfyUIImageGenerator(os.getenv('COMFYUI_URL'))

# Generate all images
try:
    images = generator.generate_all_story_images(
        story_file,
        os.getenv('COMFYUI_OUTPUT_FOLDER'),
        os.getenv('BASE_MODEL')
    )
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    
    successful = sum(1 for i in images if i is not None)
    print(f"\nGenerated {successful}/{len(images)} images")
    print(f"Check folder: images/story_20251019_184004/")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Is ComfyUI still running in the pod?")
    print("2. Is the COMFYUI_URL correct in .env?")
    print("3. Are the LoRAs in /workspace/ComfyUI/models/loras/?")