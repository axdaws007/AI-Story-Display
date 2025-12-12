from image_generator_comfyui import ComfyUIImageGenerator
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

print("="*60)
print("TESTING IMAGE GENERATION WITH COMFYUI")
print("="*60)

# Configuration
comfyui_url = os.getenv('COMFYUI_URL')
story_file = 'stories/story_20251019_184004.json'
output_folder = os.getenv('COMFYUI_OUTPUT_FOLDER', '/workspace/ComfyUI/output')
base_model = os.getenv('BASE_MODEL', 'flux1-dev.safetensors')

# Validate configuration
if not comfyui_url:
    print("\n❌ ERROR: COMFYUI_URL not set in .env file")
    print("   Please add: COMFYUI_URL=http://your-comfyui-server:8188")
    exit(1)

if not Path(story_file).exists():
    print(f"\n❌ ERROR: Story file not found: {story_file}")
    print("   Please generate a story first:")
    print("   python story_generator.py")
    exit(1)

print(f"\nConfiguration:")
print(f"  ComfyUI URL: {comfyui_url}")
print(f"  Story file: {story_file}")
print(f"  Output folder: {output_folder}")
print(f"  Base model: {base_model}")

print(f"\nThis will generate images for all pages in the story")
print("Each page will take 30-60 seconds")
print("You'll see progress for each page...\n")

# Create generator instance
try:
    generator = ComfyUIImageGenerator(comfyui_url)
except Exception as e:
    print(f"\n❌ ERROR initializing generator: {e}")
    exit(1)

# Generate all images
print("Starting image generation...\n")

try:
    images = generator.generate_all_story_images(
        story_file,
        output_folder,
        base_model
    )
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    
    successful = sum(1 for i in images if i is not None)
    failed = len(images) - successful
    
    print(f"\nResults:")
    print(f"  ✓ Generated: {successful}/{len(images)} images")
    if failed > 0:
        print(f"  ✗ Failed: {failed} images")
    print(f"  📁 Output folder: images/story_20251019_184004/")
    
    if successful == len(images):
        print("\n🎉 All images generated successfully!")
    else:
        print("\n⚠️  Some images failed. Check the errors above.")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Generation cancelled by user")
    exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTroubleshooting:")
    print("1. Is ComfyUI running and accessible?")
    print("   Test: Open http://your-comfyui-url:8188 in browser")
    print("2. Is the COMFYUI_URL correct in .env?")
    print("3. Are the LoRAs in /workspace/ComfyUI/models/loras/?")
    print("4. Are the LoRA filenames in lora_config.json correct?")
    print("5. Is the base model file correct?")
    import traceback
    traceback.print_exc()
    exit(1)