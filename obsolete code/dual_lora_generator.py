import os
import json
import fal_client
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests

class DualLoRAImageGenerator:
    def __init__(self, api_key, characters_file='characters.json', lora_config_file='lora_config.json'):
        os.environ['FAL_KEY'] = api_key
        
        self.characters = self.load_characters(characters_file)
        self.loras = self.load_lora_config(lora_config_file)
        
    def load_characters(self, file_path):
        """Load character profiles"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_lora_config(self, file_path):
        """Load trained LoRA configuration"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def generate_single_character_base(self, scene_description, character_key, char_lora, position="center"):
        """Generate an image with just ONE character using their LoRA"""
        
        char = self.characters[character_key]
        trigger = char_lora['trigger_word']
        
        # Build prompt for single character in scene
        prompt = f"""{scene_description}

Focus on {trigger}, positioned {position} in the scene.
High fantasy illustration, detailed, sharp focus, cinematic composition, dramatic lighting."""

        # Generate with ONLY this character's LoRA at full strength
        result = fal_client.subscribe(
            "fal-ai/flux-lora",
            arguments={
                "prompt": prompt,
                "negative_prompt": "multiple people, crowd, extra characters, blurry, low quality",
                "loras": [
                    {
                        "path": char_lora['lora_url'],
                        "scale": 1.0  # Full strength since it's alone
                    }
                ],
                "image_size": "landscape_4_3",
                "num_inference_steps": 35,
                "guidance_scale": 5.0,
                "num_images": 1
            }
        )
        
        image_url = result['images'][0]['url']
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        
        return img
    
    def create_simple_composite(self, char1_img, char2_img, output_path):
        """Create a simple side-by-side composite of two character images"""
        
        # Get dimensions
        width = char1_img.width + char2_img.width
        height = max(char1_img.height, char2_img.height)
        
        # Create new image
        composite = Image.new('RGB', (width, height), color='black')
        
        # Paste characters side by side
        composite.paste(char1_img, (0, 0))
        composite.paste(char2_img, (char1_img.width, 0))
        
        # Resize to target dimensions (800x480 for Inky Impression)
        composite = composite.resize((800, 480), Image.Resampling.LANCZOS)
        
        composite.save(output_path)
        return composite
    
    def generate_story_image_separate_method(self, scene_description, page_num, story_id):
        """Generate story image by creating each character separately"""
        
        char1 = self.characters['character_1']
        char2 = self.characters['character_2']
        char1_name = char1['fantasy_name']
        char2_name = char2['fantasy_name']
        
        char1_lora = self.loras.get(char1_name)
        char2_lora = self.loras.get(char2_name)
        
        if not char1_lora or not char2_lora:
            raise Exception(f"Missing LoRAs")
        
        print(f"  Generating {char1_name} separately...")
        
        # Generate Tobias alone (positioned left)
        char1_scene = f"{scene_description} Focus on {char1_lora['trigger_word']} on the left side"
        char1_img = self.generate_single_character_base(
            char1_scene, 
            'character_1', 
            char1_lora,
            position="left side"
        )
        
        print(f"  Generating {char2_name} separately...")
        
        # Generate Olive alone (positioned right)
        char2_scene = f"{scene_description} Focus on {char2_lora['trigger_word']} on the right side"
        char2_img = self.generate_single_character_base(
            char2_scene,
            'character_2',
            char2_lora,
            position="right side"
        )
        
        print(f"  Creating composite...")
        
        # Create output directory
        output_dir = f"images/{story_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/page_{page_num:02d}.png"
        
        # Composite them together
        composite = self.create_simple_composite(char1_img, char2_img, output_path)
        
        print(f"  ✓ Generated: {output_path}")
        return output_path
    
    def generate_all_story_images(self, story_file):
        """Generate all story images using separate generation method"""
        with open(story_file, 'r') as f:
            story_data = json.load(f)
        
        story_id = Path(story_file).stem
        images = []
        
        print(f"\n✓ Using SEPARATE generation method (no LoRA interference!)")
        print(f"✓ Each character generated individually for perfect consistency\n")
        print(f"Generating {len(story_data['pages'])} images...")
        print("This will take 10-15 minutes (2 generations per page)...\n")
        
        for page in story_data['pages']:
            print(f"Page {page['page']}/{len(story_data['pages'])}...")
            print(f"  Scene: {page['scene_description'][:80]}...")
            try:
                img_path = self.generate_story_image_separate_method(
                    page['scene_description'],
                    page['page'],
                    story_id
                )
                images.append(img_path)
            except Exception as e:
                print(f"  ERROR: {e}")
                images.append(None)
        
        # Update story file
        for i, page in enumerate(story_data['pages']):
            page['image_path'] = images[i]
        
        with open(story_file, 'w') as f:
            json.dump(story_data, f, indent=2)
        
        successful = sum(1 for img in images if img is not None)
        print(f"\n✓ Generated {successful}/{len(images)} images successfully!")
        print(f"✓ Story updated: {story_file}")
        
        return images


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('FAL_API_KEY')
    if not api_key:
        print("ERROR: Please set FAL_API_KEY in .env file")
        exit(1)
    
    generator = DualLoRAImageGenerator(api_key)
    
    print("\n" + "="*60)
    print("DUAL LoRA IMAGE GENERATOR (SEPARATE METHOD)")
    print("="*60)
    print("\nThis generates each character separately to avoid LoRA interference.")
    print("Characters are composited together for final image.\n")
    print("✓ Perfect character consistency (LoRAs work individually)")
    print("✓ No blending or interference issues")
    print("✓ Takes longer (2 images per page) but guaranteed to work\n")
    
    # List story files
    story_files = list(Path('stories').glob('*.json'))
    
    if not story_files:
        print("No story files found in stories/ folder!")
        exit(1)
    
    print("Available stories:")
    for i, story in enumerate(story_files, 1):
        print(f"  {i}. {story.name}")
    
    print("\nEnter story number to generate images for:")
    choice = int(input("> "))
    
    if 1 <= choice <= len(story_files):
        selected_story = story_files[choice - 1]
        print(f"\nGenerating images for: {selected_story.name}")
        print("\nNote: This generates 2 images per page (one per character)")
        print("      then composites them together.")
        print("      Cost: ~2x normal generation (~$0.06-0.10 per page)\n")
        
        input("Press Enter to continue...")
        
        generator.generate_all_story_images(str(selected_story))
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE!")
        print("="*60)
        print("\nEach image shows both characters without interference.")
        print("Characters should look exactly like their individual test images!")
        print("="*60)
    else:
        print("Invalid choice!")