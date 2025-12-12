import os
import json
import fal_client
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests

class LoRAImageGenerator:
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
        if not Path(file_path).exists():
            print(f"⚠ No LoRA config found at {file_path}")
            print("   Falling back to standard image generation")
            return None
        
        with open(file_path, 'r') as f:
            loras = json.load(f)
        
        print(f"✓ Loaded LoRA config with {len(loras)} trained characters")
        for char_name, lora_info in loras.items():
            print(f"  - {char_name}: {lora_info['trigger_word']}")
        
        return loras
    
    def build_detailed_prompt(self, scene_description, char1_trigger, char2_trigger):
        """Build a concise but explicit prompt for accurate scene rendering"""
        
        char1 = self.characters['character_1']
        char2 = self.characters['character_2']
        
        # Simplified prompt - LoRAs handle the detailed appearance
        # Just reinforce key identifiers
        detailed_prompt = f"""{scene_description}

{char1_trigger} is the {char1.get('race', 'character')} {char1.get('class', 'adventurer')}.
{char2_trigger} is the {char2.get('race', 'character')} {char2.get('class', 'adventurer')}.

Professional fantasy book illustration, detailed, sharp focus, cinematic composition, dramatic lighting."""

        return detailed_prompt
    
    def generate_story_image_with_loras(self, scene_description, page_num, story_id):
        """Generate image using trained LoRAs for perfect character consistency"""
        
        if not self.loras:
            raise Exception("No LoRAs loaded! Train LoRAs first with lora_trainer.py")
        
        char1 = self.characters['character_1']
        char2 = self.characters['character_2']
        char1_name = char1['fantasy_name']
        char2_name = char2['fantasy_name']
        
        # Get LoRA info for both characters
        char1_lora = self.loras.get(char1_name)
        char2_lora = self.loras.get(char2_name)
        
        if not char1_lora or not char2_lora:
            raise Exception(f"Missing LoRAs. Have: {list(self.loras.keys())}")
        
        # Build highly detailed prompt
        full_prompt = self.build_detailed_prompt(
            scene_description,
            char1_lora['trigger_word'],
            char2_lora['trigger_word']
        )
        
        # Very strong negative prompt - prioritize preventing errors
        negative_prompt = """multiple people beyond the two main characters, crowd scene, many people, 
        extra limbs, third arm, extra arms, multiple arms, more than two arms per person,
        deformed hands, extra fingers, fused fingers, malformed limbs, missing limbs, 
        extra legs, more than two legs per person, wrong number of limbs, mutated body parts,
        merged bodies, characters fused together, conjoined characters,
        extra heads, multiple heads, deformed anatomy, unnatural proportions,
        blurry, low quality, distorted faces, unclear which character is which,
        characters swapped, wrong character performing action,
        modern elements, contemporary clothing, photographs, photorealistic,
        text, watermarks, signatures, logos, frames, borders"""
        
        print(f"  Generating with both character LoRAs...")
        print(f"  Using triggers: {char1_lora['trigger_word']}, {char2_lora['trigger_word']}")
        
        # Generate image using BOTH LoRAs with improved settings
        result = fal_client.subscribe(
            "fal-ai/flux-lora",
            arguments={
                "prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "loras": [
                    {
                        "path": char1_lora['lora_url'],
                        "scale": 1.0  # Full strength for consistency
                    },
                    {
                        "path": char2_lora['lora_url'],
                        "scale": 1.0  # Full strength for consistency
                    }
                ],
                "image_size": "landscape_4_3",
                "num_inference_steps": 35,  # Balanced quality and speed
                "guidance_scale": 5.0,  # Higher = follows prompt more strictly
                "num_images": 1
            }
        )
        
        image_url = result['images'][0]['url']
        
        # Download and save
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        
        # Optimize for e-ink display (800x480 for Inky Impression)
        img = img.resize((800, 480), Image.Resampling.LANCZOS)
        
        output_dir = f"images/{story_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/page_{page_num:02d}.png"
        img.save(output_path)
        
        print(f"  ✓ Generated: {output_path}")
        return output_path
    
    def regenerate_single_page(self, story_file, page_number):
        """Regenerate just one specific page if it has errors"""
        with open(story_file, 'r') as f:
            story_data = json.load(f)
        
        story_id = Path(story_file).stem
        
        # Find the specific page
        page = None
        for p in story_data['pages']:
            if p['page'] == page_number:
                page = p
                break
        
        if not page:
            print(f"Error: Page {page_number} not found in story")
            return None
        
        print(f"\nRegenerating page {page_number}...")
        print(f"Scene: {page['scene_description'][:100]}...")
        
        try:
            img_path = self.generate_story_image_with_loras(
                page['scene_description'],
                page['page'],
                story_id
            )
            
            # Update story file
            page['image_path'] = img_path
            with open(story_file, 'w') as f:
                json.dump(story_data, f, indent=2)
            
            print(f"✓ Page {page_number} regenerated successfully!")
            return img_path
            
        except Exception as e:
            print(f"Error regenerating page {page_number}: {e}")
            return None
    
    def generate_all_story_images(self, story_file):
        """Generate images for all pages using trained LoRAs"""
        with open(story_file, 'r') as f:
            story_data = json.load(f)
        
        story_id = Path(story_file).stem
        images = []
        
        if not self.loras:
            print("\n⚠️  ERROR: No LoRAs loaded!")
            print("Train LoRAs first with: python lora_trainer.py")
            return []
        
        print(f"\n✓ Using trained LoRAs for PERFECT character consistency")
        print(f"✓ Improved scene accuracy with detailed prompts")
        print(f"✓ Both characters will look identical across all pages\n")
        print(f"Generating {len(story_data['pages'])} images...")
        print("This will take 8-12 minutes with improved quality settings...\n")
        
        for page in story_data['pages']:
            print(f"Page {page['page']}/{len(story_data['pages'])}...")
            print(f"  Scene: {page['scene_description'][:80]}...")
            try:
                img_path = self.generate_story_image_with_loras(
                    page['scene_description'],
                    page['page'],
                    story_id
                )
                images.append(img_path)
            except Exception as e:
                print(f"  ERROR: {e}")
                images.append(None)
        
        # Update story file with image paths
        for i, page in enumerate(story_data['pages']):
            page['image_path'] = images[i]
        
        with open(story_file, 'w') as f:
            json.dump(story_data, f, indent=2)
        
        successful = sum(1 for img in images if img is not None)
        print(f"\n✓ Generated {successful}/{len(images)} images successfully!")
        print(f"✓ Story updated: {story_file}")
        
        if successful < len(images):
            print(f"\n⚠️  {len(images) - successful} images failed to generate")
            print("You can regenerate individual pages with:")
            print("  generator.regenerate_single_page('story_file.json', page_number)")
        
        return images


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('FAL_API_KEY')
    if not api_key:
        print("ERROR: Please set FAL_API_KEY in .env file")
        exit(1)
    
    generator = LoRAImageGenerator(api_key)
    
    print("\n" + "="*60)
    print("LoRA-ENHANCED IMAGE GENERATOR (IMPROVED ACCURACY)")
    print("="*60)
    print("\nThis will generate story images using your trained LoRAs")
    print("with improved scene accuracy and error prevention!\n")
    
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
        generator.generate_all_story_images(str(selected_story))
        
        print("\n" + "="*60)
        print("REVIEW YOUR IMAGES!")
        print("="*60)
        print("\nCheck the images folder for any issues:")
        print("- Wrong character doing action? Regenerate that page")
        print("- Extra limbs or deformities? Regenerate that page")
        print("- Scene doesn't match story? Regenerate that page")
        print("\nTo regenerate a specific page:")
        print("  python")
        print("  >>> from image_generator_lora import LoRAImageGenerator")
        print("  >>> from dotenv import load_dotenv")
        print("  >>> import os")
        print("  >>> load_dotenv()")
        print("  >>> gen = LoRAImageGenerator(os.getenv('FAL_API_KEY'))")
        print(f"  >>> gen.regenerate_single_page('{selected_story}', PAGE_NUMBER)")
        print("="*60)
    else:
        print("Invalid choice!")