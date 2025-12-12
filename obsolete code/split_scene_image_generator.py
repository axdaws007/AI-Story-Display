import os
import json
import fal_client
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

class SplitSceneImageGenerator:
    def __init__(self, fal_api_key, gemini_api_key, characters_file='characters.json', lora_config_file='lora_config.json'):
        os.environ['FAL_KEY'] = fal_api_key
        self.gemini_api_key = gemini_api_key
        
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
    
    def split_scene_description(self, scene_description, page_text):
        """
        Use Gemini to split the scene into two character-specific descriptions
        """
        char1 = self.characters['character_1']
        char2 = self.characters['character_2']
        char1_name = char1['fantasy_name']
        char2_name = char2['fantasy_name']
        
        # Get visual details for prompting
        char1_visual = char1.get('visual_design', {})
        char2_visual = char2.get('visual_design', {})
        
        prompt = f"""You are creating image generation prompts for a fantasy story. You need to split a scene description into TWO separate descriptions - one for each character that can be generated independently and then placed side-by-side.

ORIGINAL SCENE: {scene_description}

PAGE TEXT: {page_text}

CHARACTER 1: {char1_name}
- {char1.get('race', 'Character')} {char1.get('class', 'Adventurer')}
- Key appearance: {char1_visual.get('hair', '')}, {char1_visual.get('eyes', '')}, {char1_visual.get('build', '')}
- Typical outfit: {char1_visual.get('typical_outfit', '')}

CHARACTER 2: {char2_name}
- {char2.get('race', 'Character')} {char2.get('class', 'Adventurer')}
- Key appearance: {char2_visual.get('hair', '')}, {char2_visual.get('eyes', '')}, {char2_visual.get('build', '')}
- Typical outfit: {char2_visual.get('typical_outfit', '')}

Create TWO separate image descriptions:

1. **{char1_name}'s description**: What {char1_name} is doing, their pose, expression, and position (will be placed on LEFT side). Include the shared environment/background details. Focus ONLY on {char1_name} - describe them as if they are alone in the scene.

2. **{char2_name}'s description**: What {char2_name} is doing, their pose, expression, and position (will be placed on RIGHT side). Include the shared environment/background details. Focus ONLY on {char2_name} - describe them as if they are alone in the scene.

CRITICAL RULES:
- Each description should describe ONLY ONE character
- Include the environment/setting in both descriptions so backgrounds match
- Make positioning natural for side-by-side composition (left character in left description, right character in right description)
- Each description should work as a standalone image prompt
- Do NOT mention the other character in each description
- Keep lighting and atmosphere consistent between both descriptions

Return your response as a JSON object:
{{
  "char1_description": "Description for {char1_name} only...",
  "char2_description": "Description for {char2_name} only...",
  "shared_environment": "Brief description of setting/background..."
}}"""

        # Call Gemini API
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        
        headers = {'Content-Type': 'application/json'}
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }
        
        response = requests.post(
            f"{url}?key={self.gemini_api_key}",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON from response
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            split_descriptions = json.loads(text)
            return split_descriptions
        else:
            raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")
    
    def generate_single_character_image(self, character_key, description):
        """Generate an image with ONLY one character using their LoRA"""
        
        char = self.characters[character_key]
        char_name = char['fantasy_name']
        char_lora = self.loras.get(char_name)
        
        if not char_lora:
            raise Exception(f"Missing LoRA for {char_name}")
        
        trigger = char_lora['trigger_word']
        
        # Build prompt with trigger word and description
        full_prompt = f"""{trigger}, {description}

Single character portrait, {trigger} alone in scene.
High fantasy illustration, detailed, professional fantasy art.
Clear focus on the character, dramatic lighting."""

        print(f"    Generating with trigger: {trigger}")
        print(f"    Prompt: {full_prompt[:100]}...")
        
        # Generate with ONLY this character's LoRA
        result = fal_client.subscribe(
            "fal-ai/flux-lora",
            arguments={
                "prompt": full_prompt,
                "negative_prompt": "multiple people, two people, crowd, extra characters, group scene, other person, second character, blurry, low quality, deformed",
                "loras": [
                    {
                        "path": char_lora['lora_url'],
                        "scale": 1.0
                    }
                ],
                "image_size": "portrait_4_3",  # Portrait for character focus
                "num_inference_steps": 35,
                "guidance_scale": 5.5,  # Higher for better prompt following
                "num_images": 1
            }
        )
        
        image_url = result['images'][0]['url']
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        
        return img
    
    def create_side_by_side_composite(self, char1_img, char2_img, output_path):
        """Create a side-by-side composite of two character images"""
        
        # Target dimensions for e-ink display
        target_width = 800
        target_height = 480
        
        # Each character gets half the width
        char_width = target_width // 2
        
        # Resize both images to same height, maintaining aspect ratio
        char1_resized = self.resize_to_height(char1_img, target_height, char_width)
        char2_resized = self.resize_to_height(char2_img, target_height, char_width)
        
        # Create composite image
        composite = Image.new('RGB', (target_width, target_height), color='black')
        
        # Center each character in their half
        char1_x = (char_width - char1_resized.width) // 2
        char2_x = char_width + (char_width - char2_resized.width) // 2
        
        composite.paste(char1_resized, (char1_x, 0))
        composite.paste(char2_resized, (char2_x, 0))
        
        composite.save(output_path)
        return composite
    
    def resize_to_height(self, img, target_height, max_width):
        """Resize image to target height while maintaining aspect ratio and not exceeding max width"""
        aspect = img.width / img.height
        new_width = int(target_height * aspect)
        
        # If too wide, scale down
        if new_width > max_width:
            new_width = max_width
            target_height = int(new_width / aspect)
        
        return img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    def generate_story_image_split_method(self, page_data, story_id):
        """
        Generate story image by:
        1. Splitting scene into character-specific descriptions
        2. Generating each character separately
        3. Compositing side-by-side
        """
        
        scene_description = page_data['scene_description']
        page_text = page_data['text']
        page_num = page_data['page']
        
        print(f"  Step 1: Splitting scene description...")
        split_descriptions = self.split_scene_description(scene_description, page_text)
        
        char1_name = self.characters['character_1']['fantasy_name']
        char2_name = self.characters['character_2']['fantasy_name']
        
        print(f"  Step 2: Generating {char1_name}...")
        print(f"    Description: {split_descriptions['char1_description'][:80]}...")
        char1_img = self.generate_single_character_image(
            'character_1',
            split_descriptions['char1_description']
        )
        
        print(f"  Step 3: Generating {char2_name}...")
        print(f"    Description: {split_descriptions['char2_description'][:80]}...")
        char2_img = self.generate_single_character_image(
            'character_2',
            split_descriptions['char2_description']
        )
        
        print(f"  Step 4: Creating composite...")
        
        # Create output directory
        output_dir = f"images/{story_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/page_{page_num:02d}.png"
        
        composite = self.create_side_by_side_composite(char1_img, char2_img, output_path)
        
        print(f"  ✓ Generated: {output_path}")
        
        # Also save the split descriptions for reference
        descriptions_path = f"{output_dir}/page_{page_num:02d}_descriptions.json"
        with open(descriptions_path, 'w') as f:
            json.dump(split_descriptions, f, indent=2)
        
        return output_path
    
    def generate_all_story_images(self, story_file):
        """Generate images for all pages using split scene method"""
        with open(story_file, 'r') as f:
            story_data = json.load(f)
        
        story_id = Path(story_file).stem
        images = []
        
        print(f"\n✓ Using SPLIT SCENE method")
        print(f"✓ Each character gets their own specific description")
        print(f"✓ Perfect character consistency (no LoRA interference)")
        print(f"✓ 3 AI calls per page: 1 scene split + 2 images\n")
        print(f"Generating {len(story_data['pages'])} images...")
        print("This will take 15-20 minutes...\n")
        
        for page in story_data['pages']:
            print(f"Page {page['page']}/{len(story_data['pages'])}...")
            print(f"  Text: {page['text'][:60]}...")
            print(f"  Scene: {page['scene_description'][:80]}...")
            try:
                img_path = self.generate_story_image_split_method(
                    page,
                    story_id
                )
                images.append(img_path)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                images.append(None)
        
        # Update story file with image paths
        for i, page in enumerate(story_data['pages']):
            page['image_path'] = images[i]
        
        with open(story_file, 'w') as f:
            json.dump(story_data, f, indent=2)
        
        successful = sum(1 for img in images if img is not None)
        print(f"\n✓ Generated {successful}/{len(images)} images successfully!")
        print(f"✓ Story updated: {story_file}")
        print(f"\nCheck the images folder - each page also has a _descriptions.json")
        print(f"file showing how the scene was split for each character.")
        
        return images


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    fal_api_key = os.getenv('FAL_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not fal_api_key or not gemini_api_key:
        print("ERROR: Please set both FAL_API_KEY and GEMINI_API_KEY in .env file")
        exit(1)
    
    generator = SplitSceneImageGenerator(fal_api_key, gemini_api_key)
    
    print("\n" + "="*60)
    print("SPLIT SCENE IMAGE GENERATOR")
    print("="*60)
    print("\nThis approach:")
    print("  1. Uses Gemini to split each scene into character-specific descriptions")
    print("  2. Generates each character separately with ONLY their description")
    print("  3. Composites them side-by-side")
    print("\nAdvantages:")
    print("  ✓ Each character generated alone (no extra people)")
    print("  ✓ Perfect character consistency (isolated LoRAs)")
    print("  ✓ AI handles the scene splitting intelligently")
    print("  ✓ Matching backgrounds/environment")
    print("\nCost per page:")
    print("  - 1 Gemini API call (free tier: ~$0.00)")
    print("  - 2 FLUX LoRA generations (~$0.06-0.10 total)")
    print("\n")
    
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
        
        # Count pages properly
        with open(selected_story, 'r') as f:
            story_data = json.load(f)
        num_pages = len(story_data['pages'])
        
        print("\nThis will:")
        print("  - Split each scene into character-specific descriptions")
        print("  - Generate 2 images per page (one per character)")
        print("  - Composite them side-by-side")
        print(f"  - Total: {num_pages} pages\n")
        
        input("Press Enter to continue...")
        
        generator.generate_all_story_images(str(selected_story))
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE!")
        print("="*60)
        print("\nEach character should now appear correctly in their own space!")
        print("Check the images folder - you'll also find _descriptions.json")
        print("files showing how each scene was split.")
        print("="*60)
    else:
        print("Invalid choice!")