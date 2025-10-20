import os
import json
import fal_client
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests

class ImageGenerator:
    def __init__(self, api_key, characters_file='characters.json'):
        # Set API key for fal_client
        os.environ['FAL_KEY'] = api_key
        
        self.characters = self.load_characters(characters_file)
        self.reference_images = {}
        
    def load_characters(self, file_path):
        """Load character profiles"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_reference_images(self):
        """Load character reference images from references folder"""
        ref_folder = Path("references")
        if not ref_folder.exists():
            print("WARNING: No references folder found. Create character references first.")
            return False
        
        # Look for character reference images
        for char_key in ['character_1', 'character_2']:
            char = self.characters[char_key]
            filename = f"{char['fantasy_name'].lower().replace(' ', '_')}.png"
            ref_path = ref_folder / filename
            
            if ref_path.exists():
                self.reference_images[char_key] = str(ref_path)
                print(f"✓ Loaded reference: {filename}")
            else:
                print(f"WARNING: No reference found for {char['fantasy_name']} at {ref_path}")
                return False
        
        return len(self.reference_images) > 0
    
    def create_character_reference_fal(self, character_key):
        """Create reference image for a character using FAL.ai"""
        char = self.characters[character_key]
        
        # Build detailed character description
        prompt = f"""Professional character design sheet, fantasy RPG style.

{char['fantasy_name']}, {char['race']} {char['class']}.

APPEARANCE:
- Build: {char['appearance']['build']}
- Height: {char['appearance']['height']}
- Hair: {char['appearance']['hair']}
- Eyes: {char['appearance']['eyes']}
- Features: {char['appearance']['distinctive_features']}

Multiple angles: front view, 3/4 view, side profile.
Clean background, detailed costume and equipment.
High fantasy illustration style, painterly, professional character art.
Consistent lighting, neutral pose for reference."""

        print(f"Generating reference for {char['fantasy_name']}...")
        
        # Use fal_client for simpler API calls
        result = fal_client.subscribe(
            "fal-ai/fast-sdxl",
            arguments={
                "prompt": prompt,
                "image_size": "landscape_16_9",
                "num_images": 1,
                "negative_prompt": "blurry, low quality, distorted, multiple characters, modern clothing"
            }
        )
        
        image_url = result['images'][0]['url']
        
        # Download and save image
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        
        output_path = f"references/{char['fantasy_name'].lower().replace(' ', '_')}.png"
        Path("references").mkdir(exist_ok=True)
        img.save(output_path)
        
        self.reference_images[character_key] = output_path
        print(f"✓ Created reference image: {output_path}")
        return output_path
    
    def generate_story_image_with_characters(self, scene_description, page_num, story_id):
        """Generate image for story page WITH character consistency using Instant Character"""
        
        # Make sure we have reference images loaded
        if not self.reference_images:
            has_refs = self.load_reference_images()
            if not has_refs:
                raise Exception("No character reference images found! Create them first with create_character_reference_fal()")
        
        char1 = self.characters['character_1']
        char2 = self.characters['character_2']
        
        # Build enhanced prompt
        full_prompt = f"""{scene_description}

Featuring {char1['fantasy_name']} ({char1['race']} {char1['class']}) and {char2['fantasy_name']} ({char2['race']} {char2['class']}).
High fantasy illustration, painterly style, dramatic lighting, detailed environment.
Book illustration quality, cinematic composition."""

        # Upload reference image using fal_client
        print(f"  Uploading reference image...")
        ref_image_url = fal_client.upload_file(self.reference_images['character_1'])
        
        print(f"  Generating image...")
        
        # Use Instant Character endpoint
        result = fal_client.subscribe(
            "fal-ai/instant-character",
            arguments={
                "prompt": full_prompt,
                "image_url": ref_image_url,
                "image_size": "landscape_4_3",
                "scale": 1.2,
                "negative_prompt": "text, watermark, signature, blurry, low quality, modern elements, inconsistent characters, different face",
                "num_images": 1,
                "guidance_scale": 4.0
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
    
    def generate_all_story_images(self, story_file):
        """Generate images for all pages in a story with character consistency"""
        with open(story_file, 'r') as f:
            story_data = json.load(f)
        
        story_id = Path(story_file).stem
        images = []
        
        # Load reference images first
        has_refs = self.load_reference_images()
        
        if not has_refs:
            print("\n⚠️  ERROR: No character reference images found!")
            print("You must create character references first:")
            print("Run: python image_generator.py")
            print("Or manually create character reference images in references/ folder\n")
            return []
        
        print(f"\n✓ Using Instant Character endpoint for consistency")
        print(f"✓ Reference images loaded: {len(self.reference_images)}\n")
        print(f"Generating {len(story_data['pages'])} images...")
        print("This will take several minutes...\n")
        
        for page in story_data['pages']:
            print(f"Page {page['page']}/{len(story_data['pages'])}...")
            try:
                img_path = self.generate_story_image_with_characters(
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
        return images

# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('FAL_API_KEY')
    if not api_key:
        print("ERROR: Please set FAL_API_KEY in .env file")
        print("Get API key from: https://fal.ai/dashboard/keys")
        exit(1)
    
    generator = ImageGenerator(api_key)
    
    print("=" * 60)
    print("CHARACTER REFERENCE GENERATOR")
    print("=" * 60)
    print("\nCreating character reference images...")
    print("These will be used to maintain consistency across all story images.\n")
    
    try:
        generator.create_character_reference_fal('character_1')
        generator.create_character_reference_fal('character_2')
        print("\n✓ Character references created successfully!")
        print("\nNext step: Generate story images with consistent characters")
        print("Usage: generator.generate_all_story_images('stories/your_story.json')")
    except Exception as e:
        print(f"\nERROR: {e}")