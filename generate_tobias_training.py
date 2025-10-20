import os
import json
import fal_client
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests
import time

class SyntheticTrainingDataGenerator:
    def __init__(self, api_key, characters_file='characters.json'):
        os.environ['FAL_KEY'] = api_key
        self.characters = self.load_characters(characters_file)
    
    def load_characters(self, file_path):
        """Load character profiles"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def generate_character_training_set(self, character_key, num_images=20):
        """
        Generate a set of training images for a character
        
        Args:
            character_key: 'character_1' or 'character_2'
            num_images: Number of training images to generate (15-20 recommended)
        """
        
        char = self.characters[character_key]
        char_name = char['fantasy_name']
        visual = char.get('visual_design', char.get('appearance', {}))
        
        print(f"\n{'='*60}")
        print(f"GENERATING TRAINING SET: {char_name}")
        print(f"{'='*60}")
        print(f"Target: {num_images} images")
        print(f"This will take 10-20 minutes and cost ~$0.70-1.00\n")
        
        # Create output directory
        output_dir = Path(f"lora_training/{character_key}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build base character description
        base_description = self._build_character_description(char)
        
        print(f"Character description being used:")
        print(f"{base_description}\n")
        
        # Generate varied prompts for different angles/expressions/poses
        prompts = self._generate_varied_prompts(base_description, num_images)
        
        print(f"Generating {len(prompts)} images...\n")
        
        generated_count = 0
        
        for i, prompt_data in enumerate(prompts, 1):
            print(f"Image {i}/{len(prompts)}: {prompt_data['description']}")
            
            try:
                # Generate image using FLUX
                result = fal_client.subscribe(
                    "fal-ai/flux/dev",  # Using FLUX dev for high quality
                    arguments={
                        "prompt": prompt_data['prompt'],
                        "image_size": "square_hd",  # 1024x1024 good for training
                        "num_inference_steps": 28,
                        "guidance_scale": 3.5,
                        "num_images": 1
                    }
                )
                
                image_url = result['images'][0]['url']
                
                # Download and save
                img_response = requests.get(image_url)
                img = Image.open(BytesIO(img_response.content))
                
                output_path = output_dir / f"{i:02d}.jpg"
                img.save(output_path, quality=95)
                
                print(f"  ✓ Saved: {output_path}")
                generated_count += 1
                
                # Small delay to avoid rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                print(f"  Continuing with next image...")
                continue
        
        print(f"\n{'='*60}")
        print(f"✓ Generated {generated_count}/{len(prompts)} images")
        print(f"✓ Saved to: {output_dir}")
        print(f"{'='*60}\n")
        
        return generated_count
    
    def _build_character_description(self, char):
        """Build detailed character description from profile"""
        visual = char.get('visual_design', char.get('appearance', {}))
        
        parts = []
        
        # Start with name and class
        parts.append(f"{char['fantasy_name']}")
        parts.append(f"{char.get('race', 'human')} {char.get('class', 'adventurer')}")
        
        # Add physical details
        if isinstance(visual, dict):
            if 'face' in visual:
                parts.append(visual['face'])
            if 'hair' in visual:
                parts.append(visual['hair'])
            if 'eyes' in visual:
                parts.append(visual['eyes'])
            if 'build' in visual:
                parts.append(visual['build'])
            if 'skin' in visual:
                parts.append(visual['skin'])
            if 'distinctive' in visual:
                parts.append(visual['distinctive'])
            if 'typical_outfit' in visual:
                parts.append(f"wearing {visual['typical_outfit']}")
        
        description = ", ".join(parts)
        return description
    
    def _generate_varied_prompts(self, base_description, num_images):
        """Generate varied prompts for different angles, expressions, and scenarios"""
        
        variations = [
            # Front-facing portraits (5-6 images)
            {"angle": "front view portrait, looking at camera", "expression": "calculating expression, slight smirk", "lighting": "dramatic lighting", "style": "detailed fantasy character portrait"},
            {"angle": "front view, direct gaze", "expression": "serious and focused", "lighting": "soft cinematic lighting", "style": "professional character art"},
            {"angle": "front facing, head slightly tilted", "expression": "confident and arrogant look", "lighting": "rim lighting", "style": "high quality fantasy portrait"},
            {"angle": "straight on portrait", "expression": "subtle smile, intelligent gaze", "lighting": "studio lighting", "style": "detailed character illustration"},
            {"angle": "front view, neutral pose", "expression": "observant and alert", "lighting": "natural daylight", "style": "rpg character portrait"},
            {"angle": "centered portrait composition", "expression": "cunning expression", "lighting": "dramatic shadows", "style": "fantasy character concept art"},
            
            # 3/4 view (5-6 images)
            {"angle": "three-quarter view from right", "expression": "slight smirk, plotting", "lighting": "side lighting", "style": "detailed fantasy portrait"},
            {"angle": "3/4 angle from left side", "expression": "thoughtful and calculating", "lighting": "soft ambient light", "style": "character concept art"},
            {"angle": "three-quarter view, looking to side", "expression": "confident bearing", "lighting": "moody dramatic lighting", "style": "professional character portrait"},
            {"angle": "3/4 portrait from right", "expression": "subtle arrogance", "lighting": "cinematic lighting", "style": "high quality fantasy art"},
            {"angle": "three-quarter angle looking over shoulder", "expression": "cautious but interested", "lighting": "backlit with rim light", "style": "rpg character illustration"},
            {"angle": "3/4 view portrait", "expression": "aristocratic demeanor", "lighting": "soft directional light", "style": "detailed character art"},
            
            # Side profiles (3-4 images)
            {"angle": "side profile view from left", "expression": "sharp features in profile", "lighting": "profile lighting", "style": "character portrait"},
            {"angle": "right side profile portrait", "expression": "noble bearing", "lighting": "dramatic side light", "style": "fantasy character art"},
            {"angle": "profile view, looking into distance", "expression": "contemplative", "lighting": "soft natural light", "style": "detailed portrait"},
            {"angle": "side profile, sharp angles", "expression": "regal and composed", "lighting": "rim lighting", "style": "high quality character illustration"},
            
            # Action/dynamic poses (3-4 images)
            {"angle": "dynamic pose, front angle, ready stance", "expression": "alert and prepared", "lighting": "dramatic action lighting", "style": "fantasy character action art"},
            {"angle": "confident stance, 3/4 view, hand on weapon", "expression": "cocky grin", "lighting": "heroic lighting", "style": "rpg character portrait"},
            {"angle": "stealth pose, crouched slightly", "expression": "focused and intense", "lighting": "moody shadows", "style": "rogue character concept art"},
            {"angle": "standing confidently, arms crossed", "expression": "self-assured smirk", "lighting": "strong directional light", "style": "detailed character portrait"},
        ]
        
        # Select the right number of variations
        selected_variations = variations[:num_images]
        
        prompts = []
        for i, var in enumerate(selected_variations, 1):
            prompt = f"{base_description}, {var['angle']}, {var['expression']}, {var['lighting']}, {var['style']}, sharp focus, detailed features, high quality, professional artwork"
            
            prompts.append({
                'prompt': prompt,
                'description': f"{var['angle']}, {var['expression']}"
            })
        
        return prompts


# Main execution
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('FAL_API_KEY')
    if not api_key:
        print("ERROR: Please set FAL_API_KEY in .env file")
        exit(1)
    
    generator = SyntheticTrainingDataGenerator(api_key)
    
    print("\n" + "="*60)
    print("SYNTHETIC LoRA TRAINING DATA GENERATOR")
    print("Testing with Character 1 (Tobias Dunsmir)")
    print("="*60)
    print("\nThis will generate AI character images for LoRA training.")
    print("No real photos needed - creates consistent fantasy character!")
    print("\nCost estimate:")
    print("  - 20 images × ~$0.035 per image = ~$0.70 total")
    print("\nTime estimate: 10-20 minutes\n")
    
    print("Press Enter to start, or Ctrl+C to cancel...")
    input()
    
    # Generate training set for Character 1 (Tobias)
    count1 = generator.generate_character_training_set('character_1', num_images=20)
    
    print("\n" + "="*60)
    print("✓ TRAINING IMAGES GENERATED!")
    print("="*60)
    print(f"\nTobias Dunsmir: {count1} images in lora_training/character_1/")
    print("\n⚠️  IMPORTANT: Review the images!")
    print("   - Do they all look like the SAME person?")
    print("   - Do they match your vision of Tobias?")
    print("   - Check the silver hair, blue eyes, grey skin")
    print("   - Look for consistency in facial features")
    print("   - Delete any that look too different or low quality")
    print("\nWhat to look for:")
    print("  ✓ Consistent facial structure across all images")
    print("  ✓ Same silver wavy hair in all images")
    print("  ✓ Same piercing blue eyes")
    print("  ✓ Grey high elf skin tone")
    print("  ✓ Sharp aristocratic features")
    print("  ✓ Various angles and expressions but SAME face")
    print("\nIf satisfied with the images:")
    print("1. Keep 15-18 best images (delete any outliers)")
    print("2. Run: python prepare_lora_training.py")
    print("3. Run: python lora_trainer.py")
    print("\nIf NOT satisfied:")
    print("1. We can adjust the character description")
    print("2. Regenerate with tweaked parameters")
    print("="*60)