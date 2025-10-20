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
        """Generate training images for a character"""
        
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
        prompts = self._generate_varied_prompts(base_description, num_images, char_name)
        
        print(f"Generating {len(prompts)} images...\n")
        
        generated_count = 0
        
        for i, prompt_data in enumerate(prompts, 1):
            print(f"Image {i}/{len(prompts)}: {prompt_data['description']}")
            
            try:
                # Generate image using FLUX with VERY STRONG negative prompt for ear control
                result = fal_client.subscribe(
                    "fal-ai/flux/dev",
                    arguments={
                        "prompt": prompt_data['prompt'],
                        "negative_prompt": "extremely pointed ears, very long ears, exaggerated elf ears, sharp pointed ears, long elf ears, dramatic elf ears, elven ears, fantasy elf ears, Legolas ears, elongated ears, pointy ears extending upward, ears longer than 2 inches, teenage, young girl, youthful face, baby face, child, adolescent, very young, under 25 years old",
                        "image_size": "square_hd",
                        "num_inference_steps": 28,
                        "guidance_scale": 4.0,  # Increased from 3.5 for better prompt following
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
        
        # Start with name and class - de-emphasize "elf" aspect
        parts.append(f"{char['fantasy_name']}")
        parts.append(f"half-human paladin warrior")  # Emphasize human side
        
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
            # Add ear description explicitly
            parts.append("barely pointed ears, almost human-looking ears")
            if 'typical_outfit' in visual:
                parts.append(f"wearing {visual['typical_outfit']}")
        
        description = ", ".join(parts)
        return description
    
    def _generate_varied_prompts(self, base_description, num_images, char_name):
        """Generate varied prompts for different angles, expressions, and scenarios"""
        
        variations = [
            # Front-facing portraits (5-6 images) - emphasizing mature adult features
            {"angle": "front view portrait, looking at camera", "expression": "warm gentle smile with laugh lines", "lighting": "soft natural lighting", "style": "detailed fantasy character portrait, mature adult woman in her late 30s"},
            {"angle": "front view, direct gaze", "expression": "determined and protective, mature features", "lighting": "soft cinematic lighting", "style": "professional character art, adult woman paladin"},
            {"angle": "front facing, head slightly tilted", "expression": "kind and caring expression, experienced face", "lighting": "golden hour lighting", "style": "high quality fantasy portrait, mature woman"},
            {"angle": "straight on portrait", "expression": "serene and wise look, adult features", "lighting": "studio lighting", "style": "detailed paladin portrait, woman in her late 30s"},
            {"angle": "front view, confident pose", "expression": "noble and compassionate, weathered by experience", "lighting": "natural daylight", "style": "rpg character portrait, mature adult paladin"},
            {"angle": "centered portrait composition", "expression": "maternal warmth, experienced woman", "lighting": "soft dramatic lighting", "style": "fantasy character concept art, late 30s woman"},
            
            # 3/4 view (5-6 images) - mature features emphasized
            {"angle": "three-quarter view from right", "expression": "gentle smile with laugh lines, protective", "lighting": "side lighting", "style": "detailed fantasy portrait, mature woman paladin"},
            {"angle": "3/4 angle from left side", "expression": "thoughtful and caring, adult features", "lighting": "soft ambient light", "style": "character concept art, woman in late 30s"},
            {"angle": "three-quarter view, looking to side", "expression": "alert and watchful, experienced", "lighting": "natural outdoor lighting", "style": "professional character portrait, mature adult"},
            {"angle": "3/4 portrait from right", "expression": "wise and calm, weathered features", "lighting": "warm cinematic lighting", "style": "high quality fantasy art, late 30s paladin"},
            {"angle": "three-quarter angle looking over shoulder", "expression": "protective glance back, mature", "lighting": "backlit with rim light", "style": "paladin character illustration, adult woman"},
            {"angle": "3/4 view portrait", "expression": "noble bearing, experienced face", "lighting": "soft directional light", "style": "detailed character art, woman late 30s"},
            
            # Side profiles (3-4 images) - mature profile
            {"angle": "side profile view from left", "expression": "strong profile, determined, mature features", "lighting": "profile lighting", "style": "character portrait, adult woman"},
            {"angle": "right side profile portrait", "expression": "regal and composed, experienced", "lighting": "dramatic side light", "style": "fantasy character art, late 30s woman"},
            {"angle": "profile view, looking into distance", "expression": "contemplative and wise, mature", "lighting": "soft natural light", "style": "detailed portrait, adult paladin"},
            {"angle": "side profile, noble bearing", "expression": "watchful and protective, weathered", "lighting": "rim lighting", "style": "high quality character illustration, mature woman"},
            
            # Action/dynamic poses (3-4 images) - strong mature paladin
            {"angle": "ready stance with shield, front angle", "expression": "determined protector, experienced warrior", "lighting": "heroic lighting", "style": "fantasy paladin action art, mature adult woman"},
            {"angle": "confident standing pose, 3/4 view", "expression": "calm and assured, seasoned", "lighting": "dramatic lighting", "style": "rpg character portrait, woman in late 30s"},
            {"angle": "protective stance, shield raised slightly", "expression": "focused and alert, veteran", "lighting": "dynamic lighting", "style": "paladin character concept art, mature woman"},
            {"angle": "standing tall and confident", "expression": "compassionate strength, experienced", "lighting": "strong directional light", "style": "detailed character portrait, adult paladin late 30s"},
        ]
        
        # Select the right number of variations
        selected_variations = variations[:num_images]
        
        prompts = []
        for i, var in enumerate(selected_variations, 1):
            # Add explicit ear descriptor to every prompt
            prompt = f"{base_description}, {var['angle']}, {var['expression']}, {var['lighting']}, {var['style']}, nearly human ears with minimal point, sharp focus, detailed features, high quality, professional artwork"
            
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
    print("Generating Olive Elmmist with HUMAN-LIKE ears")
    print("="*60)
    print("\nThis will generate AI character images for LoRA training.")
    print("AGGRESSIVE ear control: barely pointed, almost human-looking!")
    print("Emphasizing her HUMAN heritage, not elf features!")
    print("\nCost estimate:")
    print("  - 20 images × ~$0.035 per image = ~$0.70 total")
    print("\nTime estimate: 10-20 minutes\n")
    
    print("Press Enter to start, or Ctrl+C to cancel...")
    input()
    
    # Generate training set for Character 2 (Olive)
    count = generator.generate_character_training_set('character_2', num_images=20)
    
    print("\n" + "="*60)
    print("✓ TRAINING IMAGES GENERATED!")
    print("="*60)
    print(f"\nOlive Elmmist: {count} images in lora_training/character_2/")
    print("\n⚠️  IMPORTANT: Review the images!")
    print("   - Check that ears are BARELY pointed (almost human-looking)")
    print("   - Ears should look nearly human with just a slight taper")
    print("   - If ears are too long/pointed, DELETE that image")
    print("   - Do they all look like the SAME person?")
    print("   - Check the dark brown hair, hazel eyes, tanned skin")
    print("   - Look for consistency in facial features")
    print("   - Verify she looks tall and strong (paladin build)")
    print("   - Verify she looks like a mature adult (late 30s, not teenager)")
    print("   - Delete any that look too different or have wrong features")
    print("\nWhat to look for:")
    print("  ✓ Consistent facial structure across all images")
    print("  ✓ Same dark brown hair in all images")
    print("  ✓ Same hazel-olive eyes")
    print("  ✓ Tanned outdoor skin")
    print("  ✓ Tall, strong paladin build")
    print("  ✓ Mature adult features (late 30s, NOT teenage)")
    print("  ✓ BARELY pointed ears - almost human-looking!")
    print("  ✓ Various angles and expressions but SAME face")
    print("\nTarget: Keep 15-18 best images")
    print("Be ruthless - delete ANY with overly pointed ears!")
    print("\nIf satisfied with the images:")
    print("1. Keep 15-18 best images (delete any outliers)")
    print("2. Run: python prepare_lora_training.py")
    print("3. Run: python lora_trainer.py")
    print("="*60)