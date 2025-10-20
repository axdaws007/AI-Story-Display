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
        print(f"This will take 10-15 minutes and cost ~$0.70-1.00\n")
        
        # Create output directory
        output_dir = Path(f"lora_training/{character_key}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build base character description
        base_description = self._build_character_description(char)
        
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
                        "negative_prompt": "extremely pointed ears, very long ears, exaggerated elf ears",
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
                continue
        
        print(f"\n{'='*60}")
        print(f"✓ Generated {generated_count}/{len(prompts)} images")
        print(f"✓ Saved to: {output_dir}")
        print(f"{'='*60}\n")
        
        return generated_count
    
    def _build_character_description(self, char):
        """Build detailed character description from profile"""
        visual = char.get('visual_design', char.get('appearance', {}))
        
        description = f"{char['fantasy_name']}, {char.get('race', 'human')} {char.get('class', 'adventurer')}"
        
        # Add physical details
        if isinstance(visual, dict):
            if 'face' in visual:
                description += f", {visual['face']}"
            if 'hair' in visual:
                description += f", {visual['hair']}"
            if 'eyes' in visual:
                description += f", {visual['eyes']}"
            if 'build' in visual:
                description += f", {visual['build']}"
            if 'skin' in visual:
                description += f", {visual['skin']}"
            if 'distinctive' in visual:
                description += f", {visual['distinctive']}"
            if 'typical_outfit' in visual:
                description += f", wearing {visual['typical_outfit']}"
        
        return description
    
    def _generate_varied_prompts(self, base_description, num_images):
        """Generate varied prompts for different angles, expressions, and scenarios"""
        
        variations = [
            # Front-facing portraits (5-6 images)
            {"angle": "front view, looking at camera", "expression": "slight smile", "lighting": "soft natural lighting"},
            {"angle": "front view, direct gaze", "expression": "serious expression", "lighting": "dramatic lighting"},
            {"angle": "front view, head slightly tilted", "expression": "gentle smile", "lighting": "golden hour lighting"},
            {"angle": "front view, confident pose", "expression": "determined look", "lighting": "bright daylight"},
            {"angle": "front view portrait", "expression": "thoughtful expression", "lighting": "studio lighting"},
            {"angle": "front view, neutral expression", "expression": "calm demeanor", "lighting": "even lighting"},
            
            # 3/4 view (5-6 images)
            {"angle": "three-quarter view", "expression": "slight smile", "lighting": "side lighting"},
            {"angle": "three-quarter angle, looking to the side", "expression": "contemplative", "lighting": "soft lighting"},
            {"angle": "3/4 view from right side", "expression": "friendly smile", "lighting": "natural daylight"},
            {"angle": "three-quarter view from left", "expression": "focused expression", "lighting": "dramatic shadows"},
            {"angle": "3/4 angle looking over shoulder", "expression": "confident look", "lighting": "backlit"},
            {"angle": "three-quarter portrait", "expression": "serene expression", "lighting": "warm lighting"},
            
            # Side profiles (3-4 images)
            {"angle": "side profile view", "expression": "neutral expression", "lighting": "profile lighting"},
            {"angle": "profile from left side", "expression": "thoughtful look", "lighting": "rim lighting"},
            {"angle": "right side profile", "expression": "calm demeanor", "lighting": "soft side light"},
            {"angle": "profile view, looking distance", "expression": "observant", "lighting": "natural light"},
            
            # Action poses (3-4 images)
            {"angle": "dynamic pose, front angle", "expression": "determined", "lighting": "action lighting"},
            {"angle": "standing confidently", "expression": "ready for adventure", "lighting": "heroic lighting"},
            {"angle": "in motion, 3/4 view", "expression": "focused", "lighting": "dramatic lighting"},
            {"angle": "action stance", "expression": "alert expression", "lighting": "dynamic shadows"},
        ]
        
        # Select the right number of variations
        selected_variations = variations[:num_images]
        
        prompts = []
        for i, var in enumerate(selected_variations, 1):
            prompt = f"{base_description}, {var['angle']}, {var['expression']}, {var['lighting']}, high quality portrait, professional photography, detailed, sharp focus, fantasy character art"
            
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
    print("="*60)
    print("\nThis will generate AI character images for LoRA training.")
    print("No real photos needed - creates consistent fantasy characters!")
    print("\nCost estimate:")
    print("  - 20 images per character × 2 characters = 40 images")
    print("  - ~$0.035 per image = ~$1.40 total")
    print("\nTime estimate: 20-30 minutes total\n")
    
    print("Press Enter to start, or Ctrl+C to cancel...")
    input()
    
    # Generate training set for Character 1
    print("\n" + "="*60)
    print("GENERATING CHARACTER 1")
    print("="*60)
    
    count1 = generator.generate_character_training_set('character_1', num_images=20)
    
    print("\nCharacter 1 complete! Check the images in lora_training/character_1/")
    print("Press Enter to continue to Character 2...")
    input()
    
    # Generate training set for Character 2
    print("\n" + "="*60)
    print("GENERATING CHARACTER 2")
    print("="*60)
    
    count2 = generator.generate_character_training_set('character_2', num_images=20)
    
    print("\n" + "="*60)
    print("✓ ALL TRAINING IMAGES GENERATED!")
    print("="*60)
    print(f"\nCharacter 1: {count1} images in lora_training/character_1/")
    print(f"Character 2: {count2} images in lora_training/character_2/")
    print("\n⚠️  IMPORTANT: Review the images!")
    print("   - Do the characters look consistent across images?")
    print("   - Do they match your vision?")
    print("   - Delete any bad/inconsistent images")
    print("\nIf satisfied, next step:")
    print("1. Run: python prepare_lora_training.py  (to create ZIPs)")
    print("2. Run: python lora_trainer.py  (to train LoRAs)")
    print("="*60)