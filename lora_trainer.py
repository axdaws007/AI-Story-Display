import os
import json
import fal_client
import time
from pathlib import Path
from dotenv import load_dotenv

class LoRATrainer:
    def __init__(self, api_key):
        os.environ['FAL_KEY'] = api_key
        self.trained_loras = {}
        
    def train_character_lora(self, character_name, zip_file_path, trigger_word):
        """
        Train a LoRA for a character
        
        Args:
            character_name: Friendly name (e.g., "Character 1")
            zip_file_path: Path to ZIP file with training images
            trigger_word: Unique word to trigger this LoRA (e.g., "aric123")
        """
        
        print(f"\n{'='*60}")
        print(f"TRAINING LoRA: {character_name}")
        print(f"{'='*60}")
        
        if not Path(zip_file_path).exists():
            raise Exception(f"ZIP file not found: {zip_file_path}")
        
        print(f"Trigger word: {trigger_word}")
        print(f"Training images: {zip_file_path}")
        print(f"\nUploading training data...")
        
        # Upload the ZIP file
        images_url = fal_client.upload_file(zip_file_path)
        print(f"✓ Upload complete: {images_url}")
        
        print(f"\nStarting training...")
        print("This will take 10-30 minutes. You can close this window - training continues on FAL servers.")
        print("The script will wait for completion...\n")
        
        # Start training using FLUX LoRA fast trainer
        # Using 1000 steps is a good default
        result = fal_client.subscribe(
            "fal-ai/flux-lora-fast-training",
            arguments={
                "images_data_url": images_url,
                "trigger_word": trigger_word,
                "is_style": False,  # This is character training, not style
                "steps": 1000,  # Good default for character training
            },
            with_logs=True,
            on_queue_update=lambda update: self._print_progress(update)
        )
        
        # Save the trained LoRA info
        lora_url = result['diffusers_lora_file']['url']
        
        self.trained_loras[character_name] = {
            'trigger_word': trigger_word,
            'lora_url': lora_url,
            'character_name': character_name
        }
        
        print(f"\n{'='*60}")
        print(f"✓ TRAINING COMPLETE: {character_name}")
        print(f"{'='*60}")
        print(f"LoRA URL: {lora_url}")
        print(f"Trigger word: {trigger_word}")
        print(f"\nTo use in image generation, include '{trigger_word}' in your prompt")
        print(f"and provide the LoRA URL to the image generator.\n")
        
        return {
            'lora_url': lora_url,
            'trigger_word': trigger_word
        }
    
    def _print_progress(self, update):
        """Print training progress updates"""
        if hasattr(update, 'logs') and update.logs:
            for log in update.logs:
                print(f"  {log['message']}")
    
    def save_lora_config(self, output_file='lora_config.json'):
        """Save trained LoRA configuration for later use"""
        
        with open(output_file, 'w') as f:
            json.dump(self.trained_loras, f, indent=2)
        
        print(f"✓ LoRA configuration saved to: {output_file}")
        print("\nYou can use this file to generate images with your trained characters!")
    
    def test_lora(self, character_name, test_prompt=None):
        """Generate a test image using the trained LoRA"""
        
        if character_name not in self.trained_loras:
            raise Exception(f"No trained LoRA found for {character_name}")
        
        lora_info = self.trained_loras[character_name]
        trigger = lora_info['trigger_word']
        
        if test_prompt is None:
            test_prompt = f"Portrait of {trigger}, fantasy character, detailed, high quality"
        
        print(f"\nGenerating test image for {character_name}...")
        print(f"Prompt: {test_prompt}")
        
        result = fal_client.subscribe(
            "fal-ai/flux-lora",
            arguments={
                "prompt": test_prompt,
                "loras": [
                    {
                        "path": lora_info['lora_url'],
                        "scale": 1.0
                    }
                ],
                "image_size": "square_hd",
                "num_images": 1
            }
        )
        
        # Download test image
        import requests
        from PIL import Image
        from io import BytesIO
        
        image_url = result['images'][0]['url']
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        
        # Save test image
        test_dir = Path("lora_tests")
        test_dir.mkdir(exist_ok=True)
        
        output_path = test_dir / f"{character_name.lower().replace(' ', '_')}_test.png"
        img.save(output_path)
        
        print(f"✓ Test image saved: {output_path}")
        print("Check the image to verify the character looks correct!")
        
        return str(output_path)


# Example usage
if __name__ == "__main__":
    load_dotenv()
    
    api_key = os.getenv('FAL_API_KEY')
    if not api_key:
        print("ERROR: Please set FAL_API_KEY in .env file")
        exit(1)
    
    trainer = LoRATrainer(api_key)
    
    print("\n" + "="*60)
    print("FLUX LoRA TRAINER FOR CHARACTER CONSISTENCY")
    print("="*60)
    print("\nThis will train LoRAs for both characters.")
    print("Each training takes 10-30 minutes and costs ~$2-3.")
    print("\nMake sure you have:")
    print("  1. Created lora_training/character_1.zip")
    print("  2. Created lora_training/character_2.zip")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    input()
    
    # Load character names from characters.json
    with open('characters.json', 'r') as f:
        characters = json.load(f)
    
    char1_name = characters['character_1']['fantasy_name']
    char2_name = characters['character_2']['fantasy_name']
    
    # Train Character 1
    print("\n" + "="*60)
    print("STEP 1: TRAINING CHARACTER 1")
    print("="*60)
    
    # Create unique trigger words
    # These should be unique strings that don't appear in normal language
    char1_trigger = "tobias_dunsmir_tok"  # Modify based on actual character name
    char2_trigger = "olive_elmmist_tok"  # Modify based on actual character name
    
    try:
        char1_lora = trainer.train_character_lora(
            character_name=char1_name,
            zip_file_path='lora_training/character_1.zip',
            trigger_word=char1_trigger
        )
        
        # Test Character 1
        print("\nGenerating test image for Character 1...")
        trainer.test_lora(char1_name)
        
    except Exception as e:
        print(f"\nERROR training Character 1: {e}")
        print("Fix the issue and run again.")
        exit(1)
    
    # Train Character 2
    print("\n" + "="*60)
    print("STEP 2: TRAINING CHARACTER 2")
    print("="*60)
    
    try:
        char2_lora = trainer.train_character_lora(
            character_name=char2_name,
            zip_file_path='lora_training/character_2.zip',
            trigger_word=char2_trigger
        )
        
        # Test Character 2
        print("\nGenerating test image for Character 2...")
        trainer.test_lora(char2_name)
        
    except Exception as e:
        print(f"\nERROR training Character 2: {e}")
        print("You may need to train this one separately.")
    
    # Save configuration
    trainer.save_lora_config()
    
    print("\n" + "="*60)
    print("✓ ALL TRAINING COMPLETE!")
    print("="*60)
    print("\nCheck the test images in lora_tests/ folder")
    print("Do the characters look correct?")
    print("\nNext step: Update image_generator.py to use your trained LoRAs")
    print("="*60)