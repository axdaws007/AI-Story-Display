import zipfile
import os
from pathlib import Path

def create_training_zip(character_folder, output_zip):
    """Create a ZIP file of training images"""
    
    folder_path = Path(character_folder)
    
    if not folder_path.exists():
        print(f"ERROR: Folder {character_folder} does not exist!")
        return False
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {character_folder}")
        return False
    
    print(f"Found {len(image_files)} images in {character_folder}")
    
    # Create ZIP file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_file in image_files:
            # Add file to zip with just the filename (no folder structure)
            zipf.write(img_file, img_file.name)
            print(f"  Added: {img_file.name}")
    
    print(f"✓ Created: {output_zip}")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("LoRA TRAINING DATA PREPARATION")
    print("=" * 60)
    
    # Create ZIPs for both characters
    success1 = create_training_zip(
        'lora_training/character_1',
        'lora_training/character_1.zip'
    )
    
    print()
    
    success2 = create_training_zip(
        'lora_training/character_2',
        'lora_training/character_2.zip'
    )
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✓ Both training datasets ready!")
        print("\nNext step: Run lora_trainer.py to train your LoRAs")
    else:
        print("⚠ Some datasets are not ready. Check the errors above.")
    print("=" * 60)