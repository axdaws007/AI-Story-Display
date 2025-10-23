import os
import json
import requests
import time
from pathlib import Path
from PIL import Image
from io import BytesIO

class ComfyUIImageGenerator:
    """
    Image generator using ComfyUI with multi-LoRA workflow
    Supports dynamic character LoRAs with proper conditioning separation
    """
    
    def __init__(self, comfyui_url, characters_file='characters.json', lora_config_file='lora_config.json'):
        """
        Args:
            comfyui_url: URL of ComfyUI instance (e.g., "http://192.168.1.100:8188")
            characters_file: Path to characters.json
            lora_config_file: Path to LoRA configuration
        """
        self.comfyui_url = comfyui_url.rstrip('/')
        self.api_url = f"{self.comfyui_url}/prompt"
        self.characters = self.load_characters(characters_file)
        self.loras = self.load_lora_config(lora_config_file)
        
        # Load the workflow template
        self.workflow_template = self.create_workflow_template()
        
    def load_characters(self, file_path):
        """Load character profiles"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_lora_config(self, file_path):
        """Load trained LoRA configuration"""
        if not Path(file_path).exists():
            print(f"⚠ No LoRA config found at {file_path}")
            return None
        
        with open(file_path, 'r') as f:
            loras = json.load(f)
        
        print(f"✓ Loaded LoRA config with {len(loras)} trained characters")
        for char_name, lora_info in loras.items():
            print(f"  - {char_name}: {lora_info['trigger_word']}")
        
        return loras
    
    def create_workflow_template(self):
        """
        Create the ComfyUI workflow template with placeholders
        Based on the multi-LoRA conditioning approach
        """
        return {
            "1": {
                "inputs": {
                    "ckpt_name": "__BASE_MODEL__"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "__CHAR1_LORA__",
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "lora_name": "__CHAR2_LORA__",
                    "strength_model": 0.7,
                    "strength_clip": 0.7,
                    "model": ["2", 0],
                    "clip": ["2", 1]
                },
                "class_type": "LoraLoader"
            },
            "4": {
                "inputs": {
                    "text": "__CHAR1_PROMPT__",
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "5": {
                "inputs": {
                    "text": "__CHAR2_PROMPT__",
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "6": {
                "inputs": {
                    "text": "__SCENE_PROMPT__",
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "conditioning_1": ["4", 0],
                    "conditioning_2": ["5", 0]
                },
                "class_type": "ConditioningCombine"
            },
            "8": {
                "inputs": {
                    "conditioning_1": ["7", 0],
                    "conditioning_2": ["6", 0]
                },
                "class_type": "ConditioningCombine"
            },
            "9": {
                "inputs": {
                    "width": 1024,
                    "height": 768,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "10": {
                "inputs": {
                    "text": "__NEGATIVE_PROMPT__",
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "11": {
                "inputs": {
                    "seed": "__SEED__",
                    "steps": 25,
                    "cfg": 7.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["3", 0],
                    "positive": ["8", 0],
                    "negative": ["10", 0],
                    "latent_image": ["9", 0]
                },
                "class_type": "KSampler"
            },
            "12": {
                "inputs": {
                    "samples": ["11", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "13": {
                "inputs": {
                    "filename_prefix": "__FILENAME_PREFIX__",
                    "images": ["12", 0]
                },
                "class_type": "SaveImage"
            }
        }
    
def build_character_prompts(self, scene_description):
    """
    Build separate prompts for each character based on scene description.
    Ensures explicit LoRA tags and trigger tokens are present and also
    inserts the character names into the scene description to reinforce mapping.
    """
    char1 = self.characters['character_1']
    char2 = self.characters['character_2']

    char1_lora = self.loras.get(char1['fantasy_name'])
    char2_lora = self.loras.get(char2['fantasy_name'])

    if not char1_lora or not char2_lora:
        raise Exception(f"Missing LoRAs. Have: {list(self.loras.keys())}")

    # strengths - you can tune these (0.6 - 1.0). Keep consistent between tag & LoraLoader.
    strength1 = char1_lora.get('default_strength', 0.8)
    strength2 = char2_lora.get('default_strength', 0.75)

    # Build visible details
    def visual_summary(ch):
        parts = []
        if 'visual_design' in ch:
            v = ch['visual_design']
            if v.get('hair'): parts.append(v['hair'])
            if v.get('eyes'): parts.append(v['eyes'])
            if v.get('skin'): parts.append(v['skin'])
        parts.append(f"{ch['race']}")
        parts.append(f"{ch['class']}")
        return ", ".join([p for p in parts if p])

    char1_details = visual_summary(char1)
    char2_details = visual_summary(char2)

    # IMPORTANT: include both the LoRA tag and its trigger word token (if specified).
    # e.g. "<lora:ElfRanger.safetensors:0.8> arannis"
    lora_tag1 = f"<lora:{char1_lora['lora_filename']}:{strength1}>"
    lora_tag2 = f"<lora:{char2_lora['lora_filename']}:{strength2}>"

    trigger1 = char1_lora.get('trigger_word', char1['fantasy_name'])
    trigger2 = char2_lora.get('trigger_word', char2['fantasy_name'])

    # Character prompts — include pose/action hints, name token, and LoRA tag
    # Keep focused: name, tag, role, pose/action, visual summary
    char1_prompt = (
        f"{lora_tag1} {trigger1}, {char1['name']}, {char1_details}, "
        "kneeling to examine a flower, serene expression, green and brown leather armor, "
        "clear face, consistent character design"
    )

    char2_prompt = (
        f"{lora_tag2} {trigger2}, {char2['name']}, {char2_details}, "
        "standing with one hand on hip, scowling down at boots, dark blue and silver clothing, "
        "clear face, consistent character design"
    )

    # Scene prompt: include names + triggers to anchor them in the scene and preserve spatial cues.
    # Keep the original scene_description but explicitly add "Olive Elmmist (trigger1) on the right" etc.
    scene_prompt = (
        f"{scene_description}. "
        f"{char1['name']} ({trigger1}) on the right, {char2['name']} ({trigger2}) on the left. "
        "Dappled sunlight through forest, cinematic fantasy illustration, high detail, both characters clearly visible"
    )

    # Short, focused negative prompt for debugging — expand later
    negative_prompt = (
        "low quality, blurry, extra people, crowd, text, watermark, signature, mutated anatomy, bad hands"
    )

    return char1_prompt, char2_prompt, scene_prompt, negative_prompt
    
    def prepare_workflow(self, scene_description, page_num, story_id, base_model):
        """Prepare workflow by filling in placeholders"""
        
        # Build prompts
        char1_prompt, char2_prompt, scene_prompt, negative_prompt = self.build_character_prompts(scene_description)
        
        # Get LoRA info
        char1 = self.characters['character_1']
        char2 = self.characters['character_2']
        char1_lora = self.loras.get(char1['fantasy_name'])
        char2_lora = self.loras.get(char2['fantasy_name'])
        
        # Create workflow copy
        workflow = json.loads(json.dumps(self.workflow_template))
        
        # Fill in placeholders - proper way without string replacement
        # This avoids JSON encoding issues with special characters
        
        def replace_in_dict(obj, replacements):
            """Recursively replace placeholders in a dictionary"""
            if isinstance(obj, dict):
                return {k: replace_in_dict(v, replacements) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_dict(item, replacements) for item in obj]
            elif isinstance(obj, str):
                result = obj
                for placeholder, value in replacements.items():
                    result = result.replace(placeholder, str(value))
                return result
            else:
                return obj

        replacements = {
            "__BASE_MODEL__": base_model,
            "__CHAR1_LORA__": char1_lora['lora_filename'],
            "__CHAR2_LORA__": char2_lora['lora_filename'],
            "__CHAR1_PROMPT__": char1_prompt.replace('\n', ' ').replace('\r', ' '),
            "__CHAR2_PROMPT__": char2_prompt.replace('\n', ' ').replace('\r', ' '),
            "__SCENE_PROMPT__": scene_prompt.replace('\n', ' ').replace('\r', ' '),
            "__NEGATIVE_PROMPT__": negative_prompt.replace('\n', ' ').replace('\r', ' '),
            "__SEED__": str(int(time.time() * 1000) % 1000000),
            "__FILENAME_PREFIX__": f"{story_id}/page_{page_num:02d}"
        }
        
        return replace_in_dict(workflow, replacements)
    
    def queue_prompt(self, workflow):
        """Send workflow to ComfyUI"""
        payload = {"prompt": workflow}
        response = requests.post(self.api_url, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"ComfyUI API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def wait_for_completion(self, prompt_id, timeout=300):
        """Wait for ComfyUI to finish generating"""
        start_time = time.time()
        history_url = f"{self.comfyui_url}/history/{prompt_id}"
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(history_url)
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history and history[prompt_id].get('status', {}).get('completed', False):
                        return history[prompt_id]
            except:
                pass
            
            time.sleep(2)
        
        raise Exception(f"Timeout waiting for generation to complete (>{timeout}s)")
    
    def download_image(self, filename, subfolder, output_folder, local_path):
        """Download generated image from ComfyUI"""
        # ComfyUI saves to output folder with subfolder structure
        url = f"{self.comfyui_url}/view"
        params = {
            'filename': filename,
            'subfolder': subfolder,
            'type': 'output'
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            
            # Resize for e-ink display (800x480)
            img = img.resize((800, 480), Image.Resampling.LANCZOS)
            
            # Save locally
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(local_path)
            return local_path
        else:
            raise Exception(f"Failed to download image: {response.status_code}")
    
    def generate_story_image(self, scene_description, page_num, story_id, output_folder, base_model):
        """Generate a single story image"""
        
        # Prepare workflow
        workflow = self.prepare_workflow(scene_description, page_num, story_id, base_model)
        
        # Queue the job
        print(f"  Sending workflow to ComfyUI...")
        result = self.queue_prompt(workflow)
        prompt_id = result['prompt_id']
        print(f"  Workflow queued with ID: {prompt_id}")
        
        # Wait for completion
        print(f"  Waiting for ComfyUI to finish rendering...")
        history = self.wait_for_completion(prompt_id)
        
        # Get the output filename
        outputs = history.get('outputs', {})
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for img_info in node_output['images']:
                    filename = img_info['filename']
                    subfolder = img_info.get('subfolder', '')
                    
                    # Download image
                    local_path = f"images/{story_id}/page_{page_num:02d}.png"
                    self.download_image(filename, subfolder, output_folder, local_path)
                    
                    print(f"  ✓ Generated: {local_path}")
                    return local_path
        
        raise Exception("No image output found in ComfyUI response")
    
    def generate_all_story_images(self, story_file, output_folder, base_model):
        """Generate images for all pages in a story"""
        with open(story_file, 'r') as f:
            story_data = json.load(f)
        
        story_id = Path(story_file).stem
        images = []
        
        if not self.loras:
            print("\n⚠️  ERROR: No LoRAs loaded!")
            return []
        
        print(f"\n✓ Using trained LoRAs for character consistency")
        print(f"\nGenerating {len(story_data['pages'])} images...")
        print("This will take 10-15 minutes...\n")
        
        for page in story_data['pages']:
            print(f"Page {page['page']}/{len(story_data['pages'])}...")
            print(f"  Scene: {page['scene_description'][:80]}...")
            
            try:
                img_path = self.generate_story_image(
                    page['scene_description'],
                    page['page'],
                    story_id,
                    output_folder,
                    base_model
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
    
    comfyui_url = os.getenv('COMFYUI_URL')
    if not comfyui_url:
        print("ERROR: Please set COMFYUI_URL in .env file")
        exit(1)
    
    generator = ComfyUIImageGenerator(comfyui_url)
    
    # Test with your story
    story_file = 'story_20251019_184004.json'
    output_folder = os.getenv('COMFYUI_OUTPUT_FOLDER', '/workspace/ComfyUI/output')
    base_model = os.getenv('BASE_MODEL', 'sd_xl_base_1.0.safetensors')
    
    generator.generate_all_story_images(story_file, output_folder, base_model)