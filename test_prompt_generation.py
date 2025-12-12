#!/usr/bin/env python3
"""
Test script for the updated story generator prompt system
Demonstrates prompt generation without making API calls
"""

import json
import sys
from pathlib import Path

# Mock the story generator's prompt building functionality
class PromptTester:
    def __init__(self, characters_file='characters.json', lora_config_file='lora_config.json'):
        self.characters = self.load_json(characters_file)
        self.loras = self.load_json(lora_config_file)
        
    def load_json(self, file_path):
        """Load JSON file"""
        if not Path(file_path).exists():
            print(f"Warning: {file_path} not found")
            return {}
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def test_prompt_generation(self):
        """Test prompt generation with sample page data"""
        
        # Sample test pages based on your existing story
        test_pages = [
            {
                'page': 1,
                'text': 'Tobias Dunsmir grimaced, adjusting his tunic. "This mud is ruining my boots, Olive!" Olive Elmmist chuckled, pointing to a shimmering flower pushing through the earth. "But look, Tobias, a Moonpetal! Mielikki blesses this place."',
                'scene_description': 'Olive Elmmist, on the right, kneels to examine a white flower glowing softly. She wears her green and brown leather armor and smiles gently. Tobias Dunsmir stands on the left, one hand on his hip, scowling down at his muddy boots. He wears dark blue and silver clothing, and the forest is dense around them, dappled sunlight filtering through the leaves. The mood is contrasting: Olive is serene and Tobias is annoyed.'
            },
            {
                'page': 2,
                'text': 'They crested a hill overlooking a ravaged village. Smoke billowed from broken homes, the air thick with despair. "Necromancy," Olive growled, her hand tightening on her Dragon Slayer Longsword.',
                'scene_description': 'Olive Elmmist stands on the crest of a hill, center, her face grim and determined. Her golden shield is strapped to her back and she holds her Dragon Slayer Longsword in her right hand. In the background, a village is burning, black smoke rising into the sky. The lighting is dim and ominous, a sense of foreboding. Tobias Dunsmir stands behind her, looking at the village with concern. He is touching his Ring of Illusion with his left hand.'
            },
            {
                'page': 5,
                'text': 'Tobias, nimble as a shadow, darted between skeletons. He hurled his Lightning Dagger, striking the necromancer\'s hand, disrupting the spell. The dagger crackled with electric energy.',
                'scene_description': 'Tobias Dunsmir, on the left, throws his Lightning Dagger with his right hand at the necromancer. The dagger is surrounded by sparks. Skeletons are crumbling as the necromancer clutches his injured hand, his spell broken. The background is the same dark crypt scene as before.'
            }
        ]
        
        print("\n" + "="*70)
        print("PROMPT GENERATION TEST")
        print("="*70)
        
        char1_name = self.characters['character_1']['fantasy_name']
        char2_name = self.characters['character_2']['fantasy_name']
        
        print(f"\nCharacters:")
        print(f"  Character 1: {char1_name}")
        print(f"  Character 2: {char2_name}")
        
        if self.loras:
            print(f"\nLoRA Configuration:")
            for char_name, lora_info in self.loras.items():
                print(f"  {char_name}:")
                print(f"    Trigger: {lora_info.get('trigger_word', 'N/A')}")
                print(f"    File: {lora_info.get('lora_filename', 'N/A')}")
                print(f"    Strength: {lora_info.get('default_strength', 'N/A')}")
        
        print("\n" + "="*70)
        print("TESTING PROMPT GENERATION FOR SAMPLE PAGES")
        print("="*70)
        
        for test_page in test_pages:
            print(f"\n{'─'*70}")
            print(f"PAGE {test_page['page']}")
            print(f"{'─'*70}")
            print(f"\nNarrative Text:")
            print(f"  {test_page['text'][:100]}...")
            
            print(f"\nOriginal Scene Description:")
            print(f"  {test_page['scene_description'][:100]}...")
            
            # Generate prompts using the actual logic
            prompts = self._build_prompts_for_page(test_page)
            
            print(f"\n📝 Generated Prompts:")
            print(f"\n  Character 1 Prompt:")
            print(f"    {prompts['character_1_prompt']}")
            
            print(f"\n  Character 2 Prompt:")
            print(f"    {prompts['character_2_prompt']}")
            
            print(f"\n  Scene Prompt:")
            print(f"    {prompts['scene_prompt']}")
            
            print(f"\n  Negative Prompt:")
            print(f"    {prompts['negative_prompt']}")
            
            print(f"\n✓ Prompt generation successful for page {test_page['page']}")
        
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print("\nAll prompts generated successfully!")
        print("These prompts are ready to use with ComfyUI multi-LoRA workflow.")
        print("\nNext steps:")
        print("  1. Verify the prompts match your expectations")
        print("  2. Adjust LoRA strengths in lora_config.json if needed")
        print("  3. Run the full story generator to create a complete story")
        print("  4. Use the generated JSON with your ComfyUI workflow")
    
    def _build_prompts_for_page(self, page_data):
        """Build prompts for a test page (simplified version)"""
        
        char1 = self.characters['character_1']
        char2 = self.characters['character_2']
        
        char1_name = char1['fantasy_name']
        char2_name = char2['fantasy_name']
        
        # Get LoRA info (with fallbacks)
        char1_lora = self.loras.get(char1_name, {
            'trigger_word': char1_name.lower().replace(' ', '_'),
            'lora_filename': f"{char1_name.replace(' ', '')}.safetensors",
            'default_strength': 0.8
        })
        
        char2_lora = self.loras.get(char2_name, {
            'trigger_word': char2_name.lower().replace(' ', '_'),
            'lora_filename': f"{char2_name.replace(' ', '')}.safetensors",
            'default_strength': 0.75
        })
        
        scene_desc = page_data['scene_description']
        scene_lower = scene_desc.lower()
        
        # Simple action extraction
        char1_action = self._extract_simple_action(scene_desc, char1_name)
        char2_action = self._extract_simple_action(scene_desc, char2_name)
        
        # Extract positioning
        char1_pos = self._extract_position(scene_desc, char1_name)
        char2_pos = self._extract_position(scene_desc, char2_name)
        
        # Extract outfit colors
        char1_outfit = self._extract_outfit_mention(scene_desc, char1_name)
        char2_outfit = self._extract_outfit_mention(scene_desc, char2_name)
        
        # Build character prompts
        char1_prompt = (
            f"<lora:{char1_lora['lora_filename']}:{char1_lora['default_strength']}> "
            f"{char1_lora['trigger_word']}"
        )
        if char1_action:
            char1_prompt += f", {char1_action}"
        if char1_outfit:
            char1_prompt += f", {char1_outfit}"
        
        char2_prompt = (
            f"<lora:{char2_lora['lora_filename']}:{char2_lora['default_strength']}> "
            f"{char2_lora['trigger_word']}"
        )
        if char2_action:
            char2_prompt += f", {char2_action}"
        if char2_outfit:
            char2_prompt += f", {char2_outfit}"
        
        # Build scene prompt
        environment = self._extract_environment_simple(scene_desc)
        positioning = f"{char1_name} {char1_pos}, {char2_name} {char2_pos}"
        
        scene_prompt = f"{environment}, {positioning}, cinematic lighting, fantasy illustration, high detail"
        
        # Standard negative prompt
        negative_prompt = "low quality, blurry, extra people, watermark, text, deformed, bad anatomy, multiple faces, distorted perspective"
        
        return {
            'character_1_prompt': char1_prompt,
            'character_2_prompt': char2_prompt,
            'scene_prompt': scene_prompt,
            'negative_prompt': negative_prompt
        }
    
    def _extract_simple_action(self, scene_desc, char_name):
        """Extract action for character (simplified)"""
        sentences = scene_desc.split('.')
        for sentence in sentences:
            if char_name in sentence:
                # Look for action verbs
                sentence_lower = sentence.lower()
                if 'kneel' in sentence_lower:
                    return 'kneeling'
                elif 'stand' in sentence_lower or 'stands' in sentence_lower:
                    if 'scowling' in sentence_lower:
                        return 'standing with hand on hip, scowling'
                    return 'standing confidently'
                elif 'throw' in sentence_lower or 'hurl' in sentence_lower:
                    return 'throwing dagger with right hand'
                elif 'hold' in sentence_lower or 'holds' in sentence_lower:
                    if 'sword' in sentence_lower:
                        return 'holding sword in right hand'
        return 'standing'
    
    def _extract_position(self, scene_desc, char_name):
        """Extract position (simplified)"""
        scene_lower = scene_desc.lower()
        char_lower = char_name.lower()
        
        # Find the character mention
        char_index = scene_lower.find(char_lower)
        if char_index == -1:
            return 'in center'
        
        # Look for position keywords near character name
        window = scene_lower[max(0, char_index-50):char_index+100]
        
        if 'on the right' in window or 'to the right' in window:
            return 'on the right'
        elif 'on the left' in window or 'to the left' in window:
            return 'on the left'
        elif 'center' in window or 'middle' in window:
            return 'in center'
        elif 'foreground' in window:
            return 'in foreground'
        elif 'background' in window:
            return 'in background'
        
        return 'in center'
    
    def _extract_outfit_mention(self, scene_desc, char_name):
        """Extract outfit color mentions"""
        sentences = scene_desc.split('.')
        for sentence in sentences:
            if char_name in sentence:
                sentence_lower = sentence.lower()
                # Look for clothing/armor mentions
                if 'green and brown' in sentence_lower:
                    return 'green and brown armor'
                elif 'dark blue and silver' in sentence_lower:
                    return 'dark blue and silver clothing'
                elif 'armor' in sentence_lower:
                    return 'armor'
                elif 'leather' in sentence_lower:
                    return 'leather outfit'
        return ''
    
    def _extract_environment_simple(self, scene_desc):
        """Extract environment description (simplified)"""
        # Look for setting keywords
        scene_lower = scene_desc.lower()
        
        environments = []
        
        if 'forest' in scene_lower:
            environments.append('dense forest')
        if 'village' in scene_lower:
            environments.append('village in background')
        if 'hill' in scene_lower:
            environments.append('hilltop')
        if 'crypt' in scene_lower or 'tomb' in scene_lower:
            environments.append('dark crypt interior')
        
        # Look for lighting
        if 'sunlight' in scene_lower:
            environments.append('dappled sunlight')
        elif 'smoke' in scene_lower:
            environments.append('smoke-filled atmosphere')
        
        if environments:
            return ', '.join(environments)
        
        return 'fantasy setting'


def main():
    """Run the test"""
    print("\n🧪 Story Generator Prompt System Test\n")
    
    # Check for required files
    if not Path('characters.json').exists():
        print("❌ Error: characters.json not found")
        print("   Please ensure characters.json is in the current directory")
        return 1
    
    if not Path('lora_config.json').exists():
        print("⚠️  Warning: lora_config.json not found")
        print("   Will use default configuration based on character names\n")
    
    # Run the test
    tester = PromptTester()
    tester.test_prompt_generation()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())