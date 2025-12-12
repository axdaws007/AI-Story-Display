import os
import json
import requests
from datetime import datetime
from pathlib import Path

class StoryGenerator:
    def __init__(self, api_key, characters_file='characters.json', lora_config_file='lora_config.json'):
        self.api_key = api_key
        self.characters = self.load_characters(characters_file)
        self.loras = self.load_lora_config(lora_config_file)
        
    def load_characters(self, file_path):
        """Load character profiles from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def load_lora_config(self, file_path):
        """Load LoRA configuration with trigger words and strengths"""
        if not Path(file_path).exists():
            print(f"⚠ No LoRA config found at {file_path}")
            print("   Using default configuration")
            # Create default config based on character names
            char1_name = self.characters['character_1']['fantasy_name']
            char2_name = self.characters['character_2']['fantasy_name']
            return {
                char1_name: {
                    'trigger_word': char1_name.lower().replace(' ', '_'),
                    'lora_filename': f"{char1_name.replace(' ', '')}.safetensors",
                    'default_strength': 0.8
                },
                char2_name: {
                    'trigger_word': char2_name.lower().replace(' ', '_'),
                    'lora_filename': f"{char2_name.replace(' ', '')}.safetensors",
                    'default_strength': 0.75
                }
            }
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def format_characters_for_prompt(self):
        """Format character data for the LLM prompt"""
        char_text = "CHARACTER PROFILES:\n\n"
        
        # Character 1
        char1 = self.characters.get('character_1', {})
        char_text += f"CHARACTER 1: {char1.get('fantasy_name', 'Character 1')}\n"
        char_text += f"Class: {char1.get('class', 'Adventurer')}\n"
        char_text += f"Race: {char1.get('race', 'Human')}\n"
        char_text += f"Age: {char1.get('age', 'Unknown')} years old\n"
        
        # Add personality
        if 'personality' in char1 and char1['personality']:
            char_text += f"Personality: {', '.join(char1['personality'])}\n"
        
        # Add background
        if 'background' in char1:
            char_text += f"Background: {char1['background']}\n"
        
        # Add key equipment
        if 'equipment' in char1:
            weapons = char1['equipment'].get('weapons', [])
            if weapons:
                char_text += f"Weapons: {', '.join(weapons)}\n"
        
        # Add goals
        if 'goals' in char1:
            char_text += f"Goals: {char1['goals']}\n"
        
        # Add quirks
        if 'quirks' in char1:
            char_text += f"Quirks: {', '.join(char1['quirks'])}\n"
        
        char_text += "\n"
        
        # Character 2
        char2 = self.characters.get('character_2', {})
        char_text += f"CHARACTER 2: {char2.get('fantasy_name', 'Character 2')}\n"
        char_text += f"Class: {char2.get('class', 'Adventurer')}\n"
        char_text += f"Race: {char2.get('race', 'Human')}\n"
        
        # Handle age appearance for half-elves
        if 'age_appearance' in char2:
            char_text += f"Appears: {char2.get('age_appearance', '')}\n"
        else:
            char_text += f"Age: {char2.get('age', 'Unknown')} years old\n"
        
        # Add personality
        if 'personality' in char2 and char2['personality']:
            char_text += f"Personality: {', '.join(char2['personality'])}\n"
        
        # Add background
        if 'background' in char2:
            char_text += f"Background: {char2['background']}\n"
        
        # Add deity if present (for paladins, clerics, etc)
        if 'deity' in char2:
            char_text += f"Deity: {char2['deity']}\n"
        
        # Add key equipment
        if 'equipment' in char2:
            weapons = char2['equipment'].get('weapons', [])
            if weapons:
                char_text += f"Weapons: {', '.join(weapons)}\n"
        
        # Add goals
        if 'goals' in char2:
            char_text += f"Goals: {char2['goals']}\n"
        
        # Add quirks
        if 'quirks' in char2:
            char_text += f"Quirks: {', '.join(char2['quirks'])}\n"
        
        char_text += "\n"
        
        # Add relationship dynamic
        if 'relationship' in self.characters:
            char_text += f"RELATIONSHIP DYNAMIC:\n{self.characters['relationship']}\n"
        
        return char_text
    
    def build_page_prompts(self, page_data):
        """
        Build separate prompts for each character and the scene based on page data.
        
        Args:
            page_data: Dictionary containing 'page', 'text', and 'scene_description'
        
        Returns:
            Dictionary with character_1_prompt, character_2_prompt, scene_prompt, negative_prompt
        """
        char1 = self.characters['character_1']
        char2 = self.characters['character_2']
        
        char1_name = char1['fantasy_name']
        char2_name = char2['fantasy_name']
        
        # Get LoRA information
        char1_lora = self.loras.get(char1_name, {})
        char2_lora = self.loras.get(char2_name, {})
        
        char1_trigger = char1_lora.get('trigger_word', char1_name.lower().replace(' ', '_'))
        char2_trigger = char2_lora.get('trigger_word', char2_name.lower().replace(' ', '_'))
        
        char1_lora_file = os.path.splitext(char1_lora.get('lora_filename', f"{char1_name.replace(' ', '')}.safetensors"))[0]
        char2_lora_file = os.path.splitext(char2_lora.get('lora_filename', f"{char2_name.replace(' ', '')}.safetensors"))[0]
        
        char1_strength = char1_lora.get('default_strength', 0.8)
        char2_strength = char2_lora.get('default_strength', 0.75)
        
        # Get visual design information for outfit cues
        char1_visual = char1.get('visual_design', {})
        char2_visual = char2.get('visual_design', {})
        
        char1_outfit_cue = self._extract_outfit_colors(char1_visual.get('prompt_palette', ''))
        char2_outfit_cue = self._extract_outfit_colors(char2_visual.get('prompt_palette', ''))
        
        char1_prompt_keywords = self._extract_outfit_colors(char1_visual.get('prompt_keywords', ''))
        char2_prompt_keywords = self._extract_outfit_colors(char2_visual.get('prompt_keywords', ''))
        
        # Extract action/pose information from scene_description and text
        scene_desc = page_data.get('scene_description', '')

        scene_desc = scene_desc.replace(". She ", f". {char2_name} ")
        scene_desc = scene_desc.replace(". He ", f". {char1_name} ")
        scene_desc = scene_desc.replace(". Her ", f". {char2_name}'s ")
        scene_desc = scene_desc.replace(". His ", f". {char1_name}'s ")

        page_text = page_data.get('text', '')
        
        # Parse scene description to extract character-specific actions
        char1_action, char2_action, environment_desc, positioning = self._parse_scene_description(
            scene_desc, page_text, char1_name, char2_name
        )
        
        # Build character prompts
        character_1_prompt = self._build_character_prompt(
            char1_lora_file, char1_strength, char1_trigger, 
            char1_name, char1_action, char1_outfit_cue, char1_prompt_keywords
        )
        
        character_2_prompt = self._build_character_prompt(
            char2_lora_file, char2_strength, char2_trigger,
            char2_name, char2_action, char2_outfit_cue, char2_prompt_keywords
        )
        
        # Build scene prompt
        scene_prompt = self._build_scene_prompt(
            environment_desc, positioning, char1_name, char2_name
        )
        
        # Standard negative prompt
        negative_prompt = "low quality, blurry, extra people, watermark, text, deformed, bad anatomy, multiple faces, distorted perspective"
        
        return {
            'character_1_prompt': character_1_prompt,
            'character_2_prompt': character_2_prompt,
            'scene_prompt': scene_prompt,
            'negative_prompt': negative_prompt
        }
    
    def _extract_outfit_colors(self, outfit_description):
        """Extract color cues from outfit description"""
        # Common color words to extract
        colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'purple', 
                  'brown', 'grey', 'gray', 'silver', 'gold', 'golden', 'dark', 
                  'light', 'deep', 'bright']
        
        outfit_lower = outfit_description.lower()
        found_colors = [color for color in colors if color in outfit_lower]
        
        if found_colors:
            return ', '.join(found_colors) + ' clothing'
        return ''
    
    def _parse_scene_description(self, scene_desc, page_text, char1_name, char2_name):
        """
        Parse the scene description to extract character actions and environment.
        
        Returns:
            tuple: (char1_action, char2_action, environment_desc, positioning)
        """
        # This is a simplified parser - in production, you might use more sophisticated NLP
        scene_lower = scene_desc.lower()
        text_lower = page_text.lower()
        combined = scene_lower + ' ' + text_lower
        
        char1_lower = char1_name.lower()
        char2_lower = char2_name.lower()
        
        # Extract actions by finding sentences containing character names
        char1_action = self._extract_character_action(combined, char1_lower, char1_name)
        char2_action = self._extract_character_action(combined, char2_lower, char2_name)
        
        # Extract positioning information
        positioning = self._extract_positioning(scene_desc, char1_name, char2_name)
        
        # Extract environment description (everything that's not character-specific)
        environment_desc = self._extract_environment(scene_desc, char1_name, char2_name)
        
        return char1_action, char2_action, environment_desc, positioning
    
    def _extract_character_action(self, text, char_name_lower, char_name):
        """Extract action description for a specific character"""
        # Look for sentences containing the character's name
        sentences = text.split('.')
        actions = []
        
        for sentence in sentences:
            if char_name_lower in sentence.lower():
                # Extract key action words
                sentence = sentence.strip()
                # Remove character name for cleaner prompt
                sentence = sentence.replace(char_name, '').strip()
                if sentence and len(sentence) > 5:
                    actions.append(sentence)
        
        if actions:
            return ', '.join(actions[:2])  # Use up to 2 action descriptions
        return 'standing confidently'
    
    def _extract_positioning(self, scene_desc, char1_name, char2_name):
        """Extract positioning information (left/right, foreground/background)"""
        scene_lower = scene_desc.lower()
        
        position_map = {
            char1_name: '',
            char2_name: ''
        }
        
        # Check for explicit positioning
        for char_name in [char1_name, char2_name]:
            char_lower = char_name.lower()
            if 'on the left' in scene_lower and char_lower in scene_lower[:scene_lower.find('on the left') + 20]:
                position_map[char_name] = 'on the left'
            elif 'on the right' in scene_lower and char_lower in scene_lower[:scene_lower.find('on the right') + 20]:
                position_map[char_name] = 'on the right'
            elif 'center' in scene_lower and char_lower in scene_lower[:scene_lower.find('center') + 20]:
                position_map[char_name] = 'in the center'
            elif 'foreground' in scene_lower and char_lower in scene_lower[:scene_lower.find('foreground') + 20]:
                position_map[char_name] = 'in the foreground'
            elif 'background' in scene_lower and char_lower in scene_lower[:scene_lower.find('background') + 20]:
                position_map[char_name] = 'in the background'
        
        # Default positioning if not specified
        if not position_map[char1_name]:
            position_map[char1_name] = 'on the right'
        if not position_map[char2_name]:
            position_map[char2_name] = 'on the left'
        
        return f"{char1_name} {position_map[char1_name]}, {char2_name} {position_map[char2_name]}"
    
    def _extract_environment(self, scene_desc, char1_name, char2_name):
        """Extract environment/setting description without character-specific details"""
        # Remove character names and their immediate context
        env_desc = scene_desc
        
        # List of phrases to remove that are character-specific
        char_phrases = [
            char1_name, char2_name,
            'he ', 'she ', 'his ', 'her ',
            'stands', 'sitting', 'kneeling', 'crouching',
            'holding', 'wielding', 'wearing'
        ]
        
        # Extract setting-related phrases
        setting_keywords = [
            'forest', 'clearing', 'village', 'temple', 'ruins', 'cave',
            'lighting', 'sunlight', 'shadows', 'atmosphere', 'mood',
            'cinematic', 'illustration', 'fantasy', 'dramatic'
        ]
        
        # Simple extraction - keep sentences with setting keywords
        sentences = scene_desc.split('.')
        env_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            has_setting = any(keyword in sentence_lower for keyword in setting_keywords)
            has_character = char1_name.lower() in sentence_lower or char2_name.lower() in sentence_lower
            
            if has_setting or not has_character:
                env_sentences.append(sentence.strip())
        
        if env_sentences:
            return '. '.join(env_sentences)
        
        # Fallback
        return 'fantasy illustration, cinematic lighting, high detail'
    
    def _build_character_prompt(self, lora_file, strength, trigger_word, char_name, action, outfit_cue, prompt_keywords):
        """Build a character-specific prompt"""
        prompt_parts = [
            f"<lora:{lora_file}:{strength}>",
            trigger_word
        ]
        
        if action and action != 'standing confidently':
            prompt_parts.append(action)
        else:
            prompt_parts.append('standing confidently')
        
        if outfit_cue:
            prompt_parts.append(outfit_cue)
        
        if prompt_keywords:
            prompt_parts.append(prompt_keywords)

        return ' '.join(prompt_parts)
    
    def _build_scene_prompt(self, environment_desc, positioning, char1_name, char2_name):
        """Build the scene/environment prompt"""
        prompt_parts = []
        
        if environment_desc:
            prompt_parts.append(environment_desc)
        
        if positioning:
            prompt_parts.append(positioning)
        
        # Add quality and style descriptors
        prompt_parts.append('cinematic lighting, fantasy illustration, high detail')
        
        return ', '.join(prompt_parts)
    
    def generate_story_gemini(self, theme=None):
        """Generate story using Google Gemini API"""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        
        character_context = self.format_characters_for_prompt()
        
        char1_name = self.characters.get('character_1', {}).get('fantasy_name', 'Character 1')
        char2_name = self.characters.get('character_2', {}).get('fantasy_name', 'Character 2')
        
        prompt = f"""{character_context}

Write an exciting high fantasy adventure story featuring these two characters. 
The story should be suitable for display as a picture book with 10-12 pages.

REQUIREMENTS:
- Write exactly 10 distinct scenes/pages
- Each page should be 2-3 sentences of narrative
- Each page should be a clear, visually distinct scene
- Show the characters' personalities and their relationship dynamic
- Include character-appropriate actions (their quirks, fighting styles, etc.)
- Build to a satisfying conclusion
- Use vivid, descriptive language suitable for image generation
{f"- Story theme: {theme}" if theme else ""}

IMPORTANT: Make sure the story reflects their personalities:
- {char1_name}'s cautious but curious nature, love of fine things, trust issues
- {char2_name}'s protective caring nature, connection to nature/deity, maternal strength
- Their contrasting approaches but strong partnership

Format your response as a JSON array with this structure:
[
  {{
    "page": 1,
    "text": "The narrative text for this page (2-3 sentences)",
    "scene_description": "Detailed visual description of the scene for image generation. "
  }},
  ...
]

CRITICAL: The scene_description must be EXTREMELY EXPLICIT and detailed:
- State EXACTLY which character is doing what action (use their names: {char1_name} or {char2_name})
- Do not use personal pronouns without referencing the character's name in the same sentence.
- Describe WHERE each character is positioned (left/right/center, foreground/background)
- Specify WHAT each character is holding or wearing
- Describe the setting, lighting, and mood
- Be precise about body positions and actions
- If a character holds a weapon, specify WHICH character and WHICH hand

EXAMPLE of a good scene_description:
"{char2_name} stands in the foreground on the left, holding a glowing sword raised high in her right hand, shield on her left arm. {char1_name} crouches to the right in the background, daggers drawn, watching the shadows. Ancient stone temple interior, torchlight casting dramatic shadows, tense atmosphere."
"{char2_name} wears her paladin armor and has a gentle smile."
"{char1_name}'s Lightning Dagger is clipped to his belt, crackling faintly."

BAD scene_description (too vague):
"The heroes face danger in the temple"
"She wears her paladin armor and has a gentle smile."
"His Lightning Dagger is clipped to his belt, crackling faintly."

Include character names, specific positions, and explicit actions in EVERY scene description!"""

        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 8192,
            }
        }
        
        response = requests.post(
            f"{url}?key={self.api_key}",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            story_pages = json.loads(text)
            
            # Add prompt fields to each page
            print("\nGenerating prompt fields for each page...")
            for page in story_pages:
                prompts = self.build_page_prompts(page)
                page.update(prompts)
                # Remove old scene_description field (now replaced by prompts)
                # Keep it for backward compatibility or remove it
                # page.pop('scene_description', None)  # Uncomment to remove
            
            return story_pages
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def save_story(self, story_pages, output_dir='stories'):
        """Save generated story to file"""
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/story_{timestamp}.json"
        
        story_data = {
            'generated_at': timestamp,
            'characters': self.characters,
            'pages': story_pages
        }
        
        with open(filename, 'w') as f:
            json.dump(story_data, f, indent=2)
        
        print(f"Story saved to: {filename}")
        return filename

# Example usage
if __name__ == "__main__":
    # Load API key from environment variable
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: Please set GEMINI_API_KEY in .env file")
        print("Get a free API key from: https://aistudio.google.com/apikey")
        exit(1)
    
    # Generate story
    generator = StoryGenerator(api_key)
    
    print("\n" + "="*60)
    print("HIGH FANTASY STORY GENERATOR")
    print("WITH SEPARATE CHARACTER AND SCENE PROMPTS")
    print("="*60)
    print("\nGenerating story with your characters:")
    print(f"- {generator.characters.get('character_1', {}).get('fantasy_name', 'Character 1')}")
    print(f"- {generator.characters.get('character_2', {}).get('fantasy_name', 'Character 2')}")
    print("\nThis will take 30-60 seconds...\n")
    
    # Optional: custom theme
    theme = None
    # Uncomment to use a custom theme:
    # theme = "A mysterious ancient artifact hidden in dangerous ruins"
    
    story = generator.generate_story_gemini(theme=theme)
    
    # Save story
    filename = generator.save_story(story)
    
    # Print preview
    print(f"\n{'='*60}")
    print(f"✓ STORY GENERATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nGenerated {len(story)} pages")
    print(f"Saved to: {filename}")
    print(f"\nPreview of first page:")
    print(f"\n--- Page 1 ---")
    print(f"Text: {story[0]['text']}")
    print(f"\nPrompts:")
    print(f"Character 1: {story[0]['character_1_prompt']}")
    print(f"Character 2: {story[0]['character_2_prompt']}")
    print(f"Scene: {story[0]['scene_prompt']}")
    print(f"Negative: {story[0]['negative_prompt']}")
    print(f"\n{'='*60}")
    print("Next step: Generate images using the new prompt structure")
    print("The prompts are optimized for ComfyUI multi-LoRA workflow")
    print(f"{'='*60}")