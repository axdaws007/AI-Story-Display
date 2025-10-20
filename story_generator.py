import os
import json
import requests
from datetime import datetime
from pathlib import Path

class StoryGenerator:
    def __init__(self, api_key, characters_file='characters.json'):
        self.api_key = api_key
        self.characters = self.load_characters(characters_file)
        
    def load_characters(self, file_path):
        """Load character profiles from JSON file"""
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
    "scene_description": "Detailed visual description of the scene for image generation"
  }},
  ...
]

CRITICAL: The scene_description must be EXTREMELY EXPLICIT and detailed:
- State EXACTLY which character is doing what action (use their names: {char1_name} or {char2_name})
- Describe WHERE each character is positioned (left/right/center, foreground/background)
- Specify WHAT each character is holding or wearing
- Describe the setting, lighting, and mood
- Be precise about body positions and actions
- If a character holds a weapon, specify WHICH character and WHICH hand

EXAMPLE of a good scene_description:
"{char2_name} stands in the foreground on the left, holding a glowing sword raised high in her right hand, shield on her left arm. {char1_name} crouches to the right in the background, daggers drawn, watching the shadows. Ancient stone temple interior, torchlight casting dramatic shadows, tense atmosphere."

BAD scene_description (too vague):
"The heroes face danger in the temple"

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
    print(f"Scene: {story[0]['scene_description'][:200]}...")
    print(f"\n{'='*60}")
    print("Next step: Generate images with your trained LoRAs")
    print("Run: python image_generator_lora.py")
    print(f"{'='*60}")