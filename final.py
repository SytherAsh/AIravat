import streamlit as st
import wikipedia
import os
import json
import time
import re
import logging
import requests
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
from groq import Groq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wiki_comic_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WikiComicGenerator")

class WikipediaExtractor:
    def __init__(self, data_dir: str = "data", language: str = "en"):
        """
        Initialize the Wikipedia extractor
        
        Args:
            data_dir: Directory to store extracted data
            language: Wikipedia language code
        """
        self.data_dir = data_dir
        self.create_project_structure()
        
        # Set Wikipedia language
        wikipedia.set_lang(language)
        
        logger.info(f"WikipediaExtractor initialized with data directory: {data_dir}, language: {language}")

    def create_project_structure(self) -> None:
        """Create necessary directories for the project"""
        try:
            # Create data directory if it doesn't exist
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                logger.info(f"Created data directory: {self.data_dir}")
            
            # Create images directory if it doesn't exist
            images_dir = os.path.join(self.data_dir, "images")
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
                logger.info(f"Created images directory: {images_dir}")
        except Exception as e:
            logger.error(f"Failed to create project structure: {str(e)}")
            raise RuntimeError(f"Failed to create project structure: {str(e)}")

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string to be used as a filename
        
        Args:
            filename: Original filename string
            
        Returns:
            Sanitized filename safe for all operating systems
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
        # Limit filename length
        return sanitized[:200]

    def search_wikipedia(self, query: str, results_limit: int = 15, retries: int = 3) -> Union[List[str], str]:
        """
        Search Wikipedia for a given query and return search results
        
        Args:
            query: Search query
            results_limit: Maximum number of results to return
            retries: Number of retries on network failure
            
        Returns:
            List of search results or error message string
        """
        if not query or not query.strip():
            return "Please enter a valid search term."
        
        query = query.strip()
        logger.info(f"Searching Wikipedia for: {query}")
        
        # Implement retry logic with exponential backoff
        attempt = 0
        while attempt < retries:
            try:
                search_results = wikipedia.search(query, results=results_limit)
                
                if not search_results:
                    suggestions = wikipedia.suggest(query)
                    if suggestions:
                        logger.info(f"No results found. Suggesting: {suggestions}")
                        return f"No exact results found. Did you mean: {suggestions}?"
                    logger.info("No results found and no suggestions available")
                    return "No results found for your search."
                
                logger.info(f"Found {len(search_results)} results for query: {query}")
                return search_results
                
            except ConnectionError as e:
                attempt += 1
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Connection error (attempt {attempt}/{retries}): {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return f"An error occurred while searching: {str(e)}"
        
        return "Failed to connect to Wikipedia after multiple attempts. Please check your internet connection."

    def get_page_info(self, title: str, retries: int = 3) -> Dict[str, Any]:
        """
        Get detailed information about a specific Wikipedia page
        
        Args:
            title: Page title to retrieve
            retries: Number of retries on network failure
            
        Returns:
            Dictionary containing page information or error details
        """
        logger.info(f"Getting page info for: {title}")
        
        attempt = 0
        while attempt < retries:
            try:
                # First try with exact title match
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                except wikipedia.DisambiguationError as e:
                    # If disambiguation page, return options
                    logger.info(f"Disambiguation error for '{title}'. Returning options.")
                    return {
                        "error": "Disambiguation Error",
                        "options": e.options[:15],  # Limit to 15 options
                        "message": "Multiple matches found. Please be more specific."
                    }
                except wikipedia.PageError:
                    # If exact title not found, try with auto-suggest
                    try:
                        logger.info(f"Exact page '{title}' not found. Trying with auto-suggest.")
                        page = wikipedia.page(title)
                    except Exception as inner_e:
                        logger.error(f"Page retrieval error: {str(inner_e)}")
                        return {
                            "error": "Page Error",
                            "message": f"Page '{title}' does not exist."
                        }
                
                # Create a dictionary with all the information
                page_info = {
                    "title": page.title,
                    "url": page.url,
                    "content": page.content,
                    "summary": page.summary,
                    "references": page.references,
                    "categories": page.categories,
                    "links": page.links,
                    "images": page.images,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save the extracted data
                self._save_extracted_data(page_info)
                
                logger.info(f"Successfully retrieved page info for: {title}")
                return page_info
                
            except ConnectionError as e:
                attempt += 1
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Connection error (attempt {attempt}/{retries}): {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error getting page info: {str(e)}")
                return {
                    "error": "General Error",
                    "message": f"An error occurred: {str(e)}"
                }
        
        return {
            "error": "Connection Error",
            "message": "Failed to connect to Wikipedia after multiple attempts. Please check your internet connection."
        }

    def _save_extracted_data(self, page_info: Dict[str, Any]) -> None:
        """
        Save extracted data to a JSON file
        
        Args:
            page_info: Dictionary containing page information
        """
        try:
            safe_title = self.sanitize_filename(page_info["title"])
            filename = f"{self.data_dir}/{safe_title}_data.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(page_info, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved extracted data to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save extracted data: {str(e)}")

class StoryGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the Groq story generator
        
        Args:
            api_key: Groq API key
        """
        self.client = Groq(api_key=api_key)
        logger.info("StoryGenerator initialized with Groq client")

    def generate_comic_storyline(self, title: str, content: str, target_length: str = "medium") -> str:
        """
        Generate a comic storyline from Wikipedia content
        
        Args:
            title: Title of the Wikipedia article
            content: Content of the Wikipedia article
            target_length: Desired length of the story (short, medium, long)
            
        Returns:
            Generated comic storyline
        """
        logger.info(f"Generating comic storyline for: {title} with target length: {target_length}")
        
        # Map target length to approximate word count
        length_map = {
            "short": 500,
            "medium": 1000,
            "long": 2000
        }
        
        word_count = length_map.get(target_length, 1000)
        
        # Check content length and truncate if necessary to avoid token limits
        max_chars = 15000
        if len(content) > max_chars:
            logger.info(f"Content too long ({len(content)} chars), truncating to {max_chars} chars")
            content = content[:max_chars] + "..."
        
        # Create prompt for the LLM
        prompt = f"""
        Create an engaging and detailed comic book storyline based on the following Wikipedia article about "{title}".
        
        The storyline should:
        1. Be approximately {word_count} words
        2. Capture the most important facts and details from the article
        3. Have a clear beginning, middle, and end
        4. Include vivid descriptions of key scenes suitable for comic panels
        5. Feature compelling characters based on real figures from the topic
        6. Include dialogue suggestions for major moments
        7. Be organized into distinct scenes or chapters
        8. Balance educational content with entertainment value
        
        Here is the Wikipedia content to base your storyline on:
        
        {content}
        
        FORMAT YOUR RESPONSE AS:
        # {title}: Comic Storyline
        
        ## Overview
        [Brief overview of the storyline]
        
        ## Main Characters
        [List of main characters with short descriptions]
        
        ## Act 1: [Title]
        [Detailed storyline for Act 1 with scene descriptions and key dialogue]
        
        ## Act 2: [Title]
        [Detailed storyline for Act 2 with scene descriptions and key dialogue]
        
        ## Act 3: [Title]
        [Detailed storyline for Act 3 with scene descriptions and key dialogue]
        
        ## Key Visuals
        [Suggestions for important visual elements to include in the comic]
        """
        
        try:
            # Generate storyline using Groq
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert comic book writer and historian who creates engaging, accurate, and visually compelling storylines based on real information."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",  # Using Llama 3 model
                temperature=0.7,
                max_tokens=4000,
                top_p=0.9
            )
            
            storyline = response.choices[0].message.content
            logger.info(f"Successfully generated comic storyline for: {title}")
            
            return storyline
            
        except Exception as e:
            logger.error(f"Failed to generate storyline: {str(e)}")
            return f"Error generating storyline: {str(e)}"

    def generate_scene_prompts(self, title: str, storyline: str, comic_style: str, num_scenes: int = 10, 
                              age_group: str = "general", education_level: str = "standard") -> List[str]:
        """
        Generate detailed scene prompts for comic panels based on the storyline
        
        Args:
            title: Title of the article
            storyline: Generated comic storyline
            comic_style: Selected comic art style
            num_scenes: Number of scene prompts to generate (default 10)
            age_group: Target age group (kids, teens, general, adult)
            education_level: Education level for content complexity (basic, standard, advanced)
            
        Returns:
            List of scene prompts for image generation
        """
        logger.info(f"Generating {num_scenes} scene prompts for comic in {comic_style} style, targeting {age_group} with {education_level} education level")
        
        # Prepare style-specific guidance based on comic style
        style_guidance = {
            "manga": "Use manga-specific visual elements like speed lines, expressive emotions, and distinctive panel layouts. Character eyes should be larger, with detailed hair and simplified facial features. Use black and white with screen tones for shading.",
            "superhero": "Use bold colors, dynamic poses with exaggerated anatomy, dramatic lighting, and action-oriented compositions. Include detailed musculature and costumes with strong outlines and saturated colors.",
            "cartoon": "Use simplified, exaggerated character features with bold outlines. Employ bright colors, expressive faces, and playful physics. Include visual effects like motion lines and impact stars.",
            "noir": "Use high-contrast black and white or muted colors with dramatic shadows. Feature low-key lighting, rain effects, and urban settings. Characters should have realistic proportions with hardboiled expressions.",
            "european": "Use detailed backgrounds with architectural precision and clear line work. Character designs should be semi-realistic with consistent proportions. Panel layouts should be regular and methodical.",
            "indie": "Use unconventional art styles with personal flair. Panel layouts can be experimental and fluid. Line work may be sketchy or deliberately unpolished. Colors can be watercolor-like or limited palette.",
            "retro": "Use halftone dots for shading, slightly faded colors, and classic panel compositions. Character designs should reflect the comics of the 50s-70s with simplified but distinctive features.",
        }.get(comic_style.lower(), f"Incorporate distinctive visual elements of {comic_style} style consistently in all panels.")
        
        # Prepare age-appropriate guidance
        age_guidance = {
            "kids": "Use simple, clear vocabulary and straightforward concepts. Avoid complex themes, frightening imagery, or adult situations. Characters should be expressive and appealing. Educational content should be presented in an engaging, accessible way.",
            "teens": "Use relatable language and themes important to adolescents. Include more nuanced emotional content and moderate complexity. Educational aspects can challenge readers while remaining accessible.",
            "general": "Balance accessibility with depth. Include some complexity in both themes and visuals while remaining broadly appropriate. Educational content should be informative without being overly technical.",
            "adult": "Include sophisticated themes, complex characterizations, and nuanced storytelling. Educational content can be presented with full complexity and technical detail where appropriate."
        }.get(age_group.lower(), "Create content appropriate for a general audience with balanced accessibility and depth.")
        
        # Prepare education level guidance
        education_guidance = {
            "basic": "Use simple vocabulary, clear explanations, and focus on foundational concepts. Break down complex ideas into easily digestible components with examples.",
            "standard": "Use moderate vocabulary and present concepts with appropriate depth for general understanding. Balance educational content with narrative engagement.",
            "advanced": "Use field-specific terminology where appropriate and explore concepts in depth. Present nuanced details and sophisticated analysis of the subject matter."
        }.get(education_level.lower(), "Present educational content with balanced complexity suitable for interested general readers.")
        
        # Create prompt for the LLM
        prompt = f"""
        Based on the following comic storyline about "{title}", create exactly {num_scenes} sequential scene prompts for generating comic panels.

        Each scene prompt MUST:
        1. Follow a logical narrative sequence from beginning to end
        2. Include DETAILED visual descriptions of the scene, setting, characters, and actions
        3. Include SPECIFIC dialogue text between characters (this is crucial as dialogue text will be directly included in speech bubbles)
        4. Ensure all dialogue is grammatically correct and appropriate for the target audience
        5. Maintain character consistency throughout all scenes
        6. Be self-contained but connect logically to the previous and next scenes
        7. Incorporate specific visual elements from the {comic_style} comic art style

        IMPORTANT PARAMETERS TO FOLLOW:
        - Comic Style: {comic_style} — {style_guidance}
        - Age Group: {age_group} — {age_guidance}
        - Education Level: {education_level} — {education_guidance}

        Here is the comic storyline to convert into scene prompts:
        
        {storyline}
        
        FORMAT EACH SCENE PROMPT AS:
        Scene [number]: [Brief scene title]
        Visual: [Extremely detailed visual description of the scene including setting, characters, positions, expressions, actions, and any specific visual elements]
        Dialog: [Character 1 name]: "[Exact dialogue text for speech bubble]"
        Dialog: [Character 2 name]: "[Exact dialogue text for speech bubble]"
        Style: {comic_style} style with [specific stylistic elements to emphasize].
        
        PROVIDE EXACTLY {num_scenes} SCENES IN SEQUENTIAL ORDER.
        MAKE SURE EACH SCENE HAS AT LEAST ONE DIALOG LINE, as these will be directly included in speech bubbles.
        ENSURE ALL DIALOG TEXT IS GRAMMATICALLY CORRECT and appropriate for the target audience.
        SCENE DESCRIPTIONS MUST BE EXTREMELY DETAILED to ensure the image generator can create accurate images.
        """
        
        try:
            # Generate scene prompts using Groq
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert comic book artist and writer who creates detailed, engaging scene descriptions for comic panels with consistent characters and storylines. You always ensure dialog is grammatically correct and include specific dialog text for each scene."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",  # Using Llama 3 model
                temperature=0.7,
                max_tokens=4000,
                top_p=0.9
            )
            
            scenes_text = response.choices[0].message.content
            
            # Process the text to extract individual scene prompts
            scene_prompts = []
            scene_pattern = re.compile(r'Scene \d+:.*?(?=Scene \d+:|$)', re.DOTALL)
            matches = scene_pattern.findall(scenes_text)
            
            for match in matches:
                scene_prompts.append(match.strip())
            
            # If we didn't get enough scenes, pad with generic ones that include dialog
            while len(scene_prompts) < num_scenes:
                scene_num = len(scene_prompts) + 1
                scene_prompts.append(f"""Scene {scene_num}: Additional scene from {title}
                Visual: A character from the story stands in a relevant setting from {title}, looking thoughtful.
                Dialog: Character: "This is an important moment in the story of {title}."
                Style: {comic_style} style with appropriate elements for {age_group} audience.""")
            
            # If we got too many scenes, truncate
            scene_prompts = scene_prompts[:num_scenes]
            
            # Validate each scene prompt to ensure it has dialog
            validated_prompts = []
            for i, prompt in enumerate(scene_prompts):
                scene_num = i + 1
                
                # Check if dialog is present
                if "Dialog:" not in prompt:
                    # Add default dialog if missing
                    prompt += f"\nDialog: Character: \"This is scene {scene_num} of our story about {title}.\""
                    logger.warning(f"Added missing dialog to scene {scene_num}")
                
                validated_prompts.append(prompt)
            
            logger.info(f"Successfully generated {len(validated_prompts)} scene prompts")
            return validated_prompts
            
        except Exception as e:
            logger.error(f"Failed to generate scene prompts: {str(e)}")
            return [f"Error generating scene prompt: {str(e)}"]


class ComicImageGenerator:
    def __init__(self, hf_token: str, model_url: str = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev", 
                 fallback_model_url: str = None):
        """
        Initialize the Comic Image Generator
        
        Args:
            hf_token: Hugging Face API token
            model_url: URL for the primary image generation model
            fallback_model_url: URL for a fallback model (if primary fails)
        """
        self.hf_token = hf_token
        self.model_url = model_url
        self.fallback_model_url = fallback_model_url
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        logger.info("ComicImageGenerator initialized with Hugging Face token")
        
        # Test API token validity at initialization
        self._test_api_token()
        
    def _test_api_token(self):
        """Test if the API token is valid and has available quota"""
        try:
            # Make a small request to check token validity
            test_response = requests.get("https://api-inference.huggingface.co/status", 
                                        headers=self.headers, timeout=10)
            
            if test_response.status_code != 200:
                logger.warning(f"API token validation failed with status code: {test_response.status_code}")
                logger.warning("You may encounter issues with image generation due to token limitations")
            else:
                logger.info("API token validated successfully")
                
        except Exception as e:
            logger.warning(f"Could not validate API token: {str(e)}")

    def _handle_payment_required_error(self, scene_num: int):
        """Handle 402 Payment Required errors specifically"""
        logger.error(f"Payment required error for scene {scene_num}. This indicates:")
        logger.error("1. Your API token has run out of free quota")
        logger.error("2. The model you're using requires payment")
        logger.error("3. There may be billing issues with your Hugging Face account")
        logger.error("Please check your Hugging Face account settings or consider using a free model")

    def _extract_dialog_from_prompt(self, scene_prompt: str) -> list:
        """Extract dialog lines from the scene prompt for adding to the image"""
        dialog_lines = []
        
        # Match lines starting with "Dialog:" followed by character name and dialog
        dialog_pattern = re.compile(r'Dialog:\s*([^:]+?):\s*"([^"]+)"', re.IGNORECASE)
        matches = dialog_pattern.findall(scene_prompt)
        
        for character, line in matches:
            dialog_lines.append((character.strip(), line.strip()))
        
        # If no dialog found with the structured format, try to extract any quoted text
        if not dialog_lines:
            # Look for quoted text with character attribution
            alt_pattern = re.compile(r'([^:]+?):\s*"([^"]+)"')
            matches = alt_pattern.findall(scene_prompt)
            for character, line in matches:
                if "style" not in character.lower() and "visual" not in character.lower():
                    dialog_lines.append((character.strip(), line.strip()))
        
        # If still no dialog, add a generic one
        if not dialog_lines:
            dialog_lines.append(("Character", "This is an important moment in our story."))
            logger.warning("No dialog found in scene prompt, using generic dialog")
        
        return dialog_lines

    def _enhance_scene_prompt(self, scene_prompt: str) -> str:
        """Enhance the scene prompt to improve image generation accuracy"""
        # Extract the main visual description
        visual_match = re.search(r'Visual:\s*(.+?)(?=\nDialog:|Style:|$)', scene_prompt, re.DOTALL)
        
        if visual_match:
            visual_description = visual_match.group(1).strip()
            
            # Extract style information
            style_match = re.search(r'Style:\s*(.+?)$', scene_prompt, re.DOTALL)
            style_info = style_match.group(1).strip() if style_match else ""
            
            # Create an enhanced prompt focused on the visual elements
            enhanced_prompt = f"""
            Generate a detailed comic panel showing:
            {visual_description}
            
            Style details: {style_info}
            
            Important: Create a detailed, high-quality comic panel with clear characters and setting.
            Ensure accurate representation of the described scene. Leave space for dialog bubbles.
            """
            
            return enhanced_prompt
        
        return scene_prompt  # Return original if we couldn't extract visual description

    def _add_dialog_bubbles(self, image_path: str, dialog_lines: list) -> str:
        """Add dialog bubbles to the generated image"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Open the image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Try to use a system font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial", 20)
                name_font = ImageFont.truetype("Arial", 16)
            except IOError:
                font = ImageFont.load_default()
                name_font = ImageFont.load_default()
            
            # Calculate positions for dialog bubbles
            max_bubbles = min(3, len(dialog_lines))  # Limit to 3 bubbles maximum
            img_width, img_height = img.size
            
            # Spacing parameters
            bubble_padding = 10
            text_max_width = int(img_width * 0.8)
            line_spacing = 25
            
            for i in range(max_bubbles):
                character, line = dialog_lines[i]
                
                # Calculate bubble position - stagger them vertically
                vertical_position = int(img_height * 0.1) + (i * int(img_height * 0.25))
                
                # Word wrap the dialog text to fit in the bubble
                words = line.split()
                lines_of_text = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    text_width = font.getbbox(test_line)[2] if hasattr(font, 'getbbox') else font.getsize(test_line)[0]
                    
                    if text_width <= text_max_width:
                        current_line = test_line
                    else:
                        lines_of_text.append(current_line)
                        current_line = word
                
                if current_line:
                    lines_of_text.append(current_line)
                
                # Calculate bubble dimensions
                bubble_height = len(lines_of_text) * line_spacing + bubble_padding * 2
                bubble_width = text_max_width + bubble_padding * 2
                
                # Draw bubble background
                bubble_x = (img_width - bubble_width) // 2
                bubble_y = vertical_position
                
                # Draw semi-transparent white bubble background
                draw.rectangle(
                    [(bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height)],
                    fill=(255, 255, 255, 200),
                    outline=(0, 0, 0)
                )
                
                # Draw character name above bubble
                name_width = name_font.getbbox(character)[2] if hasattr(name_font, 'getbbox') else name_font.getsize(character)[0]
                name_x = bubble_x + (bubble_width - name_width) // 2
                name_y = bubble_y - 20
                
                # Draw semi-transparent background for name
                draw.rectangle(
                    [(name_x - 5, name_y), (name_x + name_width + 5, name_y + 20)],
                    fill=(200, 200, 255, 200),
                    outline=(0, 0, 0)
                )
                
                draw.text((name_x, name_y), character, fill=(0, 0, 0), font=name_font)
                
                # Draw dialog text
                for j, text_line in enumerate(lines_of_text):
                    line_width = font.getbbox(text_line)[2] if hasattr(font, 'getbbox') else font.getsize(text_line)[0]
                    text_x = bubble_x + (bubble_width - line_width) // 2
                    text_y = bubble_y + bubble_padding + j * line_spacing
                    draw.text((text_x, text_y), text_line, fill=(0, 0, 0), font=font)
            
            # Save the modified image
            img.save(image_path)
            logger.info(f"Added {max_bubbles} dialog bubbles to image")
            return image_path
            
        except Exception as e:
            logger.error(f"Failed to add dialog bubbles: {str(e)}")
            return image_path

    def generate_comic_image(self, scene_prompt: str, output_path: str, scene_num: int, 
                           attempt: int = 1, max_retries: int = 3, timeout: int = 120,
                           use_fallback: bool = False) -> bool:
        """
        Generate a comic image based on a scene prompt
        
        Args:
            scene_prompt: Textual description of the scene
            output_path: Path to save the generated image
            scene_num: Scene number for logging
            attempt: Current attempt number
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            use_fallback: Whether to use the fallback model
            
        Returns:
            Boolean indicating success
        """
        current_model_url = self.fallback_model_url if use_fallback else self.model_url
        
        if use_fallback and not self.fallback_model_url:
            logger.warning("Fallback requested but no fallback model URL configured")
            return False
        
        # Extract dialog lines before enhancing the prompt
        dialog_lines = self._extract_dialog_from_prompt(scene_prompt)
        
        # Enhance the scene prompt for better image generation
        enhanced_prompt = self._enhance_scene_prompt(scene_prompt)
            
        payload = {"inputs": enhanced_prompt, "options": {"wait_for_model": True}}
        logger.info(f"Generating image for scene {scene_num}, attempt {attempt}" + 
                   (f" using fallback model" if use_fallback else ""))
        
        try:
            response = requests.post(current_model_url, headers=self.headers, json=payload, timeout=timeout)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Successfully generated image for scene {scene_num}, saved to {output_path}")
                
                # Add dialog bubbles to the image
                self._add_dialog_bubbles(output_path, dialog_lines)
                return True
                
            elif response.status_code == 402:
                logger.warning(f"Payment required error for scene {scene_num}, attempt {attempt}")
                self._handle_payment_required_error(scene_num)
                
                # Try fallback model if available and not already using it
                if self.fallback_model_url and not use_fallback:
                    logger.info(f"Attempting to use fallback model for scene {scene_num}")
                    return self.generate_comic_image(scene_prompt, output_path, scene_num, 
                                                  attempt=1, max_retries=max_retries, 
                                                  use_fallback=True)
                elif attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1})")
                    time.sleep(wait_time)
                    return self.generate_comic_image(scene_prompt, output_path, scene_num, 
                                                  attempt + 1, max_retries, use_fallback=use_fallback)
                else:
                    logger.error(f"Max retries reached for scene {scene_num} with payment errors")
                    return self._try_placeholder_image(output_path, scene_num, dialog_lines)
            else:
                logger.warning(f"Error generating image for scene {scene_num}, attempt {attempt}: {response.status_code}")
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1})")
                    time.sleep(wait_time)
                    return self.generate_comic_image(scene_prompt, output_path, scene_num, 
                                                  attempt + 1, max_retries, use_fallback=use_fallback)
                else:
                    logger.error(f"Max retries reached for scene {scene_num}")
                    return self._try_placeholder_image(output_path, scene_num, dialog_lines)
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for scene {scene_num}, attempt {attempt}: {str(e)}")
            
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1})")
                time.sleep(wait_time)
                return self.generate_comic_image(scene_prompt, output_path, scene_num, 
                                              attempt + 1, max_retries, use_fallback=use_fallback)
            else:
                logger.error(f"Max retries reached for scene {scene_num}")
                return self._try_placeholder_image(output_path, scene_num, dialog_lines)
    
    def _try_placeholder_image(self, output_path: str, scene_num: int, dialog_lines: list = None) -> bool:
        """Generate a placeholder image with scene description when all else fails"""
        try:
            # Use PIL to create a simple placeholder image with text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a blank image with a light background
            img = Image.new('RGB', (800, 600), color=(245, 245, 245))
            draw = ImageDraw.Draw(img)
            
            # Add text explaining this is a placeholder
            main_text = f"Scene {scene_num}\nImage generation failed\nPlease check your Hugging Face token"
            
            # Try to use a system font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial", 32)
                dialog_font = ImageFont.truetype("Arial", 24)
            except IOError:
                font = ImageFont.load_default()
                dialog_font = ImageFont.load_default()
                
            # Calculate text position to center it
            text_width, text_height = draw.textsize(main_text, font=font) if hasattr(draw, 'textsize') else (300, 100)
            position = ((800 - text_width) // 2, (600 - text_height) // 2 - 100)
            
            # Draw text on image
            draw.text(position, main_text, fill=(0, 0, 0), font=font)
            
            # Add dialog if available
            if dialog_lines:
                for i, (character, line) in enumerate(dialog_lines[:2]):  # Limit to 2 lines
                    dialog_text = f"{character}: \"{line}\""
                    dialog_y = position[1] + text_height + 50 + (i * 40)
                    draw.text((50, dialog_y), dialog_text, fill=(0, 0, 150), font=dialog_font)
            
            # Save the image
            img.save(output_path)
            logger.info(f"Created placeholder image for scene {scene_num} with dialog")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create placeholder image: {str(e)}")
            return False

    def generate_comic_strip(self, scene_prompts: List[str], output_dir: str, comic_title: str,
                           fallback_free_model: str = None) -> List[str]:
        """
        Generate a full comic strip from scene prompts
        
        Args:
            scene_prompts: List of scene prompts
            output_dir: Directory to save generated images
            comic_title: Title of the comic for filenames
            fallback_free_model: URL for a free model to use as fallback
            
        Returns:
            List of paths to generated images
        """
        logger.info(f"Generating comic strip with {len(scene_prompts)} scenes")
        
        # Set fallback model if provided
        if fallback_free_model:
            self.fallback_model_url = fallback_free_model
            logger.info(f"Set fallback model to: {fallback_free_model}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Create a specific directory for this comic
        safe_title = re.sub(r'[\\/*?:"<>|]', '_', comic_title)
        comic_dir = os.path.join(output_dir, safe_title)
        if not os.path.exists(comic_dir):
            os.makedirs(comic_dir)
            logger.info(f"Created comic directory: {comic_dir}")
        
        # Generate images for each scene
        image_paths = []
        payment_errors_count = 0
        
        for i, scene_prompt in enumerate(scene_prompts):
            scene_num = i + 1
            output_path = os.path.join(comic_dir, f"scene_{scene_num}.jpg")
            
            success = self.generate_comic_image(scene_prompt, output_path, scene_num)
            
            if success:
                image_paths.append(output_path)
            else:
                logger.warning(f"Failed to generate image for scene {scene_num}")
                payment_errors_count += 1
                
            # If we've had multiple payment errors in a row, it's likely a persistent issue
            if payment_errors_count >= 3 and self.fallback_model_url is None:
                logger.warning("Multiple payment errors detected. Suggesting free model alternatives.")
                logger.warning("Consider using one of these free models:")
                logger.warning("- https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1")
                logger.warning("- https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5")
                
                # Prompt if user wants to retry with a free model
                self.fallback_model_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
                logger.info(f"Automatically setting fallback model to: {self.fallback_model_url}")
        
        logger.info(f"Generated {len(image_paths)} out of {len(scene_prompts)} scenes")
        return image_paths


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Wiki Comic Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'selected_topic' not in st.session_state:
        st.session_state.selected_topic = None
    if 'page_info' not in st.session_state:
        st.session_state.page_info = None
    if 'storyline' not in st.session_state:
        st.session_state.storyline = None
    if 'scene_prompts' not in st.session_state:
        st.session_state.scene_prompts = None
    if 'comic_images' not in st.session_state:
        st.session_state.comic_images = None
    
    # Add custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #1E88E5;
            margin-bottom: 1rem;
        }
        .info-text {
            background-color: #E3F2FD;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .success-text {
            background-color: #E8F5E9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .error-text {
            background-color: #FFEBEE;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .comic-panel {
            margin: 10px;
            padding: 10px;
            border: 2px solid #1E88E5;
            border-radius: 10px;
        }
        .comic-caption {
            font-size: 0.9rem;
            font-style: italic;
            margin-top: 0.5rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<div class="main-header">Wikipedia Comic Strip Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Transform Wikipedia articles into engaging comic strips with AI-generated images!</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png", width=100)
        st.markdown("## API Keys")
        
        # API keys
        groq_api_key = st.text_input("Groq API Key", type="password", value="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")
        hf_token = st.text_input("Hugging Face API Token", type="password", value="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", help="Enter your Hugging Face API token")
        
        st.markdown("## Settings")
        
        # Wikipedia language selection
        wiki_lang = st.selectbox(
            "Wikipedia Language",
            options=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh"],
            index=0,
            format_func=lambda x: {
                "en": "English", "es": "Spanish", "fr": "French", 
                "de": "German", "it": "Italian", "pt": "Portuguese",
                "ru": "Russian", "ja": "Japanese", "zh": "Chinese"
            }.get(x, x),
            help="Select Wikipedia language"
        )
        
        # Story length
        story_length = st.select_slider(
            "Story Length",
            options=["short", "medium", "long"],
            value="medium",
            help="Select the desired length of the generated story"
        )
        
        # Comic style
        comic_style = st.selectbox(
            "Comic Art Style",
            options=[
                "manga", "western comic", "comic book", "noir comic", 
                "superhero comic", "indie comic", "cartoon", "graphic novel",
                "golden age comic", "modern comic", "mahua"
            ],
            index=1,
            help="Select the art style for the comic images"
        )
        
        # Number of scenes
        num_scenes = st.slider(
            "Number of Scenes",
            min_value=3,
            max_value=15,
            value=10,
            help="Select the number of scenes to generate"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app extracts content from Wikipedia, uses Groq to generate
        comic book storylines, and creates comic images using Hugging Face models.
        
        Created by Airavat.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Search Wikipedia</div>', unsafe_allow_html=True)
        
        # Search interface
        query = st.text_input("Enter your search query:", placeholder="Example: Albert Einstein, Moon Landing, etc.")
        search_button = st.button("Search", type="primary")
        
        if search_button and query:
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar to continue.")
            else:
                with st.spinner("Searching Wikipedia..."):
                    wiki_extractor = WikipediaExtractor(language=wiki_lang)
                    search_results = wiki_extractor.search_wikipedia(query)
                    
                    if isinstance(search_results, str):
                        st.warning(search_results)
                    else:
                        st.session_state.search_results = search_results
                        st.session_state.selected_topic = None
                        st.session_state.page_info = None
                        st.session_state.storyline = None
                        st.session_state.scene_prompts = None
                        st.session_state.comic_images = None
                        st.success(f"Found {len(search_results)} results for '{query}'")
        
        # Display search results
        if st.session_state.search_results:
            st.markdown("### Search Results")
            
            for i, result in enumerate(st.session_state.search_results):
                if st.button(f"{i+1}. {result}", key=f"result_{i}"):
                    st.session_state.selected_topic = result
                    st.session_state.page_info = None
                    st.session_state.storyline = None
                    st.session_state.scene_prompts = None
                    st.session_state.comic_images = None
    
    with col2:
        # Display selected topic and retrieve page info
        if st.session_state.selected_topic:
            st.markdown(f'<div class="sub-header">Selected Topic: {st.session_state.selected_topic}</div>', unsafe_allow_html=True)
            
            if st.session_state.page_info is None:
                with st.spinner(f"Getting information about '{st.session_state.selected_topic}'..."):
                    wiki_extractor = WikipediaExtractor(language=wiki_lang)
                    page_info = wiki_extractor.get_page_info(st.session_state.selected_topic)
                    st.session_state.page_info = page_info
            
            # Handle errors in page retrieval
            if "error" in st.session_state.page_info:
                st.error(st.session_state.page_info["message"])
                
                # Show disambiguation options if available
                if "options" in st.session_state.page_info:
                    st.markdown("### Possible options:")
                    for i, option in enumerate(st.session_state.page_info["options"]):
                        if st.button(f"{i+1}. {option}", key=f"option_{i}"):
                            st.session_state.selected_topic = option
                            st.session_state.page_info = None
                            st.session_state.storyline = None
                            st.session_state.scene_prompts = None
                            st.session_state.comic_images = None
            else:
                # Show page summary
                st.markdown("### Summary")
                st.markdown(st.session_state.page_info["summary"])
                
                # Generate storyline button
                if st.button("Generate Comic Storyline", type="primary"):
                    if not groq_api_key:
                        st.error("Please enter your Groq API key in the sidebar to continue.")
                    else:
                        with st.spinner("Generating comic storyline... This may take a minute."):
                            story_generator = StoryGenerator(api_key=groq_api_key)
                            storyline = story_generator.generate_comic_storyline(
                                st.session_state.page_info["title"],
                                st.session_state.page_info["content"],
                                target_length=story_length
                            )
                            st.session_state.storyline = storyline
                            st.session_state.scene_prompts = None
                            st.session_state.comic_images = None
    
    # Display generated storyline
    if st.session_state.storyline:
        st.markdown("---")
        st.markdown('<div class="sub-header">Generated Comic Storyline</div>', unsafe_allow_html=True)
        
        # Add a download button for the storyline
        st.download_button(
            label="Download Storyline",
            data=st.session_state.storyline,
            file_name=f"{st.session_state.page_info['title']}_comic_storyline.md",
            mime="text/markdown"
        )
        
        # Display the storyline in an expandable section
        with st.expander("Show Full Storyline", expanded=False):
            st.markdown(st.session_state.storyline)
        
        # Generate scene prompts button
        if st.button("Generate Scene Prompts for Comic Panels", type="primary"):
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar to continue.")
            else:
                with st.spinner(f"Generating {num_scenes} scene prompts for comic panels..."):
                    story_generator = StoryGenerator(api_key=groq_api_key)
                    scene_prompts = story_generator.generate_scene_prompts(
                        st.session_state.page_info["title"],
                        st.session_state.storyline,
                        comic_style,
                        num_scenes=num_scenes
                    )
                    st.session_state.scene_prompts = scene_prompts
                    st.session_state.comic_images = None
        
        # Display generated scene prompts
    if st.session_state.scene_prompts:
        st.markdown("---")
        st.markdown('<div class="sub-header">Generated Scene Prompts</div>', unsafe_allow_html=True)
        
        # Display scene prompts in an expandable section
        with st.expander("Show Scene Prompts", expanded=False):
            for i, prompt in enumerate(st.session_state.scene_prompts):
                st.markdown(f"### Scene {i+1}")
                st.text_area(f"Prompt for Scene {i+1}", value=prompt, height=150, key=f"scene_prompt_{i}")
        
        # Generate comic images button
        if st.button("Generate Comic Images", type="primary"):
            if not hf_token:
                st.error("Please enter your Hugging Face API token in the sidebar to continue.")
            else:
                with st.spinner(f"Generating {len(st.session_state.scene_prompts)} comic images... This may take several minutes."):
                    image_generator = ComicImageGenerator(hf_token=hf_token)
                    image_paths = image_generator.generate_comic_strip(
                        st.session_state.scene_prompts,
                        "data/images",
                        st.session_state.page_info["title"]
                    )
                    st.session_state.comic_images = image_paths
                    
                    if image_paths:
                        st.success(f"Successfully generated {len(image_paths)} comic panels!")
                    else:
                        st.error("Failed to generate comic images. Please check the logs for details.")
    
    # Display generated comic images
    if st.session_state.comic_images:
        st.markdown("---")
        st.markdown('<div class="sub-header">Your Generated Comic Strip</div>', unsafe_allow_html=True)
        
        # Create columns for comic panels
        cols_per_row = 3
        panels = []
        for i in range(0, len(st.session_state.comic_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(st.session_state.comic_images):
                    with cols[j]:
                        st.image(st.session_state.comic_images[idx], caption=f"Scene {idx+1}", use_column_width=True)
                        # Add scene prompt as a caption
                        if idx < len(st.session_state.scene_prompts):
                            scene_prompt = st.session_state.scene_prompts[idx]
                            # Extract just the first line as a short caption
                            short_caption = scene_prompt.split('\n')[0] if '\n' in scene_prompt else scene_prompt
                            st.markdown(f'<div class="comic-caption">{short_caption}</div>', unsafe_allow_html=True)
        
        # Add download button for a zip file of all images
        # Note: This would require implementing a function to create a zip file
        st.markdown("### Download Options")
        st.warning("Download functionality for the complete comic strip will be implemented in a future update.")
        
        # Reset button
        if st.button("Start Over", type="secondary"):
            st.session_state.search_results = None
            st.session_state.selected_topic = None
            st.session_state.page_info = None
            st.session_state.storyline = None
            st.session_state.scene_prompts = None
            st.session_state.comic_images = None
            st.experimental_rerun()

# Call main function if script is run directly
if __name__ == "__main__":
    main()