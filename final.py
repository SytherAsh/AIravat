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

    def generate_scene_prompts(self, title: str, storyline: str, comic_style: str, num_scenes: int = 10) -> List[str]:
        """
        Generate detailed scene prompts for comic panels based on the storyline
        
        Args:
            title: Title of the article
            storyline: Generated comic storyline
            comic_style: Selected comic art style
            num_scenes: Number of scene prompts to generate (default 10)
            
        Returns:
            List of scene prompts for image generation
        """
        logger.info(f"Generating {num_scenes} scene prompts for comic based on storyline")
        
        # Create prompt for the LLM
        prompt = f"""
        Based on the following comic storyline about "{title}", create exactly {num_scenes} sequential scene prompts for generating comic panels.

        Each scene prompt should:
        1. Follow a logical narrative sequence from beginning to end
        2. Include detailed visual descriptions of the scene, setting, characters, and actions
        3. Include specific dialogue between characters in quotation marks
        4. Maintain character consistency throughout all scenes
        5. Be self-contained but connect logically to the previous and next scenes
        6. Use the {comic_style} comic art style consistently for all scenes
        
        Here is the comic storyline to convert into scene prompts:
        
        {storyline}
        
        FORMAT EACH SCENE PROMPT AS:
        Scene [number]: [Brief scene title]
        [Detailed visual description of the scene with setting, characters, actions]
        [Character 1]: "[Dialogue]"
        [Character 2]: "[Response dialogue]"
        — {comic_style} style with [specific stylistic elements of this comic style].
        
        PROVIDE EXACTLY {num_scenes} SCENES IN SEQUENTIAL ORDER.
        """
        
        try:
            # Generate scene prompts using Groq
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert comic book artist and writer who creates detailed, engaging scene descriptions for comic panels with consistent characters and storylines."},
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
            
            # If we didn't get enough scenes, pad with generic ones
            while len(scene_prompts) < num_scenes:
                scene_prompts.append(f"Scene {len(scene_prompts)+1}: Additional scene from {title} — {comic_style} style.")
            
            # If we got too many scenes, truncate
            scene_prompts = scene_prompts[:num_scenes]
            
            logger.info(f"Successfully generated {len(scene_prompts)} scene prompts")
            return scene_prompts
            
        except Exception as e:
            logger.error(f"Failed to generate scene prompts: {str(e)}")
            return [f"Error generating scene prompt: {str(e)}"]


class ComicImageGenerator:
    def __init__(self, hf_token: str, model_url: str = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"):
        """
        Initialize the Comic Image Generator
        
        Args:
            hf_token: Hugging Face API token
            model_url: URL for the image generation model
        """
        self.hf_token = hf_token
        self.model_url = model_url
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        logger.info("ComicImageGenerator initialized with Hugging Face token")

    def generate_comic_image(self, scene_prompt: str, output_path: str, scene_num: int, 
                           attempt: int = 1, max_retries: int = 3, timeout: int = 120) -> bool:
        """
        Generate a comic image based on a scene prompt
        
        Args:
            scene_prompt: Textual description of the scene
            output_path: Path to save the generated image
            scene_num: Scene number for logging
            attempt: Current attempt number
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            
        Returns:
            Boolean indicating success
        """
        payload = {"inputs": scene_prompt, "options": {"wait_for_model": True}}
        logger.info(f"Generating image for scene {scene_num}, attempt {attempt}")
        
        try:
            response = requests.post(self.model_url, headers=self.headers, json=payload, timeout=timeout)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Successfully generated image for scene {scene_num}, saved to {output_path}")
                return True
            else:
                logger.warning(f"Error generating image for scene {scene_num}, attempt {attempt}: {response.status_code}")
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1})")
                    time.sleep(wait_time)
                    return self.generate_comic_image(scene_prompt, output_path, scene_num, attempt + 1, max_retries)
                else:
                    logger.error(f"Max retries reached for scene {scene_num}")
                    return False
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for scene {scene_num}, attempt {attempt}: {str(e)}")
            
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1})")
                time.sleep(wait_time)
                return self.generate_comic_image(scene_prompt, output_path, scene_num, attempt + 1, max_retries)
            else:
                logger.error(f"Max retries reached for scene {scene_num}")
                return False

    def generate_comic_strip(self, scene_prompts: List[str], output_dir: str, comic_title: str) -> List[str]:
        """
        Generate a full comic strip from scene prompts
        
        Args:
            scene_prompts: List of scene prompts
            output_dir: Directory to save generated images
            comic_title: Title of the comic for filenames
            
        Returns:
            List of paths to generated images
        """
        logger.info(f"Generating comic strip with {len(scene_prompts)} scenes")
        
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
        for i, scene_prompt in enumerate(scene_prompts):
            scene_num = i + 1
            output_path = os.path.join(comic_dir, f"scene_{scene_num}.jpg")
            
            success = self.generate_comic_image(scene_prompt, output_path, scene_num)
            
            if success:
                image_paths.append(output_path)
            else:
                logger.warning(f"Failed to generate image for scene {scene_num}")
        
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
        groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
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