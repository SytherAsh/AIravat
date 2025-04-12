import streamlit as st
import wikipedia
import os
import json
import time
import re
import logging
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


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Wiki Comic Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
        </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<div class="main-header">Wikipedia Comic Story Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Transform Wikipedia articles into engaging comic book storylines!</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png", width=100)
        st.markdown("## Settings")
        
        # Groq API key input
        groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
        
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
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app extracts content from Wikipedia and uses Groq's 
        language models to generate comic book storylines.
        
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
                        st.success(f"Found {len(search_results)} results for '{query}'")
        
        # Display search results
        if 'search_results' in st.session_state and st.session_state.search_results:
            st.markdown("### Search Results")
            
            for i, result in enumerate(st.session_state.search_results):
                if st.button(f"{i+1}. {result}", key=f"result_{i}"):
                    st.session_state.selected_topic = result
                    st.session_state.page_info = None
                    st.session_state.storyline = None
    
    with col2:
        # Display selected topic and retrieve page info
        if 'selected_topic' in st.session_state and st.session_state.selected_topic:
            st.markdown(f'<div class="sub-header">Selected Topic: {st.session_state.selected_topic}</div>', unsafe_allow_html=True)
            
            if 'page_info' not in st.session_state or st.session_state.page_info is None:
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
    
    # Display generated storyline in a full-width section
    if 'storyline' in st.session_state and st.session_state.storyline:
        st.markdown("---")
        st.markdown('<div class="sub-header">Generated Comic Storyline</div>', unsafe_allow_html=True)
        
        # Add a download button for the storyline
        st.download_button(
            label="Download Storyline",
            data=st.session_state.storyline,
            file_name=f"{st.session_state.page_info['title']}_comic_storyline.md",
            mime="text/markdown"
        )
        
        st.markdown('<div class="success-text">Here\'s your comic book storyline based on the Wikipedia article!</div>', unsafe_allow_html=True)
        
        # Display the storyline with proper formatting
        st.markdown(st.session_state.storyline)
        
        # Save button with custom filename
        custom_filename = st.text_input("Custom filename (optional):", 
                                       value=f"{st.session_state.page_info['title']}_comic_storyline")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Save to Project", type="primary"):
                try:
                    # Create data directory if it doesn't exist
                    if not os.path.exists("data"):
                        os.makedirs("data")
                    
                    # Save storyline to file
                    wiki_extractor = WikipediaExtractor()
                    safe_filename = wiki_extractor.sanitize_filename(custom_filename)
                    file_path = f"data/{safe_filename}.md"
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(st.session_state.storyline)
                    
                    st.success(f"Storyline saved to {file_path}")
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")

if __name__ == "__main__":
    main()