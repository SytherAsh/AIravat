import wikipedia
import json
from datetime import datetime
import os

class WikipediaSearcher:
    def __init__(self):
        """Initialize the Wikipedia searcher"""
        self.history_file = "search_history.json"
        self.create_project_structure()

    def create_project_structure(self):
        """Create necessary directories and files for the project"""
        # Create data directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")

    def search_wikipedia(self, query):
        """
        Search Wikipedia for a given query and return search results
        """
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                return "No results found for your search."
            
            return search_results
        except Exception as e:
            return f"An error occurred while searching: {str(e)}"

    def get_page_info(self, title):
        """
        Get detailed information about a specific Wikipedia page
        """
        try:
            # Get the Wikipedia page
            page = wikipedia.page(title)
            
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
            
            # Save the search result
            self.save_search_result(page_info)
            
            return page_info
        except wikipedia.DisambiguationError as e:
            return {
                "error": "Disambiguation Error",
                "options": e.options,
                "message": "Multiple matches found. Please be more specific."
            }
        except wikipedia.PageError:
            return {
                "error": "Page Error",
                "message": f"Page '{title}' does not exist."
            }
        except Exception as e:
            return {
                "error": "General Error",
                "message": f"An error occurred: {str(e)}"
            }

    def save_search_result(self, page_info):
        """
        Save search result to a JSON file
        """
        filename = f"data/{page_info['title'].replace('/', '_')}_wikipediaapi.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(page_info, f, ensure_ascii=False, indent=2)

    def load_search_history(self):
        """
        Load search history from file
        """
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_to_history(self, query, selected_result):
        """
        Save search query and selected result to history
        """
        history = self.load_search_history()
        history.append({
            "query": query,
            "selected": selected_result,
            "timestamp": datetime.now().isoformat()
        })
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

def main():
    """
    Main function to run the Wikipedia search and information retrieval
    """
    wiki = WikipediaSearcher()
    
    print("Welcome to Airavat Wikipedia Search and Information Retriever!")
    print("This program allows you to search Wikipedia and get detailed information about topics.")
    
    while True:
        # Get search query from user
        query = input("\nEnter a topic to search (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid search term.")
            continue
        
        # Search for the topic
        print("\nSearching Wikipedia...")
        search_results = wiki.search_wikipedia(query)
        
        if isinstance(search_results, str):
            print(search_results)
            continue
        
        # Display search results
        print("\nSearch Results:")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. {result}")
        
        # Ask user to select a result
        try:
            selection = int(input("\nEnter the number of the topic you want to learn about (0 to search again): "))
            if selection == 0:
                continue
            if 1 <= selection <= len(search_results):
                selected_topic = search_results[selection - 1]
                
                # Save to search history
                wiki.save_to_history(query, selected_topic)
                
                # Get detailed information about the selected topic
                print(f"\nGetting information about '{selected_topic}'...")
                info = wiki.get_page_info(selected_topic)
                
                if "error" in info:
                    print(f"\nError: {info['message']}")
                    if "options" in info:
                        print("\nPossible options:")
                        for i, option in enumerate(info['options'][:10], 1):
                            print(f"{i}. {option}")
                else:
                    print("\n" + "="*50)
                    print(f"Title: {info['title']}")
                    print(f"URL: {info['url']}")
                    print("\nSummary:")
                    print(info['summary'])
                    print("\nFull Content (first 1000 characters):")
                    print(info['content'][:1000] + "...")
                    print("\nCategories (first 5):")
                    for category in info['categories'][:5]:
                        print(f"- {category}")
                    print("\nReferences (first 5):")
                    for ref in info['references'][:5]:
                        print(f"- {ref}")
                    print("\nImages (first 5):")
                    for img in info['images'][:5]:
                        print(f"- {img}")
                    print("="*50)
                    print(f"\nFull information has been saved to the data directory.")
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
