# AIravat Wikipedia Search Tool

A Python-based Wikipedia search and information retrieval tool that allows you to search Wikipedia articles and get detailed information about topics.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the program:
```bash
python wiki.py
```

2. Follow the interactive prompts to:
   - Search for topics
   - Select specific articles
   - View detailed information
   - Navigate through search results

## Features

- Search Wikipedia articles
- Get detailed page information including:
  - Summary
  - Full content
  - References
  - Categories
  - Images
  - URLs
- Save search history
- Automatic data storage in JSON format
- Error handling for disambiguation and missing pages
