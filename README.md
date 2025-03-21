# Subtitle Search Engine

An advanced search engine for video subtitles with a Streamlit web interface.

## Installation

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/Manaswinicodes/Shazam-But-Better.git
cd subtitle-search-engine
```

2. Create a virtual environment and activate it:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setting Up Data

Place your subtitle files (.srt format) in the `data/` directory. You can download sample subtitle files from the provided sources or use your own collection.

## Running the Application

There are two ways to run the application:

### Option 1: Using the run script

```bash
python run_app.py
```

This script will check for dependencies and launch the Streamlit interface automatically.

### Option 2: Running Streamlit directly

```bash
streamlit run app/streamlit_app.py
```

## Using the Interface

1. **First-time setup**:
   - Set the data directory path (default: "data")
   - Set the model directory path (default: "models")
   - Choose whether to use the enhanced search engine
   - Check "Create new model" and click "Initialize Search Engine"

2. **Searching**:
   - Enter a search query
   - Set the number of results to display
   - If using enhanced search, adjust the semantic weight
   - Click "Search"

3. **Advanced Analysis** (with enhanced search):
   - Explore document clusters
   - Analyze cluster keywords
   - View search engine statistics

## Features

- Basic search using TF-IDF vectorization
- Enhanced semantic search using transformer models
- Hybrid search combining keyword and semantic similarity
- Document clustering and cluster analysis
- Interactive visualizations

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies
