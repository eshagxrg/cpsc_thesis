# Yale CourseMap: AI-Based Academic Planner
Yale Senior Thesis for the Department of Computer Science by Esha Garg

CourseMap is an AI-powered academic planning tool designed to help students generate, visualize, and refine 4-year course schedules. It combines a Retrieval-Augmented Generation (RAG) system with a deterministic rule engine to ensure plans align with official Yale degree requirements while catering to personal academic interests.

## Features

  * **Conversational Interface:** Chat naturally to express academic goals and interests.
  * **Hybrid AI Architecture:** Uses Large Language Models (LLM) for reasoning and RAG for accurate course catalog retrieval.
  * **Degree Audit & Rules:** Integrated logic to check against specific major requirements (Prerequisites, Core, Senior Requirements).
  * **Interactive Planner:** Visualizes 4-year schedules term-by-term.
  * **Persistence:** Save, load, and compare multiple academic plans.
  * **Audit System:** Automatically flags if specific degree requirements are "MET" or "NOT MET".

## Tech Stack

  * **Backend:** Python, Flask
  * **AI/ML:** Azure OpenAI, LlamaIndex, LangGraph, HuggingFace Embeddings
  * **Frontend:** HTML5, Tailwind CSS, JavaScript
  * **Data:** Pandas, Local Vector Store

## Prerequisites

  * Python 3.10+
  * An Azure OpenAI API Key and Endpoint
  * Official Course Data (CSVs and JSON schemas)

## Installation

1.  **Clone the repository**

2.  **Create a Virtual Environment (Optional)**

3.  **Install Dependencies:**

      * For standard environments:
        ```bash
        pip install -r requirements.txt
        ```
      * For macOS (Apple Silicon M1/M2/M3) to handle specific Torch/NumPy versions:
        ```bash
        pip install -r requirements-mac.txt
        ```

## Configuration

The system requires a `config.py` file in the root directory to authenticate with Azure OpenAI.

1.  Create a file named `config.py`.
2.  Add the following variables (replace placeholders with your actual credentials):

<!-- end list -->

```python
# config.py

# Azure OpenAI Configuration
ENDPOINT = ""
API_KEY = ""
API_VERSION = "2024-12-01-preview"  # Recommended; Used for Creation
MODEL_NAME = "gpt-5-mini"  # Recommended; Used for Creation
```

## Data Setup

The agent relies on specific external data files to function. Ensure the following directory structure exists and is populated:

  * **`courses/`**: A folder containing CSV files of the course catalog (e.g., `202503.csv`, `202601.csv`).
  * **`degree_text/`**: A folder containing `.txt` descriptions of degree rules (used for RAG context).
  * **`degree_classification_final.ndjson`**: A JSON/NDJSON file containing the structured logic for major requirements.
  * **`master_major_interests.json`**: A JSON file mapping majors to their canonical subfields/interests.
  * **`course_index_storage_new/`**: (Generated automatically) The code will generate a local vector store here on the first run.

**Directory Tree:**

```text
yale-coursemap/
├── app.py                      # Flask Server
├── AcademicAgent.py            # Core Logic
├── config.py                   # API Keys (You must create this)
├── templates/
│   └── index.html              # Frontend UI
├── courses/                    # [DATA] Course CSVs
├── degree_text/                # [DATA] Degree text files
├── degree_classification_final.ndjson  # [DATA] Rules
├── master_major_interests.json         # [DATA] Interests
└── requirements.txt
```

## Usage

1.  **Start the Server:**
    Run the Flask application. On the first run, this may take a moment to build the vector index.

    ```bash
    flask run
    ```

2.  **Access the Application:**
    Open your web browser and navigate to:

    ```
    http://127.0.0.1:5000
    ```

3.  **How to Use:**

      * Click **"+ Start New Plan"** in the sidebar.
      * In the chat box, type something like: *"I want to major in Computer Science, and I'm interested in AI and Graphics."*
      * The agent will extract your interests and propose a plan.
      * You can ask it to refine the plan, e.g., *"Add more writing electives in Year 2."*

## System Components

### `AcademicAgent.py`

The brain of the system. It initializes the `LlamaIndex` vector store from the course data. It utilizes `LangGraph` to manage the state of the conversation (`PlannerState`), switching between chatting and plan generation nodes.

### `app.py`

A lightweight Flask API that bridges the frontend and the Agent. It handles session management, saving/loading plans to the `plans/` directory (JSON format), and serving the HTML frontend.

### `index.html`

A single-page application using Tailwind CSS. It polls the Flask API to render the chat history, the visual 4-year schedule, and the audit findings.

## ⚠️ Troubleshooting

  * **Import Errors:** If you see `ImportError: cannot import name ... from AcademicAgent`, ensure `IS_AGENT_MOCKED` is set to `False` in `app.py` and that all data files are present.
  * **Azure Errors:** If you get authentication errors, double-check your `config.py` against your Azure OpenAI resource settings.
  * **Vector Store Issues:** If the app crashes on startup regarding `course_index_storage_new`, try deleting that folder and restarting the app to force a rebuild of the index.
