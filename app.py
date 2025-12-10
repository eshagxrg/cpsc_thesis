import os
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template

# IMPORT EXTERNAL AGENT COMPONENTS: for simplicity and to avoid circular dependencies, we'll try to import specific functions and state.

try:
    # Must ensure all dependencies (pandas, openai, llama_index, config.py, etc.) are installed and configured
    from AcademicAgent import chat_with_agent, initialize_course_data, COURSE_CATALOG, YALE_RULES, PlannerState
    print("Agent modules imported successfully.")
    
    # Initialize course data (RAG index and catalog) at Flask startup
    initialize_course_data()
    
    IS_AGENT_MOCKED = False
except ImportError as e:
    print(f"WARNING: Failed to import real agent components from AcademicAgent.py. Using mock logic. Error: {e}")
    print("Ensure all required Python libraries (LlamaIndex, LangGraph, pandas, etc.) are installed and data files are present.")
    IS_AGENT_MOCKED = True

# Configuration
PLAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plans')
app = Flask(__name__, template_folder='templates')

# Ensure the plans directory exists
if not os.path.exists(PLAN_DIR):
    os.makedirs(PLAN_DIR)

initialState = {
    "llm_thread": [{"role": "system", "content": "You are an expert Yale academic planning assistant."}],
    "messages": [],
    "major": None,
    "major_interests": [],
    "general_interests": [],
    "plan": None,
    "auditFindings": None,
}

# Agent State Management (since the real agent manages its own global 'state', we wrap it)

# Global variable to hold the agent's current session state when not using a plan file
# Initialize it here, but it's loaded/updated in chat_api route
current_session_state = json.loads(json.dumps(initialState))

# Wrapper to call the imported chat_with_agent, handles both state update and getting the AI response text
def agent_wrapper(state, user_input):
    """
    Wraps the imported chat_with_agent to manage the state variable correctly.
    ASSUMPTION: chat_with_agent returns (response_text, new_state) tuple.
    """
    print("STATE Before Sending", json.dumps(state.get('major')), json.dumps(state.get("major_interests")),  json.dumps(state.get("general_interests")), json.dumps(state.get('plan')))
    
    # ASSUMPTION: user's external chat_with_agent function returns (response_text, new_state)
    try:
        response_text, new_state = chat_with_agent(state, user_input)
    except Exception as e:
        print(f"ERROR: AcademicAgent.chat_with_agent failed: {e}")
        response_text = f"An error occurred in the agent logic: {e}"
        new_state = state

    print("STATE After Sending", json.dumps(new_state.get('major')),  json.dumps(new_state.get("major_interests")), json.dumps(new_state.get("general_interests")), json.dumps(new_state.get('plan')))
    
    ai_messages = [m for m in new_state["messages"] if m.get("type") == "ai"]
    ai_response = ai_messages[-1]["content"] if ai_messages else response_text # Fallback to raw text if structure is off
    
    return new_state, ai_response


# File I/O Helpers

def get_state_filepath(plan_id):
    """Returns the full path for a plan file."""
    if not plan_id.endswith('.json'):
        plan_id += '.json'
    return os.path.join(PLAN_DIR, plan_id)

def save_state(state, existing_plan_id=None):
    """
    Saves the current state to a JSON file. 
    If existing_plan_id is provided and valid, it updates that file (persistence). 
    Otherwise, it generates a new timestamp ID (new session).
    Returns the ID of the file that was written to.
    """
    if existing_plan_id and existing_plan_id != 'new':
        plan_id_to_use = existing_plan_id
    else:
        plan_id_to_use = datetime.now().strftime("%Y%m%d%H%M%S")
    
    major_name = state.get('major', 'Unknown Major')
    
    plan_data = {
        'id': plan_id_to_use,
        'date': datetime.now().isoformat(),
        'major': major_name,
        'state': state
    }
    
    filepath = get_state_filepath(plan_id_to_use)
    with open(filepath, 'w') as f:
        json.dump(plan_data, f, indent=2)
    
    return plan_id_to_use

def load_state(plan_id):
    """Loads a state from a JSON file."""
    filepath = get_state_filepath(plan_id)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading or parsing plan file {plan_id}: {e}")
        return None


def list_plans():
    """Lists all available plans for the sidebar."""
    plans = []
    # Ensure the directory exists before listing
    if not os.path.exists(PLAN_DIR):
        return plans
        
    for filename in os.listdir(PLAN_DIR):
        if filename.endswith('.json'):
            try:
                # Load the whole file here, but only need metadata
                data = load_state(filename.replace('.json', ''))
                if data and 'major' in data and 'date' in data:
                    plans.append({
                        'id': data['id'],
                        'date': datetime.fromisoformat(data['date']).strftime("%b %d, %Y %I:%M%p"),
                        'major': data['major']
                    })
            except Exception as e:
                # Handled by load_state, but keeping robust
                print(f"Error loading plan file {filename}: {e}")
                continue
    plans.sort(key=lambda p: p['id'], reverse=True)
    return plans

def get_latest_plan_id():
    """Returns the ID of the most recently saved plan, or 'new' if none exist."""
    plans = list_plans()
    return plans[0]['id'] if plans else 'new'

# --- Flask Routes ---

@app.route('/')
def index():
    """Render the main chat page, loading the latest plan ID."""
    latest_plan_id = get_latest_plan_id()
    return render_template('index.html', initial_plan_id=latest_plan_id)

@app.route('/api/plans', methods=['GET'])
def plans_api():
    """API to list all saved plans for the sidebar."""
    return jsonify(list_plans())

@app.route('/api/load_plan/<plan_id>', methods=['GET'])
def load_plan_api(plan_id):
    """API to load a specific plan state."""
    data = load_state(plan_id)
    if data:
        # Save the loaded state globally for the next chat interaction.
        global current_session_state
        current_session_state = data['state']
        return jsonify(data['state'])
    return jsonify({"error": "Plan not found"}), 404

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API for handling chat messages and plan updates."""
    data = request.json
    user_input = data.get('message', '').strip()
    plan_id = data.get('plan_id')

    # Load or Initialize State
    global current_session_state
    
    if plan_id == 'new':
        # Start a new session state (deep copy of the agent's initial state)
        current_session_state = json.loads(json.dumps(initialState))
    elif plan_id:
        # Load state from file
        print("loading a plan state from the file", plan_id)
        loaded_data = load_state(plan_id)
        if loaded_data:
            current_session_state = loaded_data['state']
        else:
            return jsonify({"error": "Invalid plan ID"}), 400
    
    # If starting fresh and no ID provided, use the current global state (already initialized)
    if 'current_session_state' not in globals() or not current_session_state:
        current_session_state = json.loads(json.dumps(initialState))

    # Invoke Agent: Call the wrapper which uses the real/mock agent
    new_state, ai_response = agent_wrapper(current_session_state, user_input)

    # Save State: Use the modified state object to save the plan
    plan_id_to_return = save_state(new_state, existing_plan_id=plan_id)
    return jsonify({
        'response': ai_response,
        'state': new_state,
        'new_plan_id': plan_id_to_return,
        'plans': list_plans()
    })


if __name__ == '__main__':
    # Usage: flask --app app.py run
    # Note: If IS_AGENT_MOCKED is False, the agent's data is initialized above.
    app.run(debug=True)