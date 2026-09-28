### Thematic analysis module modified from "LLM-in-the-loop: Leveraging Large Language Model for Thematic Analysis." Authors: Dai, Shih-Chieh and Xiong, Aiping and Ku, Lun-Wei. EMNLP 2023.

import json
import os
import sys
import re
from google import genai
import os
import sys
import dotenv
from google.genai import types
import pandas as pd
import pickle

# Prompts/parts of prompts used in multiple functions
initial_system_prompt = "You are a researcher conducting a thematic analysis."

# Load environment variables from a .env file
dotenv.load_dotenv()

# --- Global Gemini API Setup ---
# Retrieve the Gemini API key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Exit if the API key is not set, providing a helpful message
if GEMINI_API_KEY is None:
    print("Your API key is not set correctly! Please set the GEMINI_API_KEY environment variable.")
    sys.exit(1)

# Initialize the Gemini client with the API key
CLIENT = genai.Client(api_key=GEMINI_API_KEY)

# --- Core Chat Function for Gemini ---
def chat(messages):
    """
    Sends a list of messages to the Gemini model and returns the response.
    Handles system instructions separately as per Gemini API requirements.

    Args:
        messages (list[types.Content]): A list of message objects for the conversation.

    Returns:
        types.GenerateContentResponse: The response object from the Gemini API.

    Raises:
        Exception: If an error occurs during the API call.
    """
    system_instruction = ''
    contents = [] # Messages to be sent as 'contents' to the model

    # Iterate through messages to separate system instruction from user/assistant messages
    for msg in messages:
        if msg.role == 'system':
            # Extract system instruction text; Gemini expects it as a separate parameter
            system_instruction = msg.parts[0].text
        else:
            contents.append(msg) # Add other messages to contents

    try:
        # Call the Gemini API to generate content
        response = CLIENT.models.generate_content(
            model="gemini-2.0-flash", # Using gemini-2.0-flash as specified by the user
            config=types.GenerateContentConfig(
                system_instruction=system_instruction # Pass the system instruction
            ),
            contents=contents # Pass the conversation history
        )
        return response
    except Exception as e:
        # Catch and re-raise any exceptions from the API call
        print(f"An error occurred during API call: {e}")
        raise

# --- Utility Functions ---
def msg_builder(role, msg):
    """
    Builds a Gemini-compatible message object.

    Args:
        role (str): The role of the message sender ('user', 'model', 'system').
        msg (str): The text content of the message.

    Returns:
        types.Content: A Gemini message object.
    """
    return types.Content(
        role=role,
        parts=[types.Part.from_text(text=msg)]
    )

def get_content(resp):
    """
    Extracts the text content from a Gemini API response object.

    Args:
        resp (types.GenerateContentResponse): The response object from the Gemini API.

    Returns:
        str: The generated text content.
    """
    return resp.text

def json_writer(file_name, obj):
    """
    Writes a Python object to a JSON file.

    Args:
        file_name (str): The name of the file to write to.
        obj (any): The Python object to serialize to JSON.
    """
    with open(file_name, 'w', encoding='utf-8') as json_f:
        json.dump(obj, json_f, indent=4, ensure_ascii=False)

def print_pretty(obj):
    """
    Prints a JSON object in a human-readable, indented format.
    Handles both JSON strings and Python dict/list objects.

    Args:
        obj (str or dict or list): The object to print.
    """
    if isinstance(obj, str):
        try:
            tmp = json.loads(obj) # Try to load if it's a JSON string
        except json.JSONDecodeError:
            print(obj) # If not valid JSON, print as is
            return
    else:
        tmp = obj # Already a dict/list

    print(json.dumps(tmp, indent=4)) # Pretty print the JSON

def load_data(free_text, line_start, line_end):
    """
    Loads text data from a file, formats it into questions and answers,
    and prepares it for machine processing.

    Args:
        free_text (str): Path to the input text file.
        line_start (int): Starting line number (0-indexed).
        line_end (int): Ending line number (exclusive).

    Returns:
        tuple: A tuple containing:
            - code_prompt (list): List of dictionaries with "Q" and "A" keys.
            - text_for_machine (str): Concatenated text for the model.
    """
    with open(free_text, 'r', encoding='utf-8') as f:
        text = f.read().split("\n")[line_start:line_end]

    code_prompt = []
    text_for_machine = ""

    response_number = line_start + 1
    for i, line in enumerate(text):
        line = line.strip()
        # Responses get quoted in some but not all cases when saving - make it consistent
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        txt = f"Response {response_number + i}. {line}"
        tmp = {"Q": txt, "A": "Let's code the response!"}
        code_prompt.append(tmp)
        text_for_machine += txt + "\n"

    return code_prompt, text_for_machine

def preprocess_json(response_text):
    """
    Extracts and parses the first valid JSON object from a response string.
    Handles markdown-wrapped JSON (```json ... ```) and raw text.

    Args:
        response_text (str): The raw text response from the model.

    Returns:
        dict or list: Parsed JSON content.

    Raises:
        ValueError: If no valid JSON object is found or decoding fails.
    """
    # 1. Try to extract content within triple backticks first
    matches = re.findall(r"```(?:json)?(.*?)```", response_text, flags=re.DOTALL)
    if matches:
        candidate = matches[0].strip()
    else:
        # 2. Fallback: Try to extract the first JSON object via bracket counting
        # This is less reliable but can catch cases where markdown is missing.
        brace_count = 0
        start_index = None
        candidate = None
        for i, char in enumerate(response_text):
            if char == '{':
                if start_index is None:
                    start_index = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_index is not None:
                    candidate = response_text[start_index:i+1]
                    break
        if candidate is None:
            raise ValueError("No valid JSON object found in the response.")

    # 3. Try parsing the extracted candidate
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}\nCandidate text: {candidate}")


def read_file(file_path:str):
    with open(file_path, 'r') as f:
        contents = f.read()
    return contents

# --- Thematic Analysis Workflow Functions ---


def create_exemplar_prompt(exemplar_file:str, init_coding_task_file:str):
    '''
    Reads the two files to create a prompt for initial coding

    Args:
        exemplar_file (str): path to a file containing a description of the overall TA task and the exemplars
        init_task_file (str): path to a file containing the text for initial coding of some number of responses.

    Returns:
        str: String with the contents of the two files concatenated (new line separated)
    '''
    exemplars_for_codes = read_file(exemplar_file)
    init_coding_task_description = read_file(init_coding_task_file)

    return exemplars_for_codes + "\n" + init_coding_task_description

def init_code_gen(prompt_exp, prompt_codes, init_codes_out_file='init_codes.json'):
    """
    Generates initial codes based on explanations and prompts.

    Args:
        prompt_exp (str): Explanatory prompt for the model.
        prompt_codes (list): List of code prompts (Q/A pairs).

    Returns:
        tuple: A tuple containing:
            - resps (types.GenerateContentResponse): The raw response object.
            - msgs (list): The updated message history.
            - content (str): The concatenated content sent to the model.
    """
    msgs = [
        msg_builder("system", initial_system_prompt),
        msg_builder("user", prompt_exp)
    ]
    # Concatenate prompt_codes into a single string for the model
    content = "".join(f"{x['Q']}\n{x['A']}\n\n" for x in prompt_codes)
    msgs.append(msg_builder("user", content))

    print("Waiting for the response...")
    resps = chat(msgs) # Send messages to Gemini
    content_result = get_content(resps)

    # Attempt to parse as JSON first, then fallback to line splitting
    try:
        parsed_content = json.loads(content_result)
        if isinstance(parsed_content, list):
            content_result = parsed_content
        else:
            content_result = [parsed_content] # Wrap single item in a list
    except json.JSONDecodeError:
        content_result = [line.strip() for line in content_result.split("\n") if line.strip()]

    # Load existing data from init_codes.json if it exists
    existing_data = []
    if os.path.exists(init_codes_out_file):
        with open(init_codes_out_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data] # Ensure it's a list
            except json.JSONDecodeError:
                pass # Treat as empty if file is corrupted

    # Append new content to existing data
    existing_data.extend(content_result)

    # Write updated data back to file
    with open(init_codes_out_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    return resps, msgs, content

def create_msgs_object_for_coding_summary_from_responses(all_coded_responses:str, exemplar_file:str):
    """
    Creates a message history for the coded responses to use as input for generating a summary of the coding responses.

    Args:
        all_coded_responses (str): The content of previous previous coding by the model.
        exemplar_file (str): Path to a file containing the initial overall task description and exemplars

    Returns:
        tuple: A tuple containing:
            - existing_data (dict): The updated summary of codes.
            - codes (list): A list of unique code labels.
    """
    # Construct initial messages to ensure that llm gets the full history
    initial_prompt = read_file(exemplar_file)
    initial_prompt += (
        "\nYou were asked to code a dataset of responses. Those responses and your codes appear in the next message."
    )
    initial_prompt += (
        "\nYou were asked to code a dataset of responses. Those responses and your codes appear in the next message in the following format:"
        "'Response number': \n"
        "'quote' refers to / mentions 'definition of the code'. Therefore, we got a code: 'code'.\n"
    )

    msgs = [
        msg_builder("system", initial_system_prompt),
        msg_builder("user", initial_prompt)
    ]

    # Append the assistant's previous response to the message history
    msgs.append(msg_builder("assistant", all_coded_responses))
    return msgs

def create_msgs_object_for_coding_summary_from_msg_and_response_history(resps, msgs):
    """
    Appends content of resps to the msgs history.
    Args:
        resps (types.GenerateContentResponse): The previous response from the model.
        msgs (list): The current message history.

    Returns:
        list: list of msgs with content of responses added on
    """
    return msgs + [msg_builder("assistant", get_content(resps))]


def sum_code_gen_from_messages(msgs, summary_out_file="summary.json"):
    """
    Summarizes and categorizes generated codes into a structured JSON format.

    Args:
        resps (types.GenerateContentResponse): The previous response from the model.
        msgs (list): The current message history.

    Returns:
        tuple: A tuple containing:
            - existing_data (dict): The updated summary of codes.
            - codes (list): A list of unique code labels.
    """
    
    # Prompt instructing the model on the exact JSON structure for code summary
    #     prompt_sum = (
    #     "Please turn your previous coding of responses into a JSON object with the following structure:\n"
    # "{\n"
    # '  "1": [\n'
    # '    { "Programming/Coding": "studying of coding, examining codes, writing programs" },\n'
    # '    { "Problem-Solving": "understanding and solving logical or technical problems" }\n'
    # "  ]\n"
    # "}\n\n"
    # "- Each key at the top level is a string representing the response number (e.g., \"1\", \"2\").\n"
    # "- Each value is a list of dictionaries.\n"
    # "- Each dictionary has ONE key (the category label), and the value is a string definition.\n"
    # "- Do NOT use any other keys like 'code', 'key', or 'label'.\n"
    # "- The output must be ONLY a valid JSON object, with no markdown or other formatting."
    # )
    prompt_sum = (
        "Please turn your previous coding of responses into a JSON object with the following structure:\n"
    "{\n"
    '  "ResponseNumber": [\n'
    '    { "Code1": "Code1Definition" },\n'
    '    { "Code2": "Code2Definition" }\n'
    "  ]\n"
    "}\n\n"
    "- Each key at the top level is a string representing the response number (e.g., \"1\", \"2\", etc.).\n"
    "- Each value is a list of dictionaries. Lists may have one or more dictionaries in them.\n"
    "- Each dictionary has ONE key (the code - e.g., \"Programming\", \"Communication\", etc.), and the value is a string definition.\n"
    "- Do NOT use any other keys like 'code', 'key', or 'label'.\n"
    "- The output must be ONLY a valid JSON object, with no markdown or other formatting."
    )

    msgs.append(msg_builder("user", prompt_sum))

    print("Waiting for the response...")
    resps = chat(msgs) # Send the summary prompt to Gemini
    sum_codes_raw = get_content(resps)

    # Preprocess and parse the raw response into a JSON object
    try:
        codes_detail = preprocess_json(sum_codes_raw)
    except Exception as e:
        print("Exception:", e)
        return sum_codes_raw,None

    existing_data = {}

    # # Load existing summary.json if it exists
    # if os.path.exists(summary_out_file):
    #     with open(summary_out_file, "r", encoding="utf-8") as f:
    #         try:
    #             existing_data = json.load(f)
    #         except json.JSONDecodeError:
    #             pass # Treat as empty if corrupted

    # Merge new codes_detail into existing_data
    for key, value in codes_detail.items():
        if key in existing_data:
            if isinstance(value, list):
                existing_data[key].extend(value)
            else:
                existing_data[key].append(value)
        else:
            existing_data[key] = value

    # Save the updated data back to summary.json
    with open(summary_out_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    # Extract unique code labels from the summarized data
    codes = []
    for entry_list in existing_data.values():
        for item_dict in entry_list:
            if isinstance(item_dict, dict):
                codes.extend(list(item_dict.keys()))
    codes = list(set(codes)) # Get unique labels

    return existing_data, codes


def sum_code_gen_from_responses(all_coded_responses, exemplar_file, summary_out_file="summary.json"):
    """
    Summarizes and categorizes generated codes into a structured JSON format.

    Args:
        all_coded_responses (str): The content of previous previous coding by the model.
        exemplar_file (str): Path to a file containing the initial overall task description and exemplars

    Returns:
        tuple: A tuple containing:
            - existing_data (dict): The updated summary of codes.
            - codes (list): A list of unique code labels.
    """

    # Construct initial messages to ensure that gemini gets the full history
    initial_prompt = read_file(exemplar_file)
    initial_prompt += (
        "\nYou were asked to code a dataset of responses. Those responses and your codes appear in the next message."
    )

    msgs = [
        msg_builder("system", initial_system_prompt),
        msg_builder("user", initial_prompt)
    ]

    # Append the assistant's previous response to the message history
    msgs.append(msg_builder("assistant", all_coded_responses))
    
    # Prompt instructing the model on the exact JSON structure for code summary
    prompt_sum = (
        "Please turn your previous coding of responses into a JSON object with the following structure:\n"
    "{\n"
    '  "1": [\n'
    '    { "Programming/Coding": "studying of coding, examining codes, writing programs" },\n'
    '    { "Problem-Solving": "understanding and solving logical or technical problems" }\n'
    "  ]\n"
    "}\n\n"
    "- Each key at the top level is a string representing the response number (e.g., \"1\", \"2\").\n"
    "- Each value is a list of dictionaries.\n"
    "- Each dictionary has ONE key (the category label), and the value is a string definition.\n"
    "- Do NOT use any other keys like 'code', 'key', or 'label'.\n"
    "- The output must be ONLY a valid JSON object, with no markdown or other formatting."
)

    msgs.append(msg_builder("user", prompt_sum))

    print("Waiting for the response...")
    resps = chat(msgs) # Send the summary prompt to Gemini
    sum_codes_raw = get_content(resps)

    # Preprocess and parse the raw response into a JSON object
    try:
        codes_detail = preprocess_json(sum_codes_raw)
    except Exception as e:
        print("Exception:", e)
        return sum_codes_raw,None

    existing_data = {}

    # # Load existing summary.json if it exists
    # if os.path.exists(summary_out_file):
    #     with open(summary_out_file, "r", encoding="utf-8") as f:
    #         try:
    #             existing_data = json.load(f)
    #         except json.JSONDecodeError:
    #             pass # Treat as empty if corrupted

    # Merge new codes_detail into existing_data
    for key, value in codes_detail.items():
        if key in existing_data:
            if isinstance(value, list):
                existing_data[key].extend(value)
            else:
                existing_data[key].append(value)
        else:
            existing_data[key] = value

    # Save the updated data back to summary.json
    with open(summary_out_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    # Extract unique code labels from the summarized data
    codes = []
    for entry_list in existing_data.values():
        for item_dict in entry_list:
            if isinstance(item_dict, dict):
                codes.extend(list(item_dict.keys()))
    codes = list(set(codes)) # Get unique labels

    return existing_data, codes

def codes_to_themes(codes_detail, codes, question):
    """
    Organizes generated codes into themes.

    Args:
        codes_detail (dict): Detailed breakdown of codes.
        codes (list): List of unique code labels.
        question (str): The original question related to the responses.

    Returns:
        tuple: A tuple containing:
            - resp_theme_json (dict): The generated themes in JSON.
            - msgs_theme (list): The updated message history.
    """
    prompt_theme = (f"Here is the question for the responses: {question}\n"
                    "Please organize the codes into themes in JSON format. "
                    "The JSON should contain a single top-level key, 'themes', "
                    "whose value is a list of theme objects. "
                    "Ensure that each code belongs to only one theme. Assign a name to each theme."
                    "If there are any duplicate codes please merge them into a single entry in the code book."
                    "Example: 'Computer systems' and 'systems' should be merged to be one code called 'Computer systems'."
                    "Each theme object should have a 'theme' key (string) and a 'codes' key (list of strings). "
                    "Example: {'themes': [{'theme': 'Learning', 'codes': ['Programming/Coding', 'Problem-Solving']}]}"
                    "Each theme should have between 3 and 12 codes. "
                    "Return ONLY a valid JSON object with no explanations or markdown formatting."
                   )
    # Prepare detailed codes and unique codes for the model's context
    codes_detail_str = json.dumps(codes_detail, ensure_ascii=False, indent=2)

    msgs_theme = [
        msg_builder("system", "You are a researcher to conduct thematic analysis. The output format should be in JSON."),
        msg_builder("user", f"Here are the codes and their definitions:\n{codes_detail_str}\n\n"
                            f"Here are the unique codes: {', '.join(codes)}\n\n"
                            f"{prompt_theme}")
    ]
    print("Waiting for the response...")
    resp_theme = chat(msgs_theme) # Send the theme generation prompt to Gemini
    print("Completed!")
    resp_theme_content = get_content(resp_theme)
    # Append the assistant's response to the message history
    msgs_theme.append(msg_builder("assistant", resp_theme_content))

    # Preprocess and parse the response into JSON
    resp_theme_json = preprocess_json(resp_theme_content)
    json_writer("revised_theme.json", resp_theme_json)
    return resp_theme_json, msgs_theme

def codes_to_themes_and_definitions(codes_detail, codes, question, json_out_file="revised_theme.json"):
    """
    Organizes generated codes into themes.

    Args:
        codes_detail (dict): Detailed breakdown of codes.
        codes (list): List of unique code labels.
        question (str): The original question related to the responses.

    Returns:
        tuple: A tuple containing:
            - resp_theme_json (dict): The generated themes in JSON.
            - msgs_theme (list): The updated message history.
    """
    prompt_theme = (f"Here is the question for the responses: {question}\n"
                    "Please organize the codes into themes in JSON format.\n "
                    "The JSON should contain a single top-level key, 'themes', "
                    "whose value is a list of theme objects. "
                    "Ensure that each code belongs to only one theme. Assign a name to each theme."
                    "Please merge duplicate or highly similar codes into a single entry in the code book."
                    "Example: 'Computer systems' and 'systems' should be merged to be one code called 'Computer systems'.\n"
                    "Each theme object should have a 'theme' key (string) and a 'codes' key (list of code objects). "
                    "Each code object should have a 'code' key (string) with the name of the code and a 'definition' key (string) with the definition of that code.\n"
                    "Example: \n {'themes': [{'theme': 'Learning', 'codes': [{'code': 'Programming/Coding', 'definition': 'Writing instructions for computers to execute; writing code in a programming language.'},"
                    "{'code': 'Problem-Solving', 'definition': 'Ientifying, analyzing, and resolving problems, often using a logical process.']}]}\n"
                    "Themes should typically have 3-12 codes. \n"
                    "Definitions should typically be no more than 25 words."
                    "Return ONLY a valid JSON object with no explanations or markdown formatting."
                   )
    # Prepare detailed codes and unique codes for the model's context
    codes_detail_str = json.dumps(codes_detail, ensure_ascii=False, indent=2)

    msgs_theme = [
        msg_builder("system", initial_system_prompt),
        msg_builder("user", f"Here are the codes and their definitions:\n{codes_detail_str}\n\n"
                            f"Here are the unique codes: {', '.join(codes)}\n\n"
                            f"{prompt_theme}")
    ]
    print("Waiting for the response...")
    resp_theme = chat(msgs_theme) # Send the theme generation prompt to Gemini
    print("Completed!")
    resp_theme_content = get_content(resp_theme)
    # Append the assistant's response to the message history
    msgs_theme.append(msg_builder("assistant", resp_theme_content))

    # Preprocess and parse the response into JSON
    resp_theme_json = preprocess_json(resp_theme_content)
    json_writer(json_out_file, resp_theme_json)
    return resp_theme_json, msgs_theme

def discussion(resp_theme, msgs_theme, action_description, json_out_file="revised_theme.json"):
    """
    Facilitates a discussion with the model about theme revisions.

    Args:
        resp_theme (dict): The initial generated themes.
        msgs_theme (list): The current message history for themes.
        action_description (str): Description of actions taken for revision.

    Returns:
        list: The updated message history after the discussion.
    """
    # Load the revised themes from file (assuming it was saved previously)
    with open(json_out_file, 'r', encoding='utf-8') as json_f:
        revised_theme = json.load(json_f)

    discussion_prompt = (f"Here is your original version:\n{json.dumps(resp_theme, indent=2, ensure_ascii=False)}\n"
                         f"Here is the revised version:\n{json.dumps(revised_theme, indent=2, ensure_ascii=False)}\n"
                         f"Here are the reasons for the revision: {action_description}\n"
                         "What do you think? Please provide your feedback in JSON format, "
                         "listing parts with which you agree and disagree, and the reasons. "
                         "The JSON should contain a 'discussion' key, whose value is an object "
                         "with 'agree' and 'disagree' keys. Each of these should be a list of objects, "
                         "where each object has a 'point' (string) and 'reason' (string). "
                         "Return ONLY a valid JSON object with no explanations or markdown formatting."
                        )

    msgs_theme.append(msg_builder("user", discussion_prompt))
    print("Waiting for the response...")
    diss_resp = chat(msgs_theme) # Send the discussion prompt to Gemini
    resp_text = get_content(diss_resp)
    
    print_pretty(json.dumps(resp_text)) # Print the JSON object for review
    # Append the raw text response to the message history
    msgs_theme.append(msg_builder("assistant", resp_text))
    return msgs_theme

def discussion_llm_revises(resp_theme, msgs_theme, action_description, json_out_file="revised_theme.json"):
    """
    Facilitates a discussion with the model about theme revisions.

    Args:
        resp_theme (dict): The initial generated themes.
        msgs_theme (list): The current message history for themes.
        action_description (str): Description of actions taken for revision.

    Returns:
        list: The updated message history after the discussion.
    """
    # Load the revised themes from file (assuming it was saved previously)
    with open(json_out_file, 'r', encoding='utf-8') as json_f:
        revised_theme = json.load(json_f)

    discussion_prompt = (f"Here is your original version:\n{json.dumps(resp_theme, indent=2, ensure_ascii=False)}\n"
                         f"Here is a revision that I'm proposing and why: {action_description}\n"
                         "What do you think about this revision? Please provide your feedback on this revision in JSON format, "
                         "listing parts with which you agree and disagree, and the reasons. "
                         "The JSON should contain a 'discussion' key, whose value is an object "
                         "with 'agree' and 'disagree' keys. Each of these should be a list of objects, "
                         "where each object has a 'point' (string) and 'reason' (string). "
                         "Return ONLY a valid JSON object with no explanations or markdown formatting."
                        )

    msgs_theme.append(msg_builder("user", discussion_prompt))
    print("Waiting for the response...")
    diss_resp = chat(msgs_theme) # Send the discussion prompt to Gemini
    resp_text = get_content(diss_resp)
    
    print_pretty(json.dumps(resp_text)) # Print the JSON object for review
    # Append the raw text response to the message history
    msgs_theme.append(msg_builder("assistant", resp_text))
    return msgs_theme

def diss_process(msgs_theme, json_out_file="revised_theme.json"):
    """
    Processes the discussion feedback and generates a revised set of themes.

    Args:
        msgs_theme (list): The current message history for themes.

    Returns:
        tuple: A tuple containing:
            - resp_theme (dict): The revised themes in JSON.
            - msgs_theme (list): The updated message history.
    """
    msgs_theme.append(msg_builder("user", "Please generate the revised version of the themes and codes in JSON format, based on our discussion. \n" 
    "                                       Your revision should only change the areas we discussed."
                                          "The JSON should follow the same structure as before: e.g., \n"
                                          "{'themes': [{'theme': 'Theme Name', 'codes': [{'code': 'Code1Name', 'definition': 'Code1Definition'}, {'code': 'Code2Name', 'definition': 'Code2Definition'}]}]}\n "
                                          "Return ONLY a valid JSON object with no explanations or markdown formatting."))
    print("Waiting for the response...")
    diss_resp = chat(msgs_theme) # Send the revised themes request to Gemini
    response_text = get_content(diss_resp)
    # Append the raw text response to the message history
    msgs_theme.append(msg_builder("assistant", response_text))
    
    # Preprocess and parse the response into JSON
    resp_theme = preprocess_json(response_text)
    
    #make changes to revised_themes
    json_writer(json_out_file, resp_theme)
    print(f"Updated '{json_out_file}' with the new themes.")

    return resp_theme, msgs_theme

def examine_additional_responses_to_revise_codebook(question, additional_responses, codebook, additional_coding_out_file='additional_coded_responses.json'):
    msgs_theme = [
            msg_builder("system", initial_system_prompt),

    ]
    prompt_add_codes = (
        f"Participants were asked to response to the following question: {question}"
        "Here is a codebook for coding responses: \n"
        f"{json.dumps(codebook, indent=2)} \n"
        "Use the codebook to code a set of new responses, using existing codes to the degree you can, and identify what additional codes are needed to capture the themes in these new responses.\n"
        "Add a new code when a theme is present that is not expressed in the original codebook. In most cases, you will be able to use codes already in the codebook. \n"
        "For each response , do the following:\n"
        "Generate one or more codes and explain your reasoning using the following output format for each response:"
        "'Response number': \n"
        "'quote' refers to / mentions 'definition of the code'. Therefore, we got a code: 'code'. "
    )
    msgs_theme.append(msg_builder("user", prompt_add_codes))
    content = "".join(f"{x['Q']}\n{x['A']}\n\n" for x in additional_responses)
    msgs_theme.append(msg_builder("user", content))

    print("Waiting for the response...")
    resps = chat(msgs_theme) # Send messages to Gemini
    content_result = get_content(resps)
    # Attempt to parse as JSON first, then fallback to line splitting
    try:
        parsed_content = json.loads(content_result)
        if isinstance(parsed_content, list):
            content_result = parsed_content
        else:
            content_result = [parsed_content] # Wrap single item in a list
    except json.JSONDecodeError:
        content_result = [line.strip() for line in content_result.split("\n") if line.strip()]


    # Write updated data back to file
    with open(additional_coding_out_file, "w", encoding="utf-8") as f:
        json.dump(content_result, f, ensure_ascii=False, indent=2)

    return resps, msgs_theme, content

def create_new_codebook_from_additional_responses(msgs_history, codebook_out_file):
    prompt_summarize = (f"Create a new codebook that combines the old codebook and any new codes that you added when coding the new responses.\n"
                        "If no new codes were added, the new codebook should be the same as the old codebook.\n"
                        "The codebook shoul be organize the codes into themes in JSON format.\n "
                    "The JSON should contain a single top-level key, 'themes', "
                    "whose value is a list of theme objects. "
                    "Ensure that each code belongs to only one theme."
                    "Please merge duplicate or highly similar codes into a single entry in the code book."
                    "Example: 'Computer systems' and 'systems' should be merged to be one code called 'Computer systems'.\n"
                    "Each theme object should have a 'theme' key (string) and a 'codes' key (list of code objects). "
                    "Each code object should have a 'code' key (string) with the name of the code and a 'definition' key (string) with the definition of that code.\n"
                    "Example: \n {'themes': [{'theme': 'Learning', 'codes': [{'code': 'Programming/Coding', 'definition': 'Writing instructions for computers to execute; writing code in a programming language.'},"
                    "{'code': 'Problem-Solving', 'definition': 'Ientifying, analyzing, and resolving problems, often using a logical process.']}]}\n"
                    "Themes should typically have 3-12 codes. \n"
                    "Definitions should typically be no more than 25 words."
                    "Return ONLY a valid JSON object with no explanations or markdown formatting."
                   )
    msgs_history.append(msg_builder("user", prompt_summarize))

    print("Waiting for the response...")
    resp_theme = chat(msgs_history) 
    print("Completed!")
    resp_theme_content = get_content(resp_theme)
    # Append the assistant's response to the message history
    msgs_history.append(msg_builder("assistant", resp_theme_content))

    # Preprocess and parse the response into JSON
    resp_theme_json = preprocess_json(resp_theme_content)
    json_writer(codebook_out_file, resp_theme_json)
    return resp_theme_json

def merge_codebooks(codebook1_json, codebook2_json, codebook_out_file="merged_codebook.json"):
    msgs_theme = [
        msg_builder("system", initial_system_prompt),
    ]
    prompt_merge =(
        "Two codebooks have been created with themes, codes, and definitions. The codebooks are similar, but may have differences.\n"
        "Your task is to merge the two codebooks to create a new codebook.\n" 
        "This codebook should include all distinct codes that exist in either codebook, but codes that are very similar should be merged.\n" 
        "For example, if codebook 1 has the code 'Computer systems' and codebook 2 has the code 'Systems', the final codebook should " \
        "only merge these codes into one code called 'Computer systems' because the two codes express the same concept.\n"
        "Here are the two codebooks: "
        f"Codebook 1:\n"
        f"{json.dumps(codebook1_json, indent=2)} \n\n"
        f"Codebook 2:\n"
        f"{json.dumps(codebook2_json, indent=2)} \n\n"
        "Please output a JSON codebook in the same format as the original codebooks."
        "The JSON should contain a single top-level key, 'themes', "
        "whose value is a list of theme objects. "
        "Ensure that each code belongs to only one theme."
        "Each theme object should have a 'theme' key (string) and a 'codes' key (list of code objects). "
        "Each code object should have a 'code' key (string) with the name of the code and a 'definition' key (string) with the definition of that code.\n"
        "Example: \n {'themes': [{'theme': 'Learning', 'codes': [{'code': 'Programming/Coding', 'definition': 'Writing instructions for computers to execute; writing code in a programming language.'},"
        "{'code': 'Problem-Solving', 'definition': 'Ientifying, analyzing, and resolving problems, often using a logical process.']}]}\n"
        "Themes should typically have 3-12 codes. \n"
        "Definitions should typically be no more than 25 words."
        "Return ONLY a valid JSON object with no explanations or markdown formatting."
    )
    msgs_theme.append(msg_builder("user", prompt_merge))

    print("Waiting for the response...")
    resp_theme = chat(msgs_theme) 
    print("Completed!")
    resp_theme_content = get_content(resp_theme)
    # Append the assistant's response to the message history
    msgs_theme.append(msg_builder("assistant", resp_theme_content))

    # Preprocess and parse the response into JSON
    resp_theme_json = preprocess_json(resp_theme_content)
    json_writer(codebook_out_file, resp_theme_json)
    return resp_theme_json, msgs_theme


def final_codebook_gen(name, msgs_theme):
    """
    Generates the final code book, incorporating all revisions and assigning IDs to codes.

    Args:
        name (str): The filename for the final code book.
        msgs_theme (list): The current message history for themes.

    Returns:
        dict: The final code book in JSON format.
    """
    final_prompt = ("Please generate the final code book in JSON format, incorporating all revisions. "
                    "The JSON should have a top-level key 'themes', whose value is a list of theme objects. "
                    "Each theme object should have a 'theme' key (string) and a 'codes' key (list of code objects).\n "
                    "Each code object should have a 'code' key (string) with the name of the code and a 'definition' key (string) with the definition of that code.\n"
                    "Additionally, assign each code object an ID number as an additional key, numbering codes consecutively across themes so that each id number appears only once."
                    "Example: \n{'themes': [{'theme1': 'Theme1Name', 'codes': [{'id': 1, 'code': 'Code1Name', 'definition': 'Code1Definition'}, {'id': 2, 'code': 'Code2Name', 'definition': 'Code2Definition'}]}]}\n "
                    "Return ONLY a valid JSON object with no explanations or markdown formatting."
                   )
    msgs_theme.append(msg_builder("user", final_prompt))
    print("Waiting for the response...")
    final_code_book_resp = chat(msgs_theme) # Send the final code book request to Gemini
    final_content = get_content(final_code_book_resp)
    
    # Preprocess and load the final code book
    final_code_book = preprocess_json(final_content)
    
    print("Final Code Book before processing (raw from model):")
    print_pretty(final_code_book)

    # # Ensure the structure is as requested: {'ID_NUMBER': 'CodeName'}
    # if "themes" in final_code_book and isinstance(final_code_book["themes"], list):
    #     for theme_entry in final_code_book["themes"]:
    #         if "codes" in theme_entry and isinstance(theme_entry["codes"], list):
    #             new_codes_list = []
    #             for i, code_item in enumerate(theme_entry["codes"]):
    #                 code_value = ""
    #                 if isinstance(code_item, dict):
    #                     # If already in {'ID': 'CodeName'} or {'CodeName': 'definition'}
    #                     if len(code_item) == 1 and str(list(code_item.keys())[0]).isdigit():
    #                         new_codes_list.append(code_item) # Keep as is if already ID:CodeName
    #                     else:
    #                         # Try to get the value, assuming it's the code name
    #                         code_value = list(code_item.values())[0] if code_item else ""
    #                         new_codes_list.append({str(i + 1): code_value})
    #                 elif isinstance(code_item, str):
    #                     code_value = code_item
    #                     new_codes_list.append({str(i + 1): code_value})
    #             theme_entry["codes"] = new_codes_list
    
    json_writer(name, final_code_book)
    print(f"The code book has been generated! The file name is {name}.")
    return final_code_book


def generate_code_definitions(codebook, context):
    final_prompt = ("Please add definitions to the codebook using the context gained from the following response examples provided."
                    f"------------Codebook---------\n{codebook}"
                    f"--- Contextual Responses ---\n{context[:30000]}"
                    "Here is a codebook developed to label all of the responses. "
                    "It has a top level 'themes', whose value is a list of theme objects. "
                    "Each theme object contains 'theme' (string) and 'codes' (list of dictionaries). "
                    "For the 'codes' list, there is a unique numerical ID for each code, "
                    "where each code is represented as a dictionary like {'ID_NUMBER': 'CodeName'}. "
                    "Do not change the structure of the json file except to add code definition after codename."
                    "Generate a definition for all 30 codes even if you are unsure."
                    "The format should now be {'ID_NUMBER': 'CodeName-Code Definition'}"
                    "Return ONLY a valid JSON object with no explanations or markdown formatting."
                   )
    messages_for_llm = [msg_builder("user", final_prompt)]

    print("Waiting for response...")
    
    # Send the combined prompt to the LLM
    # 'chat' is assumed to be your function to interact with the LLM API
    llm_response = chat(messages_for_llm)
    
    # Get the raw content from the LLM's response
    raw_content = get_content(llm_response)
    
    # Preprocess and load the JSON from the LLM's response
    # 'preprocess_json' should handle cleaning up markdown, etc.
    codebook_with_definitions = preprocess_json(raw_content)

    json_writer("codebook_definitions.json", codebook_with_definitions)
    print("Codebook with definitions (raw from model):")
    # Using json.dumps to pretty print for readability in console
    print(json.dumps(codebook_with_definitions, indent=2, ensure_ascii=False))


    return codebook_with_definitions


def content_list_to_string(content_list):
    content = ""
    for j in range(1, len(content_list) + 1):
        cur_content = content_list[j-1].replace("\n"," ")
        # print(cur_content)
        content += f"Response {j}. {cur_content}\n"
    return content
   
def label(final_code_book, content_list, coded_outfile="coded_result.json"):
    """
    Labels responses using the codes from the final code book.

    Args:
        final_code_book (dict): The final code book with themes and coded IDs.
        content (str): The raw text content to be labeled.

    Returns:
        dict: A JSON object with labeled responses.
    """
    content_str = content_list_to_string(content_list) 
    label_prompt = ("Using the codes from the code book (not the themes), label each response with all codes that apply to that response based on its contents.\n"
                    "The codebook has the following JSON format: a top-level key 'themes', whose value is a list of theme objects. "
                    "Each theme object has a 'theme' key (string) and a 'codes' key (list of code objects).\n "
                    "Each code object has an 'id' key (number) with the number for the code, a 'code' key (string) with the name of the code, "
                    "and a 'definition' key (string) with the definition of that code.\n"
                    "The code book is:\n"
                    f"{json.dumps(final_code_book, indent=2, ensure_ascii=False)}\n"
                    "Here are the responses to be labeled:\n"
                    f"{content_str}\n\n"
                    "For each response, provide a list of relevant codes including both the text and IDs (the numerical IDs from the code book)"
                    "Choose all codes that describe the content of the response, using the definitions to determine the meaning of each code. "
                    "The output should be a JSON object where keys are response numbers (e.g., 'Response 1', 'Response 2') "
                    "and values are lists of dictionaries with code IDs and code names. "
                    "If there are no codes that match a response, the response should still be included in the dictionary with an empty list for a value. \n"
                    "Example: \n{'Response 1': [{'code': Code1Name, 'id': Code1ID},{'code': Code2Name, 'id': Code2ID}], 'Response 2': [{'code': Code1Name, 'id': Code1id}]}\n "
                    "Return ONLY a valid JSON object with no explanations or markdown formatting."
                   )
    print("Waiting for the response...")
    code_result_resp = chat([msg_builder("user", label_prompt)]) # Send labeling request to Gemini
    result_content = get_content(code_result_resp)
    
    # Preprocess and parse the response into JSON
    code_result = preprocess_json(result_content)
    
    json_writer(coded_outfile, code_result)
    return code_result



def convert_labeled_responses_to_records(labeled_response_json, response_offset:int = 0):
    '''
    
    response_offset: Response 1 will instead have index value 1 + response_offset, response 2 will have index value 2 + response_offset, and so on.
    '''
    all_records = []
    for i in range(1, len(labeled_response_json) + 1):
        cur_codes = labeled_response_json[f'Response {i}']
        cur_record = {'Response' : i + response_offset}
        for code in cur_codes:
            cur_record[code['code']] = 1
            cur_record[code['id']] = 1
        all_records.append(cur_record)
    return all_records
    # return pd.DataFrame.from_records(all_records, index='Response').fillna(0)


#### Discuss final codebook and coding discrepancies to improve definitions/codes

def get_initial_messages_for_codebook_discrepancy_discussion(final_code_book, question):
    """
    Labels responses using the codes from the final code book.

    Args:
        final_code_book (dict): The final code book with themes and coded IDs.
        question (str): The question participants responded to.

    Returns:
        msgs: A list of messages for interacting with llm (includes initial system message, for instance).
    """
    msgs = [
        msg_builder("system", initial_system_prompt),
    ]
    prompt_discuss_discrepancies =(
        f"Participants were asked to response to the following question: {question}"
        "Here is a codebook for coding responses: \n"
        f"{json.dumps(final_code_book, indent=2)} \n"
        "You and I both coded a subset of participant responses. We'll now discuss "
        "some cases where we disagreed to try to refine the codebook and definitions, "
        "so that for future coding, we'll have higher agreement."
    )
    msgs.append(msg_builder("user", prompt_discuss_discrepancies))
    return msgs

def discuss_discrepancy(msgs, response_text, discrepancy_description):
    prompt_discuss_discrepancy = (
        "Here is a participant response where we disagreed on at least one code:\n" \
        f"Response: {response_text}\n"
        "Here's a description of one or more codes where we disagreed and my thoughts on the correct resolution:\n"
        f"{discrepancy_description}\n"
        "What do you think? Please provide your feedback in JSON format, "
        "listing parts with which you agree and disagree, and the reasons. \n"
        "The JSON should contain a 'discussion' key, whose value is an object "
        "with 'agree' and 'disagree' keys. Each of these should be a list of objects, "
        "where each object has a 'point' (string) and 'reason' (string). If you have an " \
        "alternative resolution that you think would be preferable, please include it " \
        "as an item in your list for 'disagree', where the 'point' would be the alternative " \
        "resolution."
    )
    msgs.append(msg_builder("user", prompt_discuss_discrepancy))

    print("Waiting for the response...")
    diss_resp = chat(msgs) # Send the discussion prompt to Gemini
    resp_text = get_content(diss_resp)
    
    print_pretty(json.dumps(resp_text)) # Print the JSON object for review
    # Append the raw text response to the message history
    msgs.append(msg_builder("assistant", resp_text))
    return msgs

def resolve_discrepancy(msgs, resolution_text, codebook, json_out_file='revised_final_codebook.json'):
    prompt_resolution = (
         "Please generate the revised version of the themes and codes in JSON format, based on our discussion. Specifically, please revise this codebok: \n"
        f"{json.dumps(codebook, indent=2)} \n"
        " to implement this resolution: \n"
         f"{resolution_text}\n"
        "Your revision should only change the areas we discussed."
        "The JSON should follow the same structure as before: e.g., \n"
        "{'themes': [{'theme': 'Theme Name', 'codes': [{'id': 'Code1ID', 'code': 'Code1Name', 'definition': 'Code1Definition'}, {'id': 'Code2ID', 'code': 'Code2Name', 'definition': 'Code2Definition'}]}]}\n "
    )
    msgs.append(msg_builder("user", prompt_resolution))
    print("Waiting for the response...")
    diss_resp = chat(msgs) # Send the revised themes request to Gemini
    response_text = get_content(diss_resp)
    # Append the raw text response to the message history
    msgs.append(msg_builder("assistant", response_text))
    
    # Preprocess and parse the response into JSON
    resp_theme = preprocess_json(response_text)
    
    #make changes to revised_themes
    json_writer(json_out_file, resp_theme)
    print(f"Updated '{json_out_file}' with the new themes.")

    return resp_theme, msgs



def backup_discussion(msgs_theme_obj, msgs_theme_outfile, rev_themes_json, rev_themes_outfile):
    with open(msgs_theme_outfile, 'wb') as f:
        pickle.dump(msgs_theme_obj, f)

    with open(rev_themes_outfile, "w", encoding="utf-8") as f:
            json.dump(rev_themes_json, f, indent=2, ensure_ascii=False)