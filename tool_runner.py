import ast
import os
import requests
import re
import json
from dotenv import load_dotenv
from typing import List, Tuple
import shutil
import threading
import pathspec  # pip install pathspec

# Global lock to ensure only one model request is processed at a time
MODEL_LOCK = threading.Lock()

# Load environment variables
load_dotenv()
API_URL = os.getenv("api_url", "http://localhost:8000/predict")
if not API_URL:
    print("Error: 'api_url' not found in .env file")
    raise ValueError("API URL not found in .env file")

def split_functions(filepath: str) -> List[Tuple[str, str, str, int, int]]:
    if not os.path.isfile(filepath):
        print(f"Error: File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        source = file.read()
        lines = source.splitlines()
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []
    
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                func_name = node.name
                start_line = node.lineno - 1
                end_line = max(
                    (getattr(child, 'end_lineno', node.end_lineno) or node.end_lineno) - 1
                    for child in ast.walk(node)
                )
                func_source = '\n'.join(lines[start_line:end_line + 1]).rstrip()
                docstring = ast.get_docstring(node, clean=True) or ""
                functions.append((func_name, func_source, docstring, start_line, end_line))
            except Exception as e:
                print(f"Error processing function node in {filepath}: {e}")
                continue
    
    return functions

def generate_sphinx_docstring(func_name: str, func_source: str, existing_docstring: str, api_url: str) -> str:
    if existing_docstring:
        prompt = (
            f"Improve the following Python function's docstring to be Sphinx-compatible (NumPy format). "
            f"Ensure it includes sections for Parameters, Returns, and Raises (if applicable), and maintain the original intent. "
            f"Function name: {func_name}\n"
            f"Existing docstring: {existing_docstring}\n"
            f"Function source:\n{func_source}\n"
            f"Return only the improved docstring text, without any additional explanation."
            f"Do not include any information except for the function's purpose, arguments or parameters, return values, and exceptions."
            f"Do not mention the docstring is generated or for sphinx. Just return the docstring."
        )
    else:
        prompt = (
            f"Generate a Sphinx-compatible (NumPy format) docstring for the following Python function. "
            f"Include sections for Parameters, Returns, and Raises (if applicable). "
            f"Function name: {func_name}\n"
            f"Function source:\n{func_source}\n"
            f"Return only the docstring text, without any additional explanation."
            f"Do not include any information except for the function's purpose, arguments or parameters, return values, and exceptions."
            f"Do not mention the docstring is generated or for sphinx. Just return the docstring."
        )
    
    payload = {
        "prompt": prompt,
        "max_length": 2000,
        "temperature": 0.01,
        "top_k": 50,
        "top_p": 0.95
    }
    
    print(f"Sending payload for {func_name}: {json.dumps(payload, indent=2)}")
    
    # Use the lock to ensure only one API call happens at a time
    with MODEL_LOCK:
        response = requests.post(api_url, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()
    new_docstring = result.get("response", "").strip()
    new_docstring = re.sub(r'^```[\w]*\n|```$', '', new_docstring, flags=re.MULTILINE).strip()
    if not new_docstring:
        print(f"Warning: Received empty docstring for function '{func_name}'")
    return new_docstring    

def update_function_source(func_source: str, new_docstring: str) -> str:
    lines = func_source.splitlines()
    def_line_idx = 0
    for i, line in enumerate(lines):
        if re.match(r'^(async\s+)?def\s+', line):
            def_line_idx = i
            break
    
    docstring_pattern = re.compile(r'^\s*"""[\s\S]*?"""', re.MULTILINE)
    match = docstring_pattern.search(func_source)
    
    if match:
        start, end = match.span()
        before_doc = func_source[:start]
        after_doc = func_source[end:]
        updated_source = f"{before_doc}    \"\"\"{new_docstring}\"\"\"{after_doc}"
    else:
        indent = '    '
        updated_lines = lines[:def_line_idx + 1] + [f"{indent}\"\"\"{new_docstring}\"\"\""] + lines[def_line_idx + 1:]
        updated_source = '\n'.join(updated_lines)
    
    return updated_source.rstrip()

def load_gitignore(base_dir: str):
    gitignore_path = os.path.join(base_dir, ".gitignore")
    patterns = []
    if os.path.isfile(gitignore_path):
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = f.read().splitlines()
        except Exception as e:
            print(f"Error reading .gitignore at {gitignore_path}: {e}")
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

def process_file(filepath: str, api_url: str = API_URL):
    print(f"Processing file: {filepath}")
    try:
        backup_filepath = filepath + '.bak'
        shutil.copyfile(filepath, backup_filepath)
        print(f"Backup created at: {backup_filepath}")
    except Exception as e:
        print(f"Error creating backup for file {filepath}: {e}")
        return

    functions = split_functions(filepath)
    if not functions:
        print(f"No valid functions found in the file: {filepath}")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return

    # Process functions in reverse order to avoid line number shifts
    functions.sort(key=lambda x: x[3], reverse=True)
    
    for func_name, func_source, existing_docstring, start_line, end_line in functions:
        print(f"\nProcessing function '{func_name}' in file {filepath}")
        try:
            new_docstring = generate_sphinx_docstring(func_name, func_source, existing_docstring, api_url)
        except Exception as e:
            print(f"Error generating new docstring for function '{func_name}': {e}")
            print("Skipping update; retaining the original docstring.")
            continue

        if not new_docstring:
            print(f"Skipping update for function '{func_name}' due to empty docstring.")
            continue

        try:
            updated_source = update_function_source(func_source, new_docstring)
        except Exception as e:
            print(f"Error updating source for function '{func_name}': {e}")
            print("Skipping update; retaining the original docstring.")
            continue

        lines[start_line:end_line + 1] = updated_source.splitlines(keepends=True)
        print(f"Updated function '{func_name}':")
        print(updated_source)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        print(f"File updated successfully: {filepath}")
    except Exception as e:
        print(f"Error writing updates to file {filepath}: {e}")

def process_directory(directory: str, api_url: str = API_URL):
    print(f"Processing directory: {directory}")
    spec = load_gitignore(directory)
    
    for root, dirs, files in os.walk(directory):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not spec.match_file(os.path.relpath(os.path.join(root, d), directory))]
        for filename in files:
            rel_path = os.path.relpath(os.path.join(root, filename), directory)
            if spec.match_file(rel_path):
                continue
            if filename.endswith('.py'):
                file_path = os.path.join(root, filename)
                process_file(file_path, api_url)

if __name__ == "__main__":
    whatsapp_bot_dir = "/home/david/work/whatsapp-bot"
    process_directory(whatsapp_bot_dir)
