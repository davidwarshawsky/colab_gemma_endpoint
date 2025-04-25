import ast
import os
import requests
import re
import json
from dotenv import load_dotenv
from typing import List, Tuple

# Load environment variables
load_dotenv()
API_URL = os.getenv("api_url")
if not API_URL:
    raise ValueError("api_url not found in .env file")

def split_functions(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Split a Python file into individual functions based on function definitions.
    
    Args:
        filepath (str): Absolute path to the Python file.
        
    Returns:
        List[Tuple[str, str, str]]: List of tuples, each containing the function name, 
        its source code, and its existing docstring (if any).
        
    Raises:
        FileNotFoundError: If the file does not exist.
        SyntaxError: If the file contains invalid Python syntax.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as file:
        source = file.read()
        lines = source.splitlines()
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid Python syntax in {filepath}: {str(e)}")
    
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            start_line = node.lineno - 1
            end_line = max(
                (getattr(child, 'end_lineno', node.end_lineno) or node.end_lineno) - 1
                for child in ast.walk(node)
            )
            func_source = '\n'.join(lines[start_line:end_line + 1]).rstrip()
            docstring = ast.get_docstring(node, clean=True) or ""
            functions.append((func_name, func_source, docstring))
    
    return functions

def generate_sphinx_docstring(func_name: str, func_source: str, existing_docstring: str, api_url: str) -> str:
    """
    Generate or enhance a Sphinx-compatible docstring for a function using the Gemma AI model.
    
    Args:
        func_name (str): Name of the function.
        func_source (str): Full source code of the function.
        existing_docstring (str): Existing docstring, if any.
        api_url (str): URL of the API endpoint for the Gemma model.
        
    Returns:
        str: The generated or enhanced Sphinx-compatible docstring.
        
    Raises:
        requests.RequestException: If the API call fails.
    """
    if existing_docstring:
        prompt = (
            f"Improve the following Python function's docstring to be Sphinx-compatible (NumPy format). "
            f"Ensure it includes sections for Parameters, Returns, and Raises (if applicable), and maintain the original intent. "
            f"Function name: {func_name}\n"
            f"Existing docstring: {existing_docstring}\n"
            f"Function source:\n{func_source}\n"
            f"Return only the improved docstring text, without any additional explanation."
        )
    else:
        prompt = (
            f"Generate a Sphinx-compatible (NumPy format) docstring for the following Python function. "
            f"Include sections for Parameters, Returns, and Raises (if applicable). "
            f"Function name: {func_name}\n"
            f"Function source:\n{func_source}\n"
            f"Return only the docstring text, without any additional explanation."
        )
    
    payload = {
        "prompt": prompt,
        "max_length": 500,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        new_docstring = result.get("response", "").strip()
        new_docstring = re.sub(r'^```[\w]*\n|```$', '', new_docstring, flags=re.MULTILINE).strip()
        return new_docstring
    except requests.RequestException as e:
        raise requests.RequestException(f"API call failed: {e}")

def update_function_source(func_source: str, new_docstring: str) -> str:
    """
    Update a function's source code with a new docstring.
    
    Args:
        func_source (str): Original source code of the function.
        new_docstring (str): New docstring to insert or replace.
        
    Returns:
        str: Updated source code with the new docstring.
    """
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

def main(filepath: str, api_url: str = API_URL):
    """
    Process functions in a Python file, enhance their docstrings, and print the results.
    
    Args:
        filepath (str): Absolute path to the Python file.
        api_url (str): URL of the Gemma API endpoint.
    """
    try:
        functions = split_functions(filepath)
        if not functions:
            print("No functions found in the file.")
            return
        
        for func_name, func_source, existing_docstring in functions:
            print(f"\nFunction: {func_name}")
            print("-" * 50)
            new_docstring = generate_sphinx_docstring(func_name, func_source, existing_docstring, api_url)
            updated_source = update_function_source(func_source, new_docstring)
            print("Updated Source Code:")
            print(updated_source)
            print("-" * 50)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except SyntaxError as e:
        print(f"Error: {e}")
    except requests.RequestException as e:
        print(f"Error calling API: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    absolute_path = "/home/david/work/whatsapp-bot/managers.py"
    main(absolute_path)