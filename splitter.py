import ast
import os
from typing import List, Tuple

def split_functions(filepath: str) -> List[Tuple[str, str]]:
    """
    Split a Python file into individual functions based on function definitions.
    
    Args:
        filepath (str): Absolute path to the Python file.
        
    Returns:
        List[Tuple[str, str]]: List of tuples, each containing the function name and its source code.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        SyntaxError: If the file contains invalid Python syntax.
    """
    # Check if file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Read the file content
    with open(filepath, 'r', encoding='utf-8') as file:
        source = file.read()
        lines = source.splitlines()
    
    # Parse the source code into an AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise SyntaxError(f"Invalid Python syntax in {filepath}: {str(e)}")
    
    # List to store function info (name, source code)
    functions = []
    
    # Iterate through all nodes to find function definitions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get function name
            func_name = node.name
            
            # Get the start and end line numbers
            start_line = node.lineno - 1  # Convert to 0-based indexing
            # Find the end line by looking at the last child node or the node itself
            end_line = max(
                (getattr(child, 'end_lineno', node.end_lineno) or node.end_lineno) - 1
                for child in ast.walk(node)
            )
            
            # Extract the source code for the function
            func_source = '\n'.join(lines[start_line:end_line + 1]).rstrip()
            
            # Add to the list
            functions.append((func_name, func_source))
    
    return functions

if __name__ == "__main__":
    absolute_path = "/home/david/work/whatsapp-bot/managers.py"
    functions = split_functions(absolute_path)
    for func_name, func_source in functions:
        print(f"Function Name: {func_name}")
        print("Source Code:")
        print(func_source)
        print("-" * 40)