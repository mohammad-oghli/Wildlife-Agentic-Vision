from pathlib import Path
import subprocess
import sys
import re

def extract_python_code(text: str) -> str:
    # This pattern looks for text inside ```python ... ```
    pattern = r"```python\s+(.*?)\s+```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip() # Fallback to original text

def run_code(code, path):
    code = extract_python_code(code)
    script_path = path
    script_path.write_text(code)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=60
    )
    if result.returncode != 0:
            raise RuntimeError(f"Execution failed:\n{result.stderr}")
