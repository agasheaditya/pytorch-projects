from bs4 import BeautifulSoup
import re
import os


def extract_code_cells(soup):
    """Extract all Python code cells from the notebook."""
    code_cells = []
    for cell in soup.find_all("div", class_=re.compile(r"jp-CodeCell")):
        # Skip cells without outputs
        if "jp-mod-noOutputs" in cell.get("class", []):
            continue

        code_block = cell.find("pre")
        if code_block:
            code = code_block.get_text()
            code_cells.append(code.strip())
    return code_cells


def extract_output_cells(soup):
    """Extract all output cells from the notebook."""
    output_cells = []
    for cell in soup.find_all("div", class_=re.compile(r"jp-Cell-outputWrapper")):
        output_block = cell.find("pre")
        if output_block:
            output = output_block.get_text()
            output_cells.append(output.strip())
        else:
            # Handle rich outputs like plots or HTML
            output_cells.append(str(cell).strip())
    return output_cells


def parse_notebook(html_path):
    """
    Main function to parse an HTML-exported Jupyter Notebook.
    Returns a list of structured code/output pairs.
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    code_cells = extract_code_cells(soup)
    breakpoint()
    output_cells = extract_output_cells(soup)

    # Pair code with corresponding output (skip no-output cells)
    parsed_data = []
    output_idx = 0
    for cell in soup.find_all("div", class_=re.compile(r"jp-CodeCell")):
        if "jp-mod-noOutputs" in cell.get("class", []):
            continue

        code_block = cell.find("pre")
        if not code_block:
            continue

        code = code_block.get_text().strip()

        # Match with output if available
        if output_idx < len(output_cells):
            output = output_cells[output_idx]
            output_idx += 1
        else:
            output = ""

        parsed_data.append({
            "code": code,
            "output": output
        })

    return parsed_data


# Example usage
if __name__ == "__main__":
    result = parse_notebook("data/submissions/sample-submission-1.html")  # rename to .html if needed
    for idx, item in enumerate(result):
        print(f"\n--- Code Block {idx + 1} ---")
        print("CODE:")
        print(item["code"])
        if item["output"]:
            print("\nOUTPUT:")
            print(item["output"])