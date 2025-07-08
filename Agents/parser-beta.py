from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import base64
import os
import json
import re


# -----------------------------
# Step 1: Code Cell Extraction
# -----------------------------
def extract_code_cells(soup):
    """Extract all Python code cells from the notebook, including those without outputs."""
    code_cells = []
    for cell in soup.find_all("div", class_=re.compile(r"jp-CodeCell")):
        code_block = cell.find("pre")
        if code_block:
            code = code_block.get_text()
            code_cells.append(code.strip())
        else:
            code_cells.append("")  # Handle empty code cells gracefully
    return code_cells


# -----------------------------
# Step 2: Output Cell Extraction
# -----------------------------
def extract_output_cells(soup):
    """Extract all output cells from the notebook."""
    output_cells = []
    for wrapper in soup.find_all("div", class_=re.compile(r"jp-Cell-outputWrapper")):
        # Check if this is a DataFrame/table output
        html_table = wrapper.find("table", class_="dataframe")
        if html_table:
            table_html = str(html_table)
            try:
                df = pd.read_html(StringIO(table_html))[0]
                output = "DataFrame Output:\n" + df.head().to_string()
            except Exception:
                output = "DataFrame Output: Failed to parse"
            output_cells.append(output)
        else:
            # Try to get plain text output
            pre_tag = wrapper.find("pre")
            img_tag = wrapper.find("img")
            if pre_tag:
                output_cells.append(pre_tag.get_text().strip())
            elif img_tag:
                output_cells.append(str(img_tag))
            else:
                # Fallback: capture raw HTML for rich outputs
                output_cells.append("")
    return output_cells


# -----------------------------
# Step 3: Image Extraction & Saving
# -----------------------------
def extract_and_save_images(html_path, output_folder="extracted_images_beta"):
    """
    Extracts and saves Base64-encoded images from HTML.
    Returns list of image paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    img_tags = soup.find_all("img", src=re.compile(r"data:image/.*?;base64"))

    image_paths = []
    for idx, img in enumerate(img_tags):
        src = img["src"]
        header, encoded = src.split(",", 1)
        image_data = base64.b64decode(encoded)
        filename = os.path.join(output_folder, f"image_{idx + 1}.png")

        with open(filename, "wb") as f:
            f.write(image_data)

        image_paths.append(filename)

    return image_paths


# -----------------------------
# Step 4: Main Parsing Function
# -----------------------------
def parse_notebook(html_path, output_json="parsed_output-beta.json"):
    """
    Main function to parse notebook and save results in JSON format.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    code_cells = extract_code_cells(soup)
    output_cells = extract_output_cells(soup)
    image_paths = extract_and_save_images(html_path)

    parsed_data = []
    image_index = 0

    for idx, code in enumerate(code_cells):
        if idx < len(output_cells):
            output = output_cells[idx]
        else:
            output = ""

        # Check if output is an image tag
        if isinstance(output, str) and output.startswith("<img"):
            item = {
                "code": code,
                "output": "",
                "image_path": image_paths[image_index] if image_index < len(image_paths) else None
            }
            if image_index < len(image_paths):
                image_index += 1
        else:
            # Strip DataFrame prefix before saving to JSON
            if isinstance(output, str) and output.startswith("DataFrame Output:"):
                output = output.replace("DataFrame Output:\n", "")
            item = {
                "code": code,
                "output": output,
                "image_path": None
            }

        parsed_data.append(item)

    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2)

    print(f"\nParsed data saved to {output_json}")
    return parsed_data


# -----------------------------
# Step 5: Markdown Report Generation
# -----------------------------
def generate_markdown(parsed_data, output_md="submission_report-beta.md"):
    """
    Generates a Markdown file that preserves the structure of the original notebook.
    """
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("#Student Submission Report\n\n")
        f.write("This report reconstructs the Jupyter Notebook submission from parsed HTML.\n\n")

        for idx, block in enumerate(parsed_data):
            # Write code block
            f.write(f"### Code Block {idx + 1}\n")
            f.write("```python\n")
            f.write(block["code"] + "\n")
            f.write("```\n\n")

            # Write output if it exists
            if block["output"]:
                f.write("#### Output:\n")
                if block["output"].startswith("   Unnamed: 0 ") or '\n' in block["output"]:
                    # Assume it's a DataFrame or tabular output
                    f.write(block["output"] + "\n\n")
                else:
                    f.write("```\n")
                    f.write(block["output"] + "\n")
                    f.write("```\n\n")

            # Write image if present
            if block["image_path"]:
                f.write("#### Output Image:\n")
                f.write(f"![Generated Plot]({block['image_path']}\n\n")

            # Add spacing between blocks
            f.write("---\n\n")

    print(f"\nMarkdown report saved to {output_md}")


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    result = parse_notebook("data/submissions/sample-submission-1.html")

    # Generate Markdown report
    generate_markdown(result, output_md="submission_report-beta.md")

    print("\nFull notebook parsing and Markdown generation complete!")