from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import base64
import os
import json
import re


def extract_code_cells(soup):
    """Extract all Python code cells from the notebook, including those without outputs."""
    code_cells = []
    for cell in soup.find_all("div", class_=re.compile(r"jp-CodeCell")):
        code_block = cell.find("pre")
        if code_block:
            code = code_block.get_text()
            code_cells.append(code.strip())
        else:
            # Optional: Handle code cells that might be empty or malformed
            code_cells.append("")  # Or skip appending if desired
    return code_cells


def extract_output_cells(soup):
    """Extract all output cells from the notebook."""
    output_cells = []
    for wrapper in soup.find_all("div", class_=re.compile(r"jp-Cell-outputWrapper")):
        # Check if this is a dataframe output
        html_table = wrapper.find("table", class_="dataframe")
        if html_table:
            # Convert HTML table to pandas DataFrame
            table_html = str(html_table)
            df = pd.read_html(StringIO(table_html))[0]
            # Convert DataFrame to clean string representation
            output = "DataFrame Output:\n" + df.head().to_string()
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
                output_cells.append(str(wrapper).strip())
    return output_cells


def extract_and_save_images(html_path, output_folder="extracted_images"):
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


def parse_notebook(html_path, output_json="parsed_output.json"):
    """
    Main function to parse notebook and save results in JSON format.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    code_cells = extract_code_cells(soup)
    output_cells = extract_output_cells(soup)
    image_paths = extract_and_save_images(html_path)

    # Match images to output cells
    parsed_data = []
    image_index = 0

    for code in code_cells:
        if not output_cells:
            # No more output cells — assign null output
            parsed_data.append({
                "code": code,
                "output": "",
                "image_path": None
            })
            continue

        output = output_cells.pop(0)

        # Check if output is an image tag
        if output.startswith("<img"):
            item = {
                "code": code,
                "output": "",
                "image_path": image_paths[image_index] if image_index < len(image_paths) else None
            }
            if image_index < len(image_paths):
                image_index += 1
        else:
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


def generate_markdown(parsed_data, output_md="submission_report.md"):
    """
    Generates a Markdown file that preserves the structure of the original notebook.
    
    Args:
        parsed_data (list): List of dictionaries with 'code', 'output', and 'image_path'
        output_md (str): Path to save the generated Markdown file
    """
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Student Submission Report\n\n")
        f.write("This report reconstructs the Jupyter Notebook submission from parsed HTML.\n\n")

        for idx, block in enumerate(parsed_data):
            # Write code block header
            f.write(f"### Code Block {idx + 1}\n")
            
            # Write code
            f.write("```python\n")
            f.write(block["code"] + "\n")
            f.write("```\n\n")

            # Write output if it exists
            if block["output"]:
                f.write("#### Output:\n")
                if block["output"].startswith("DataFrame Output:"):
                    # Pretty format DataFrame output
                    df_output = block["output"].replace("DataFrame Output:\n", "")
                    f.write(df_output + "\n\n")
                else:
                    # Regular text output
                    f.write("```\n")
                    f.write(block["output"] + "\n")
                    f.write("```\n\n")
            
            # Write image if present
            if block["image_path"]:
                f.write("#### Output Image:\n")
                f.write(f"![Generated Plot]({block['image_path']})\n")
                f.write("\n")

            # Add spacing between blocks
            f.write("---\n\n")

    print(f"\nMarkdown report saved to {output_md}")


# Example usage
if __name__ == "__main__":
    result = parse_notebook("data/submissions/sample-submission-1.html")
    for idx, item in enumerate(result):
        print(f"\n--- Code Block {idx + 1} ---")
        print("CODE:")
        print(item["code"])
        if item["output"]:
            print("\nOUTPUT TEXT:")
            print(item["output"])
        if item["image_path"]:
            print("\nIMAGE SAVED AT:")
            print(item["image_path"])
    generate_markdown(result, output_md="submission_report.md")
    