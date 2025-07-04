from bs4 import BeautifulSoup
import re
import base64
import os


def extract_code_cells(soup):
    """Extract all Python code cells from the notebook."""
    code_cells = []
    for cell in soup.find_all("div", class_=re.compile(r"jp-CodeCell")):
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
    for wrapper in soup.find_all("div", class_=re.compile(r"jp-Cell-outputWrapper")):
        output_block = wrapper.find("pre") or wrapper.find("img")
        if output_block:
            if output_block.name == "img":
                # This is an image output
                output_cells.append(str(output_block))
            else:
                # This is text output
                output_cells.append(output_block.get_text().strip())
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


def parse_notebook(html_path):
    """
    Main function to parse notebook and link code with output (text or image).
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
        if len(output_cells) == 0:
            break

        output = output_cells.pop(0)

        # Check if output is an image tag
        if output.startswith("<img"):
            if image_index < len(image_paths):
                parsed_data.append({"code": code,"output_text": "","image_path": image_paths[image_index]})
                image_index += 1
            else:
                parsed_data.append({"code": code,"output_text": "","image_path": None})

        else:
            # It's regular text output
            parsed_data.append({"code": code,"output_text": output,"image_path": None})

    return parsed_data


# Example usage
if __name__ == "__main__":
    result = parse_notebook("data/submissions/sample-submission-1.html")
    for idx, item in enumerate(result):
        print(f"\n--- Code Block {idx + 1} ---")
        print("CODE:")
        print(item["code"])
        if item["output_text"]:
            print("\nOUTPUT TEXT:")
            print(item["output_text"])
        if item["image_path"]:
            print("\nIMAGE SAVED AT:")
            print(item["image_path"])