# agentic-expt1.py
"""
Helper functions for crawling and extracting sections from HTML submissions (webpage-like crawling).
"""
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any
import os

class HTMLCrawler:
    def extract_cells_dynamic(self) -> List[Dict[str, Any]]:
        """
        Dynamically extract logical cells (code, output, markdown, image) in order of appearance.
        For images, saves them to a folder named after the submission file and stores the saved image path and metadata.
        Returns a list of dicts with type and content/path/metadata.
        """
        import base64
        cells = []
        # Prepare image output folder
        base_name = os.path.splitext(os.path.basename(self.html_path))[0]
        img_folder = os.path.join(os.path.dirname(self.html_path), f"{base_name}_images")
        os.makedirs(img_folder, exist_ok=True)
        img_counter = 1
        for tag in self.soup.find_all(True):  # True = all tags
            # Code cell: <pre>, or class contains 'code'
            if tag.name == "pre" or (tag.has_attr("class") and any("code" in c.lower() for c in tag["class"])):
                code_text = tag.get_text(strip=True)
                if code_text:
                    cells.append({"type": "code", "content": code_text})
            # Output cell: class contains 'output'
            elif tag.has_attr("class") and any("output" in c.lower() for c in tag["class"]):
                output_text = tag.get_text(strip=True)
                if output_text:
                    cells.append({"type": "output", "content": output_text})
            # Markdown cell: headers or paragraphs
            elif tag.name in ["p", "h1", "h2", "h3", "h4", "h5", "h6"]:
                md_text = tag.get_text(strip=True)
                if md_text:
                    cells.append({"type": "markdown", "content": md_text})
            # Image
            elif tag.name == "img" and tag.has_attr("src"):
                src = tag["src"]
                img_path = None
                saved_path = None
                if src.startswith("data:image/"):
                    # Base64-encoded image
                    try:
                        header, encoded = src.split(",", 1)
                        ext = header.split("/")[-1].split(";")[0]
                        img_path = os.path.join(img_folder, f"image_{img_counter}.{ext}")
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(encoded))
                        saved_path = img_path
                        img_counter += 1
                    except Exception as e:
                        saved_path = f"ERROR: {e}"
                else:
                    # Regular image file (relative or absolute path)
                    img_path = os.path.join(img_folder, f"image_{img_counter}{os.path.splitext(src)[-1]}")
                    try:
                        # Try to copy the image if it exists locally
                        src_path = os.path.join(os.path.dirname(self.html_path), src)
                        if os.path.exists(src_path):
                            with open(src_path, "rb") as fsrc, open(img_path, "wb") as fdst:
                                fdst.write(fsrc.read())
                            saved_path = img_path
                        else:
                            saved_path = src  # Just store the src if not found
                        img_counter += 1
                    except Exception as e:
                        saved_path = f"ERROR: {e}"
                cells.append({"type": "image", "path": saved_path})
        return cells
    def __init__(self, html_path: str):
        self.html_path = html_path
        self.soup = self._load_html()

    def _load_html(self) -> BeautifulSoup:
        with open(self.html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return BeautifulSoup(html_content, 'html.parser')

    def get_all_sections(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all top-level sections (e.g., <h1>, <h2>, <div>, <section>, etc.)
        Each section is a dict with tag, text, and children (if any).
        """
        sections = []
        for tag in self.soup.find_all(['h1', 'h2', 'h3', 'section', 'article', 'div']):
            section = {
                'tag': tag.name,
                'attrs': dict(tag.attrs),
                'text': tag.get_text(strip=True),
                'children': [child for child in tag.children if getattr(child, 'name', None)]
            }
            sections.append(section)
        return sections

    def get_section_by_id(self, section_id: str) -> Optional[BeautifulSoup]:
        """
        Returns the section/tag with the given id attribute.
        """
        return self.soup.find(id=section_id)

    def get_links(self) -> List[str]:
        """
        Returns all href links in the HTML.
        """
        return [a['href'] for a in self.soup.find_all('a', href=True)]

    def get_images(self) -> List[str]:
        """
        Returns all image src paths in the HTML.
        """
        return [img['src'] for img in self.soup.find_all('img', src=True)]

    def get_tables(self) -> List[BeautifulSoup]:
        """
        Returns all <table> tags in the HTML.
        """
        return self.soup.find_all('table')

# Example usage
if __name__ == "__main__":
    html_path = 'data/submissions/sample-submission-4.html'  # Change as needed
    crawler = HTMLCrawler(html_path)
    # Only print sections with text or children
    print("Sections:", [s for s in crawler.get_all_sections() if s['text'] or s['children']])
    print("Links:", crawler.get_links())
    print("Tables found:", len(crawler.get_tables()))
    print("\nDynamic cell extraction (in order):")
    for cell in crawler.extract_cells_dynamic():
        if cell['type'] == 'image':
            if cell['path'] and not str(cell['path']).startswith('ERROR'):
                print({'type': 'image', 'path': cell['path']})
        elif cell['type'] in ['code', 'output', 'markdown']:
            if cell['content']:
                print(cell)
