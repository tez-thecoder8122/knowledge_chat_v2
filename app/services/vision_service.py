import os
import fitz
import io
import json
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

import cv2
import numpy as np
from pdf2image import convert_from_path
import pdfplumber
from PIL import Image
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class VisionService:
    """Service for extracting and processing visual content from PDFs"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    @staticmethod
    def extract_images_from_pdf(file_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        extracted = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                filename = f"page{page_num+1}_img{img_idx+1}.{ext}"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                extracted.append({
                    "filepath": filepath,
                    "format": ext,
                    "page_number": page_num+1,
                    "description": "",
                })
                print(f"Extracted: {filepath}")
        return extracted
    
    @staticmethod
    def render_pdf_pages_as_images(pdf_path: str, output_dir: str) -> list:
        os.makedirs(output_dir, exist_ok=True)
        images_info = []
        pages = convert_from_path(pdf_path, dpi=200)
        for i, page in enumerate(pages):
            filepath = os.path.join(output_dir, f"pdfpage_{i+1}.png")
            page.save(filepath, "PNG")
            images_info.append({
                "filepath": filepath,
                "format": "png",
                "page_number": i + 1,
                "description": "Rendered full page image"
            })
        return images_info
    
    
    @staticmethod
    def extract_tables_from_pdf(pdf_path: str) -> List[Dict]:
        """
        Extract tables from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of table metadata dicts
        """
        try:
            tables_data = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    if not tables:
                        continue
                    
                    for table_idx, table in enumerate(tables):
                        # Convert table to different formats
                        csv_data = VisionService._table_to_csv(table)
                        html_data = VisionService._table_to_html(table)
                        json_data = VisionService._table_to_json(table)
                        
                        tables_data.append({
                            "page_number": page_num,
                            "table_index": table_idx,
                            "csv": csv_data,
                            "html": html_data,
                            "json": json_data,
                            "rows": len(table),
                            "cols": len(table[0]) if table else 0
                        })
                        
                        logger.info(f"Extracted table from page {page_num}, table {table_idx}")
            
            return tables_data
        
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {e}")
            return []
    
    @staticmethod
    def _table_to_csv(table: List[List]) -> str:
        """Convert table to CSV format"""
        csv_lines = []
        for row in table:
            csv_lines.append(",".join([str(cell or "") for cell in row]))
        return "\n".join(csv_lines)
    
    @staticmethod
    def _table_to_html(table: List[List]) -> str:
        """Convert table to HTML format"""
        html = "<table border='1'>\n"
        for row in table:
            html += "  <tr>\n"
            for cell in row:
                html += f"    <td>{cell or ''}</td>\n"
            html += "  </tr>\n"
        html += "</table>"
        return html
    
    @staticmethod
    def _table_to_json(table: List[List]) -> str:
        """Convert table to JSON format"""
        if not table:
            return "[]"
        
        headers = table[0]
        rows = []
        for row in table[1:]:
            row_dict = {headers[i]: row[i] for i in range(len(headers))}
            rows.append(row_dict)
        
        return json.dumps(rows, indent=2)
    
    def analyze_image_with_vision(self, image_path: str) -> str:
        """
        Use GPT-4 Vision to extract text and describe image content.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Text description of image content
        """
        try:
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Determine image format
            img_format = Path(image_path).suffix.lower().replace(".", "")
            if img_format not in ["jpeg", "jpg", "png", "gif", "webp"]:
                img_format = "png"
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{img_format};base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": """Analyze this image and provide:
                                        1. A detailed description of what's shown
                                        2. Any text visible in the image
                                        3. Key elements or data points
                                        4. Context or purpose of this image
                                        Be concise but comprehensive."""
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            description = response.choices[0].message.content
            logger.info(f"Analyzed image: {image_path}")
            return description
        
        except Exception as e:
            logger.error(f"Error analyzing image with vision: {e}")
            return "Unable to analyze image"
    
    @staticmethod
    def get_image_as_base64(image_path: str) -> str:
        """Convert image file to base64 string for API response"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""
    
    @staticmethod
    def check_image_relevance_to_text(image_description: str, text_chunk: str) -> float:
        """
        Simple relevance scoring between image description and text.
        Can be enhanced with embeddings for better accuracy.
        
        Args:
            image_description: Description of image
            text_chunk: Text chunk to compare
        
        Returns:
            Relevance score (0-1)
        """
        # Convert to lowercase for comparison
        desc_lower = image_description.lower()
        text_lower = text_chunk.lower()
        
        # Simple keyword overlap
        desc_words = set(desc_lower.split())
        text_words = set(text_lower.split())
        
        overlap = len(desc_words & text_words)
        union = len(desc_words | text_words)
        
        if union == 0:
            return 0.0
        
        return overlap / union