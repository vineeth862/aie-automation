from PIL import Image
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
import fitz
import base64
import io
from langchain_core.messages import HumanMessage
import streamlit as st
class IsometricDataResponse(TypedDict):
    line_number: list = Field(description="Drawing line number, which appears after the keyword 'Title:'")

class PDFLineExtractor:
    def __init__(self, pdf_path: str, batch_size: int = 5):
        self.pdf_path = pdf_path
        self.batch_size = batch_size
        # self.openai_llm = ChatOpenAI(model="gpt-4o")
        self.base_model = ChatAnthropic(
                    model="claude-3-7-sonnet-20250219",
                    temperature=0,
                    api_key=st.secrets["CLAUDE_API_KEY"],
                    max_tokens=64000
                )
        self.claude_llm_with_structured_output = self.base_model.with_structured_output(IsometricDataResponse)
        self.query = """
        Extract only the title names from the provided merged image, ensuring no additional labels or prefixes are included.
        """
        self.pdf_document = None

    def load_pdf(self):
        """Loads the PDF document."""
        try:
            self.pdf_document = fitz.open(self.pdf_path)
        except Exception as e:
            return {"return_code": "PLE_01", "status": f"Failed to open PDF: {str(e)}"}
        return {"return_code": 1, "status": "success"}

    def image_to_base64(self, image: Image):
        """Converts an image to a base64-encoded string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def extract_pdf_batch(self, start_page: int, end_page: int):
        """Extracts and processes a batch of PDF pages into a merged image."""
        try:
            cropped_images = []
            
            for page_num in range(start_page, end_page):
                page = self.pdf_document[page_num]
                page_width, page_height = page.rect.width, page.rect.height
                
                """Crop the image to get only line number details and append that to cropped images"""
                x0 = page_width * 0.7  # 70% from the left
                y0 = page_height * 0.85  # 85% from the top
                x1 = page_width  # Full width
                y1 = page_height * 0.98  # 98% from the top
                bottom_right_rect = fitz.Rect(x0, y0, x1, y1)

                pix = page.get_pixmap(dpi=300, clip=bottom_right_rect)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
                cropped_images.append(img)

            if not cropped_images:
                return {"return_code": "ERR_02", "status": "No images extracted from pages."}

            """Create a single merged images"""
            single_width = cropped_images[0].width
            single_height = cropped_images[0].height
            total_height = single_height * len(cropped_images)
            merged_image = Image.new("RGB", (single_width, total_height))
            
            y_offset = 0
            for img in cropped_images:
                merged_image.paste(img, (0, y_offset))
                y_offset += single_height

            return {"return_code": 1, "status": "success", "image_base64": self.image_to_base64(merged_image)}
        except Exception as e:
            return {"return_code": "ERR_03", "status": f"Failed to process PDF pages: {str(e)}"}

    def extract_line_numbers(self):
        """Extracts line numbers from the PDF using OpenAI's LLM."""
        pdf_load_status = self.load_pdf()
        if pdf_load_status["return_code"] != 1:
            return pdf_load_status
        
        num_pages = self.pdf_document.page_count
        all_responses = []
        
        for start_page in range(0, num_pages, self.batch_size):
            end_page = min(start_page + self.batch_size, num_pages)
            print(f"Processing pages {start_page + 1} to {end_page}...")

            batch_result = self.extract_pdf_batch(start_page, end_page)
            if batch_result["return_code"] != 1:
                return batch_result
            
            image_data = base64.b64decode(batch_result['image_base64'])
            image = Image.open(io.BytesIO(image_data))
            # # image_path = os.path.join(output_folder, f"batch_{start_page+1}_{end_page}.jpg")
            # image.save(image_path, format="PNG")
            # message = HumanMessage(
            #     content=[
            #         {"type": "text", "text": self.query},
            #         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{batch_result['image_base64']}"}},
            #     ],
            # )
            message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": self.query
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{batch_result['image_base64']}"
                    }
                }
            ]
        )
            try:
                response = self.claude_llm_with_structured_output.invoke([message])
                all_responses.extend(response.get('line_number', []))
            except Exception as e:
                return {"return_code": "ERR_04", "status": f"Failed to invoke LLM: {str(e)}"}

        return {"return_code": 1, "status": "success", "line_numbers": all_responses}
