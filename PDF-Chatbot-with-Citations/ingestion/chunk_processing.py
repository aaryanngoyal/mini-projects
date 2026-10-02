import os
from typing import List
from langchain_core.documents import Document

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class ChunkerProcessor:

    @staticmethod
    def separate_content(chunk):

        content_data = {
            'text': chunk.page_content,
            'tables': [],
            'images': [],
            'types': ['text']
        }

        if chunk.metadata["category"] == "Table":
            content_data['types'].append("table")
            table_html = chunk.metadata.get("text_as_html", chunk.page_content)
            content_data['tables'].append(table_html)

        elif chunk.metadata["category"] == "Image":
            if "image_base64" in chunk.metadata:
                content_data['types'].append("image")
                content_data['images'].append(chunk.metadata["image_base64"])

        content_data['types'] = list(set(content_data["types"]))
        return content_data

    @staticmethod
    def ai_summary(text: str, tables: List[str], images: List[str]) -> str:

        try:
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

            prompt_text = f"""You are creating a searchable description for document content retrieval.

            CONTENT TO ANALYZE:
            TEXT CONTENT:
            {text}

            """

            if tables:
                prompt_text += "TABLES:\n"
                for i, table in enumerate(tables, 1):
                    prompt_text += f"Table {i+1}:\n{table}\n\n"
        
                    prompt_text += """
                    YOUR TASK:
                    Generate a comprehensive, searchable description that covers:

                    1. Key facts, numbers, and data points from text and tables
                    2. Main topics and concepts discussed  
                    3. Questions this content could answer
                    4. Visual content analysis (charts, diagrams, patterns in images)
                    5. Alternative search terms users might use

                    Make it detailed and searchable - prioritize findability over brevity.

                    SEARCHABLE DESCRIPTION:"""

            text_part = types.Part.from_text(text=prompt_text)

            message_content = [text_part]
            
            if images:
                image_part = types.Part.from_bytes(
                    data=images[0],
                    mime_type="image/jpeg"
                )
                message_content.append(image_part)
            

            llm = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=message_content,
            )

            return llm.text

        except Exception as e:
            print(f"AI summary failed: {e}")
            summary = f"{text[:400]}..."

            if tables:
                summary += f"Contains {len(tables)} tables"
            if images:
                summary += f"Contains {len(images)} tables"
            return summary

    @staticmethod
    def summarize_chunk(chunks: list[Document]) -> list[Document]:

        total_chunk = len(chunks)
        for i, chunk in enumerate(chunks[100:200], 100):
            print(f"Processing Chunk: {i}/{total_chunk}")

            content_data = ChunkerProcessor.separate_content(chunk)

            print(f"Types found: {content_data["types"]}")
            print(f"Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

            if content_data['tables'] or content_data["images"]:
                try:
                    enhanced_content = ChunkerProcessor.ai_summary(
                        content_data['text'],
                        content_data["tables"],
                        content_data['images']
                    )

                    print("AI summary created successfully")

                except Exception as e:
                    print(f"Summary failed: {e}")
                    enhanced_content = content_data["text"]

            else:
                enhanced_content = content_data["text"]

            chunk.page_content = enhanced_content

        return chunks