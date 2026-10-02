import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document

from tab_img_processor import TableImage
from chunk_processing import ChunkerProcessor


HEADING_CATEGORY = {'Title', "Header"}
BODY_CATEGORY = {'UncategorizedText', 'ListItem', 'Image', 'FigureCaption', 'NarrativeText', 'Formula', 'Table'}

_HEADING_NUM_RE = re.compile(r"^([A-Z]|\d+(?:\.\d+)*)\.?\s+(.*)$") # "1", "1.", "1.2", "1.2.", "1.2.3", "1.2.3.", "A", "A.", "A.1", "A.1."
_RUNIN_RE = re.compile(r"^([A-Z][\w\s,–—\-]+?\.)\s+(.+)$")

_BAD_HEADINGS = [
    re.compile(r"^\s*$"),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^Figure\s+\d+", re.I),
    re.compile(r"^Table\s+\d+", re.I),
]

MAX_HEAD_CHARS = 100
MIN_HEAD_CHARS = 2

class Loader:

    @staticmethod
    def partition_documents(docs_path: str):

        loader = UnstructuredPDFLoader(
            file_path=docs_path,
            strategy="hi_res",
            mode="elements",
            infer_table_structure=True,
            include_page_breaks=False,
            extract_image_block_types=["Image", "Table"], 
            extract_image_block_to_payload=True
            )

        docs = loader.load()
        return docs

    @staticmethod
    def dedupe(docs: list[Document]) -> list[Document]:

        seen = set()
        out = []

        for doc in docs:
            key = doc.metadata["element_id"]
            if key is None:
                key = (doc.metadata["page_number"],
                    doc.metadata["category"],
                    doc.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append(doc)
        return out

    @staticmethod
    def classify_heading(text: str) -> tuple[str, str | None]:

        m = _HEADING_NUM_RE.match(text)
        if not m:
            return "main", None
        num, _title = m.group(1), m.group(2)
        depth = num.count(".") + 1
        return ("main" if depth == 1 else "sub"), num

    @staticmethod
    def is_valid_heading(text: str):

        t = text.strip()
        if not MIN_HEAD_CHARS <= len(t) <= MAX_HEAD_CHARS:
            return False
        
        if t.endswith(("?", "!", ",")) and not _HEADING_NUM_RE.match(t):
            return False

        if _RUNIN_RE.match(t):
            return False
        
        for pat in _BAD_HEADINGS:
            if pat.match(t):
                return False

        if not any(c.isalpha() for c in t):
            return False
        return True

    @staticmethod
    def add_to_metadata(docs: list[Document]) -> list[Document]:

        current_heading = "Uncategorized"
        current_subheading = "Uncategorized"
        current_heading_num = None
        current_subheading_num = None

        for doc in docs:

            category = doc.metadata.get("category", "")
            text = doc.page_content.strip()

            if category in HEADING_CATEGORY and Loader.is_valid_heading(text):
                kind, num = Loader.classify_heading(text)

                if kind == "main":
                    current_heading = text
                    current_heading_num = num

                    current_subheading = "Uncategorized"
                    current_subheading_num = None

                else:
                    current_subheading = text
                    current_subheading_num = num

                doc.metadata["heading"] = current_heading
                doc.metadata["sub-heading"] = current_subheading
                doc.metadata["section_kind"] = kind
                doc.metadata["section_number"] = num
                continue

            if category in BODY_CATEGORY:

                doc.metadata["heading"] = current_heading
                doc.metadata["sub-heading"] = current_subheading
                doc.metadata["heading_number"] = current_heading_num
                doc.metadata["subheading_number"] = current_subheading_num

        return docs

    @staticmethod
    def split_documents(docs: list[Document]) -> list[Document]:

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=600,
            chunk_overlap=0
        )

        chunks = splitter.split_documents(docs)

        for i, doc in enumerate(chunks, 1):
            doc.metadata["chunk_no"] = i

        return chunks


    def loading_pipeline() -> list[Document]:

        docs = Loader.partition_documents("./docs/resnet.pdf")
        docs = Loader.add_to_metadata(docs)
        docs = TableImage.merge_split_table(docs)
        docs = TableImage.reclassify_fake(docs)
        docs = TableImage.bind_captions(docs)
        docs = Loader.dedupe(docs)
        chunks = Loader.split_documents(docs)
        chunks = ChunkerProcessor.summarize_chunk(chunks)

if __name__ == "__main__":
    Loader.loading_pipeline()