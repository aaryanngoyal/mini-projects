import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document


HEADING_CATEGORY = {'Title', "Header"}
BODY_CATEGORY = {'UncategorizedText', 'ListItem', 'Image', 'FigureCaption', 'NarrativeText', 'Formula', 'Table'}

_HEADING_NUM_RE = re.compile(r"^([A-Z]|\d+(?:\.\d+)*)\.?\s+(.*)$") # "1", "1.", "1.2", "1.2.", "1.2.3", "1.2.3.", "A", "A.", "A.1", "A.1."
_RUNIN_RE = re.compile(r"(?:\s+(.*))?$")

_BAD_HEADINGS = [
    re.compile(r"^\s*$"),
    re.compile(r"^\d+\s*$"),
    re.compile(r"^Figure\s+\d+", re.I),
    re.compile(r"^Table\s+\d+", re.I),
]

MAX_HEAD_CHARS = 100
MIN_HEAD_CHARS = 2


def partition_documents(docs_path: str):

    loader = UnstructuredPDFLoader(
        file_path=docs_path,
        strategy="hi_res",
        mode="elements",
        infer_table_structure=True,
        include_page_breaks=False
        )

    docs = loader.load()
    return docs

def classify_heading(text: str) -> tuple[str, str | None]:

    m = _HEADING_NUM_RE.match(text)
    if not m:
        return "main", None
    num, _title = m.group(1), m.group(2)
    depth = num.count(".") + 1
    return ("main" if depth == 1 else "sub"), num

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

def parent_number(num:str) -> str | None:

    parts = num.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None

def add_to_metadata(docs: list[Document]) -> list[Document]:

    current_heading = "Uncategorized"
    current_subheading = "Uncategorized"
    current_heading_num = None
    current_subheading_num = None

    for doc in docs:

        category = doc.metadata.get("category", "")
        text = doc.page_content.strip()

        if category in HEADING_CATEGORY and is_valid_heading(text):
            kind, num = classify_heading(text)

            if kind == "main":
                current_heading = text
                current_heading_num = num

                current_subheading = "Uncategorized"
                current_subheading_num = None

            else:
                if num:
                    expected_parent = parent_number(num) 
                    if current_heading_num != expected_parent:
                        current_heading     = current_heading
                        current_heading_num = expected_parent
                        current_subheading  = "Uncategorized"
                        current_subheading_num = None
                
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

docs = partition_documents("./docs/resnet.pdf")

docs = add_to_metadata(docs)

for doc in docs:
    if doc.metadata["category"] == "NarrativeText":
        print(f"{doc.metadata["heading"]} | {doc.metadata["sub-heading"]}")
        print(f"{doc.page_content[:80]}...", "\n")