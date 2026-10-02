import re
from langchain_core.documents import Document

CAPTION_RE = re.compile(
    r"^\s*(Figure|Fig\.?|Table)\s+(\d+)\s*[\.\:\-—]?\s*",
    re.I | re.MULTILINE
)

class TableImage:

    @staticmethod
    def is_empty_or_fake_table(doc) -> bool:

        html = doc.metadata["text_as_html"]
        if not html or "<tbody></tbody>" in html or len(re.sub(r"<[^>]+>", "", html).strip()) < 20:
            return True

        prob = doc.metadata["detection_class_prob"]
        if prob < 0.6 and doc.metadata["is_extracted"] == "partial":
            return True
        return False

    @staticmethod
    def reclassify_fake(docs: list[Document]) -> list[Document]:

        for doc in docs:
            if doc.metadata["category"] == "Table" and TableImage.is_empty_or_fake_table(doc):
                doc.metadata["category"] = "Image"

        return docs


    @staticmethod
    def same_table(doc1, doc2) -> bool:

        if doc1.metadata["element_id"] == doc2.metadata["element_id"]:
            return True

        return (doc1.metadata["heading_number"] == doc2.metadata["heading_number"] and
                doc1.metadata["subheading_number"] == doc2.metadata["subheading_number"] and
                doc1.metadata["page_number"] == doc2.metadata["page_number"])

    @staticmethod
    def merge_html(doc1: list[Document], doc2: list[Document]) -> str:

        if not doc1: return doc2
        if not doc2: return doc1
        
        body1 = re.search(r"<tbody>(.*?)</tbody>", doc1, re.S)
        body2 = re.search(r"<tbody>(.*?)</tbody>", doc2, re.S)
        head  = re.search(r"<thead>.*?</thead>", doc1, re.S)
        head_s = head.group(0) if head else ""
        b1 = body1.group(1) if body1 else ""
        b2 = body2.group(1) if body2 else ""

        return f"<table>{head_s}<tbody>{b1}{b2}</tbody></table>"

    @staticmethod
    def merge_split_table(docs: list[Document]) -> list[Document]:

        out = []
        for doc in docs:
            if (out
                and doc.metadata["category"] == "Table"
                and out[-1].metadata["category"] == "Table"
                and TableImage.same_table(out[-1], doc)):

                prev = out[-1]
                prev.page_content += "\n" + doc.page_content
                prev.metadata["text_as_html"] = TableImage.merge_html(
                    prev.metadata["text_as_html"],
                    doc.metadata["text_as_html"]
                )
                continue
            out.append(doc)
            
        return out


    @staticmethod
    def extract_captions(docs: list[Document]):

        captions = {}
        for doc in docs:
            m = CAPTION_RE.search(doc.page_content.strip())
            if m:
                kind = "Figure" if m.group(1).lower().startswith("fig") else "Table"
                num = m.group(2)
                
                key = (kind, num)
                if key not in captions:
                    captions[key] = doc
        return captions

    @staticmethod
    def _top_y(doc):
        pts = doc.metadata["coordinates"]["points"]
        return min(p[1] for p in pts) if pts else 0

    @staticmethod
    def _bottom_y(doc):
        pts = doc.metadata["coordinates"]["points"]
        return max(p[1] for p in pts) if pts else 0

    @staticmethod
    def _center_x(doc):
        pts = doc.metadata["coordinates"]["points"]
        xs = [p[0] for p in pts]
        return (min(xs) + max(xs)) / 2 if pts else 0

    @staticmethod
    def _overlaps_horizontally(doc1, doc2, min_overlap_ratio=0.3):
        p1 = doc1.metadata["coordinates"]["points"]
        p2 = doc2.metadata["coordinates"]["points"]
        x1_min, x1_max = min(p[0] for p in p1), max(p[0] for p in p1)
        x2_min, x2_max = min(p[0] for p in p2), max(p[0] for p in p2)
        overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        width = min(x1_max - x1_min, x2_max - x2_min)
        return (overlap / width) >= min_overlap_ratio if width > 0 else False

    @staticmethod
    def bind_captions(docs: list[Document]) -> list[Document]:

        for doc in docs:
            if doc.metadata["category"] not in ("Image", "Table"):
                continue

            page = doc.metadata["page_number"]
            kind = "Table" if doc.metadata["category"] == "Table" else "Figure"

            candidates = [
                (key, cap) for key, cap in TableImage.extract_captions(docs).items()
                if key[0] == kind and cap.metadata["page_number"] == page
                ]

            if not candidates:
                continue

            y_d = TableImage._bottom_y(doc)

            candidates = [
                (key, cap) for key, cap in candidates
                if TableImage._top_y(cap) >= y_d
                and TableImage._overlaps_horizontally(doc, cap)
            ]

            if not candidates:
                continue    

            best = min(candidates, key=lambda kc: abs(TableImage._top_y(kc[1]) - y_d))

            doc.metadata["caption"] = best[1].page_content
            doc.metadata["caption_number"] = best[0][1]

        return docs