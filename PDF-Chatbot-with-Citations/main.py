import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from retrieval.retrieve import retrieve_docs

load_dotenv()

INDEX_NAME = "doc-store"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def main():

    print("PDF Chatbot with Citations")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        index = pc.Index(INDEX_NAME)
        raw = retrieve_docs(index, query)
        result = json.loads(raw)

        print(f"Answer:\n{result["answer"]}\n")
        print("Sources:")
        for c in result.get("citations", []):
            heading = c.get("heading") or "—"
            sub = c.get("sub_heading") or ""
            section = f"{heading} › {sub}" if sub and sub != "None" else heading
            print(f"  - {c['filename']} | {section} | p.{c['page_number']}")
        print(f"Confidence: {result.get('confidence', 'unknown')}\n")

if __name__ == "__main__":
    main()