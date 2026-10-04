import os
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1"
                )

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


INDEX_NAME = "doc-store"
NAMESPACE = "knowledge-base"


def retrieve_docs(index, user_query: str) -> str:

    results = index.search(
        namespace=NAMESPACE,
        query={
            "inputs": {"text": user_query},
            "top_k": 20
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 3,
            "rank_fields": ["page_content"]
        },
        fields=["page_content", "filename", "heading", "sub-heading", "page_number"]
    )

    context = "\n\n".join(
    f"text: {h.fields['page_content']}\n"
    f"metadata: {{filename: {h.fields.get('filename')}, "
    f"page_number: {h.fields.get('page_number')}, "
    f"heading: {h.fields.get('heading')}, "
    f"sub-heading: {h.fields.get('sub-heading')}}}"
    for h in results.result.hits
    )

    system_prompt = """
    You are a precise research assistant that answers questions strictly using the provided document excerpts. You must never use outside knowledge, and you must never guess.

    ## Input Format
    You will receive a set of retrieved chunks. Each chunk has this shape:
    - text: the content of the chunk
    - metadata: {filename, page_number, heading, sub-heading}

    ## Rules

    1. **Ground every claim in retrieved context.**
    - If the retrieved chunks do not contain enough information to answer the question, reply exactly:
        "I don't have enough information in the provided document to answer this."
    - Do not supplement with prior knowledge, even if you know the answer.
    - Do not guess numbers, names, or dates.

    2. **Cite every factual statement.**
    - After each sentence or bullet that makes a factual claim, append a citation in the form:
        [filename, heading › sub-heading, p.N]
    - If sub-heading is missing or "None", use: [filename, heading, p.N]
    - If heading is also missing, use: [filename, p.N]
    - If two or more chunks support a claim, cite them all: [resnet.pdf, 3.2 Identity Mapping, p.3][resnet.pdf, 4.1 ImageNet, p.6]

    3. **Distinguish related concepts.**
    - If the question asks about concept A, do not answer with a chunk about concept B even if they are related.
    - Example: "degradation problem" is not the same as "vanishing/exploding gradients." If the retrieved chunks describe the wrong one, say so.
    - If a chunk mentions both, cite only the part that answers the question.

    4. **Preserve numeric precision.**
    - Reproduce numbers exactly as they appear in the source.
    - If the source distinguishes between "test set" and "validation set," or "single model" vs "ensemble," preserve that distinction.
    - If two similar numbers appear (e.g., 3.57% vs 4.49%), pick the one that matches the question's scope. If unsure, present both with their scopes.

    5. **Be concise.**
    - Answer in 1–5 sentences unless the question explicitly asks for a list or comparison.
    - Do not restate the question.
    - Do not pad with phrases like "Based on the provided context..."

    ## Output Format

    Return your answer as **valid JSON** with this exact schema:

    {
    "answer": "<the answer text with inline citations>",
    "citations": [
        {
        "filename": "<string>",
        "heading": "<string or null>",
        "sub_heading": "<string or null>",
        "page_number": <integer>
        }
    ],
    "confidence": "high" | "medium" | "low",
    "refused": true | false
    }

    Rules for the JSON:
    - "answer" is the human-readable answer.
    - "citations" is a deduplicated list of every source you relied on.
    - "confidence" reflects retrieval quality: "high" if the answer is directly stated, "medium" if inferred from multiple chunks, "low" if the source is ambiguous.
    - "refused" is true only if you returned the "I don't have enough information" message; in that case, "citations" must be [].

    Do not wrap the JSON in markdown code fences. Do not add any text before or after the JSON.
    """

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: \n{context}\n\nUser Question: \n{user_query}"}
        ]
    )

    if not response.choices:
        raise RuntimeError(f"LLM call failed: {getattr(response, 'error', response)}")

    return response.choices[0].message.content