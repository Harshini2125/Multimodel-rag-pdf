# ============================================================
# MULTIMODAL PDF RAG (GOOGLE COLAB)
# Layout-Aware PDF Parsing + Tables + Images + Gemini + FAISS
# ============================================================

# -------------------------
# INSTALL (RUN ONCE)
# -------------------------
# !pip -q install docling google-generativeai sentence-transformers faiss-cpu pymupdf pillow numpy pandas tqdm

# -------------------------
# API KEY
# -------------------------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import uuid
import faiss
import fitz
import numpy as np

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

import google.generativeai as genai
from docling.document_converter import DocumentConverter

# ============================================================
# GOOGLE COLAB PDF UPLOAD
# ============================================================

PDF_PATH = "/content/Multimode-rag-pdf/samplepaper.pdf"

# ============================================================
# GEMINI
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)

gemini = genai.GenerativeModel(
    "gemini-2.5-pro"
)

# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ============================================================
# PDF PARSING (DOCLING)
# ============================================================

print("Parsing PDF...")

converter = DocumentConverter()

result = converter.convert(PDF_PATH)

doc = result.document

markdown_content = doc.export_to_markdown()

print("PDF Parsed Successfully")

# ============================================================
# IMAGE EXTRACTION
# ============================================================

print("Extracting Images...")

image_dir = Path("images")
image_dir.mkdir(exist_ok=True)

pdf = fitz.open(PDF_PATH)

images = []

image_counter = 1

for page_idx in range(len(pdf)):

    page = pdf[page_idx]

    page_images = page.get_images(full=True)

    for img in page_images:

        xref = img[0]

        extracted = pdf.extract_image(xref)

        image_bytes = extracted["image"]

        image_ext = extracted["ext"]

        image_path = image_dir / f"Image_{image_counter}.{image_ext}"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        images.append(
            {
                "image_id": f"Image_{image_counter}",
                "page": page_idx + 1,
                "path": str(image_path)
            }
        )

        image_counter += 1

print(f"Images Extracted: {len(images)}")

# ============================================================
# IMAGE UNDERSTANDING
# ============================================================

def describe_image(image_path):

    image = Image.open(image_path)

    prompt = """
    Analyze this image thoroughly.

    If it contains:
    - diagram
    - chart
    - table
    - workflow
    - architecture
    - process flow

    explain them in detail.

    Return a detailed description.
    """

    response = gemini.generate_content(
        [prompt, image]
    )

    return response.text

print("Generating Image Descriptions...")

for item in tqdm(images):

    try:
        item["description"] = describe_image(
            item["path"]
        )
    except Exception as e:
        item["description"] = f"Description failed: {str(e)}"

# ============================================================
# TABLE EXTRACTION
# ============================================================

tables = []

if hasattr(doc, "tables"):

    for idx, table in enumerate(doc.tables, start=1):

        try:

            df = table.export_to_dataframe()

            tables.append(
                {
                    "table_id": f"Table_{idx}",
                    "data": df.to_dict(
                        orient="records"
                    )
                }
            )

        except:
            pass

print(f"Tables Extracted: {len(tables)}")

# ============================================================
# STRUCTURED DOCUMENT
# ============================================================

structured_document = {
    "document_name": PDF_PATH,
    "content": markdown_content,
    "tables": tables,
    "images": images
}

with open(
    "structured_document.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        structured_document,
        f,
        indent=2,
        ensure_ascii=False
    )

# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text, chunk_size=1200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start = end

    return chunks

# ============================================================
# KNOWLEDGE BASE
# ============================================================

knowledge_base = []

text_chunks = chunk_text(markdown_content)

for chunk in text_chunks:

    knowledge_base.append(
        {
            "id": str(uuid.uuid4()),
            "type": "text",
            "content": chunk,
            "page": None
        }
    )

for table in tables:

    knowledge_base.append(
        {
            "id": table["table_id"],
            "type": "table",
            "content": json.dumps(
                table,
                ensure_ascii=False
            ),
            "page": None
        }
    )

for image in images:

    knowledge_base.append(
        {
            "id": image["image_id"],
            "type": "image",
            "content": image["description"],
            "page": image["page"]
        }
    )

print("Knowledge Base Size:", len(knowledge_base))

# ============================================================
# EMBEDDINGS
# ============================================================

print("Generating Embeddings...")

texts = [
    item["content"]
    for item in knowledge_base
]

embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

# ============================================================
# FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(embeddings)

faiss.write_index(
    index,
    "knowledge_base.faiss"
)

print("FAISS Index Ready")

# ============================================================
# DIRECT REFERENCE RESOLUTION
# ============================================================

def resolve_direct_reference(query):

    query = query.lower()

    table_match = re.search(
        r"table\s+(\d+)",
        query
    )

    if table_match:

        table_id = f"Table_{table_match.group(1)}"

        return [
            x for x in knowledge_base
            if x["id"] == table_id
        ]

    image_match = re.search(
        r"image\s+(\d+)",
        query
    )

    if image_match:

        image_id = f"Image_{image_match.group(1)}"

        return [
            x for x in knowledge_base
            if x["id"] == image_id
        ]

    return None

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query, k=10):

    direct = resolve_direct_reference(query)

    if direct:
        return direct

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices:
        if idx < len(knowledge_base):
            results.append(knowledge_base[idx])

    return results

# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(results):

    context_blocks = []

    for item in results:

        block = f"""
TYPE: {item['type']}
ID: {item['id']}
PAGE: {item['page']}

CONTENT:
{item['content']}
"""

        context_blocks.append(block)

    return "\n\n".join(context_blocks)

# ============================================================
# QA ENGINE
# ============================================================

def ask_pdf(question):

    retrieved = retrieve(
        question,
        k=10
    )

    context = build_context(
        retrieved
    )

    prompt = f"""
You are a PDF expert.

Answer ONLY using the provided context.

Question:
{question}

Context:
{context}

Requirements:
- Be factual
- Use table information if available
- Use image descriptions if available
- Mention source IDs used
"""

    response = gemini.generate_content(
        prompt
    )

    return response.text

# ============================================================
# INTERACTIVE CHAT
# ============================================================

print("\n" + "="*60)
print("MULTIMODAL PDF RAG READY")
print("="*60)

while True:

    question = input(
        "\nAsk Question (or 'exit'): "
    )

    if question.lower() == "exit":
        break

    try:

        answer = ask_pdf(question)

        print("\n")
        print(answer)

    except Exception as e:

        print("\nERROR:")
        print(e)
