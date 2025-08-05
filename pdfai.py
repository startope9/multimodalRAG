import base64
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import io
from PIL import Image
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain.memory import ConversationBufferMemory
import warnings
from llama_index.llms.llama_cpp import LlamaCPP
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import logging
import os
import pymupdf
import time
from sentence_transformers import CrossEncoder

import fitz  # PyMuPDF

MAX_TURNS = 5

def ocr_image_inline(doc, xref):
    pix = fitz.Pixmap(doc, xref)
    pdf_bytes = pix.pdfocr_tobytes()  
    ocr_doc = fitz.open("pdf", pdf_bytes)
    text = ocr_doc[0].get_text()
    ocr_doc.close()
    return text


# VECTOR DB & LLM IMPORTS
# from llama_cpp import Llama
# Torch and projection for image embedding
torch.manual_seed(42)
# --- Embedding setup ---
text_embedder = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Use torchvision ResNet18 for image embeddings (free, local, no API key)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_text_embedding(text):
    return text_embedder.get_text_embedding(text)


# Project image embedding from 1000 to 768 dims
image_proj = nn.Linear(1000, 768, bias=False)
image_proj.weight.data.normal_(mean=0.0, std=0.02)


def get_image_embedding(base64_image):
    image = Image.open(io.BytesIO(
        base64.b64decode(base64_image))).convert("RGB")
    img_tensor = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor)  # (1, 1000)
        projected = image_proj(features)  # (1, 768)
    return projected[0].tolist()


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# Create the directories


def create_directories(base_dir):
    directories = ["images", "text", "tables", "page_images"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)


# Process tables using pdfplumber
def process_tables(pdfplumber_doc, page_num, base_dir, items, filepath):
    print(
        f"[DEBUG] Entering process_tables for page {page_num}, file: {filepath}")
    try:
        page = pdfplumber_doc.pages[page_num]
        tables = page.extract_tables()
        if not tables:
            print(f"[INFO] No tables found on page {page_num}")
            return
        for table_idx, table in enumerate(tables):
            try:
                table_text = "\n".join(
                    [" | ".join(map(str, row)) for row in table])
                table_file_name = f"{base_dir}/tables/{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt"
                with open(table_file_name, 'w', encoding='utf-8') as f:
                    f.write(table_text)
                items.append({"page": page_num, "type": "table",
                             "text": table_text, "path": table_file_name})
            except Exception as e:
                print(
                    f"[ERROR] Failed to process table {table_idx} on page {page_num}: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(
            f"[FATAL] Unexpected error in process_tables for page {page_num}: {str(e)}")
        import traceback
        traceback.print_exc()

# Process links using pdfplumber


def process_links(pdfplumber_doc, page_num, items):
    try:
        page = pdfplumber_doc.pages[page_num]
        links = []
        if hasattr(page, 'hyperlinks'):
            # pdfplumber >=0.10.0
            links = page.hyperlinks
        else:
            # fallback for older versions
            if 'annots' in page.objects:
                for annot in page.objects['annots']:
                    if annot.get('uri'):
                        links.append({'uri': annot['uri']})
        for link in links:
            items.append({"page": page_num, "type": "link", "uri": link.get(
                'uri', ''), "target": link.get('target', '')})
    except Exception as e:
        print(f"[ERROR] Failed to extract links from page {page_num}: {e}")
        import traceback
        traceback.print_exc()

# Process text chunks


def process_text_chunks(text, text_splitter, page_num, base_dir, items, filepath):
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{base_dir}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w') as f:
            f.write(chunk)
        items.append({"page": page_num, "type": "text",
                     "text": chunk, "path": text_file_name})


# Process images
def process_images(page, page_num, base_dir, items, filepath, doc):
    images = page.get_images()
    for idx, image in enumerate(images):
        xref = image[0]
        pix = pymupdf.Pixmap(doc, xref)
        image_name = f"{base_dir}/images/{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png"
        pix.save(image_name)
        with open(image_name, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf8')
        ocr_text = ocr_image_inline(doc, xref)
        items.append({"page": page_num, "type": "image",
                     "path": image_name, "image": encoded_image})
        items.append({
            "page": page_num, "type": "image_ocr",
            "path": image_name, "text": ocr_text
        })

# Process page images


def process_page_images(page, page_num, base_dir, items):
    pix = page.get_pixmap()
    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page",
                 "path": page_path, "image": page_image})

    # --- Pinecone vector DB setup ---
    # Set your Pinecone API key and environment


def PINECONE_INIT(items):
    print(f"[DEBUG] Initializing Pinecone with {len(items)} items")
    
    # Filter items that have embeddings
    valid_items = [item for item in items if 'embedding' in item and item['embedding'] is not None]
    print(f"[DEBUG] Found {len(valid_items)} items with embeddings")
    
    if not valid_items:
        print("[ERROR] No valid items with embeddings found!")
        return None
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Try to delete existing index (ignore errors if it doesn't exist)
    try:
        pc.delete_index("mmrag")
        print("[DEBUG] Deleted existing index")
        time.sleep(5)  # Wait for deletion to complete
    except Exception as e:
        print(f"[DEBUG] Index deletion failed (may not exist): {e}")
    
    index_name = "mmrag"
    embedding_vector_dimension = len(valid_items[0]['embedding'])
    print(f"[DEBUG] Using embedding dimension: {embedding_vector_dimension}")
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"[DEBUG] Creating new index with dimension {embedding_vector_dimension}")
        pc.create_index(
            name=index_name,
            dimension=embedding_vector_dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        time.sleep(10)  # Wait for index creation
    
    index = pc.Index(index_name)
    
    # Prepare data for upload
    upload_data = []
    for i, item in enumerate(valid_items):
        upload_data.append({
            'id': str(i),
            'values': item['embedding'],
            'metadata': {
                'type': item['type'],
                'page': item['page'],
                'text': item.get('text', '')  # Truncate text for metadata
            }
        })
    
    # Upload in batches
    batch_size = 100
    for start in range(0, len(upload_data), batch_size):
        batch = upload_data[start:start+batch_size]
        index.upsert(vectors=batch)
        print(f"[DEBUG] Uploaded batch {start//batch_size + 1}/{(len(upload_data) + batch_size - 1)//batch_size}")
    
    print(f"[DEBUG] Successfully uploaded {len(upload_data)} vectors to Pinecone")
    print(f"[DEBUG] Pinecone index {index} is ready")
    return index, valid_items


def query_pinecone(index, items_list, query_embedding, top_k=5):
    print(f"[DEBUG] Querying Pinecone with embedding dimension: {len(query_embedding)}")
    
    result = index.query(
        vector=query_embedding,
        top_k=top_k, 
        include_metadata=True
    )
    
    matches = result['matches']
    print(f"[DEBUG] Found {len(matches)} matches")
    
    matched_items = []
    for match in matches:
        idx = int(match['id'])
        if idx < len(items_list):
            item = items_list[idx].copy()
            item['similarity_score'] = match['score']
            matched_items.append(item)
            print(f"[DEBUG] Match: {match['score']:.3f} - {item['type']} - {item.get('text', '')[:100]}...")
    
    return matched_items


def invoke_local_llama(llm, query, context_items, chat_history, max_context_tokens=400):
    """
    Improved prompt for better RAG responses
    """
    print(f"[DEBUG] Generating response with {len(context_items)} context items")
    
    # Build context from matched items
    context_sections = []
    for i, item in enumerate(context_items):
        if item['type'] in ['text', 'table', 'image_ocr']:
            content = item.get('text', '').strip()
            if content:
                similarity = item.get('similarity_score', 0)
                context_sections.append(f"[Context {i+1}] (Score: {similarity:.3f})\n{content}")
    
    # Limit context length
    context = "\n\n".join(context_sections)
    
    # Build conversation history
    history_str = ""
    if chat_history:
        # Only include last 2 exchanges to avoid context overflow
        recent_history = chat_history[-1:]
        for h in recent_history:
            history_str += f"Human: {h['user']}\nAssistant: {h['bot']}\n\n"
    
    # Improved prompt with clear instructions
    full_prompt = f"""**System Prompt:**

You are a helpful AI assistant. Your goal is to answer user questions accurately and comprehensively, drawing information from the provided context and considering the conversation history. Avoid generating context from your own knowledge base. If the requested information is not found in the context, clearly state that you lack sufficient information to answer the question. 

**Conversation History:**

{history_str}  

**Context:**

{context} 

**User Question:**

{query}

**Instructions:**

1. Analyze the conversation history and the current user question to understand the user's intent and any implied context.
2. Carefully examine the provided context (retrieved documents) to generate the most relevant information to answer the user's question.
3. Synthesize the information from the context and incorporate it into a coherent and helpful response.
4. If the conversation history provides additional details or clarifications, leverage those to refine the answer and provide more nuanced responses.
5. If the context does not contain enough information to fully answer the question, acknowledge that limitation and avoid generating speculative or inaccurate content.
6. Structure your response clearly, potentially using bullet points or concise paragraphs, depending on the complexity of the answer.
7. Maintain a polite, informative, and helpful tone throughout the interaction.

Generate the response based on the above guidelines.
Response:"""
    
    print(f"[DEBUG] Prompt length: {len(full_prompt)} characters")
    
    output = llm.complete(full_prompt)
    return output.text.strip()


def transform_query(llm, user_query, chat_history, items):
    """
    Improved query transformation with better context awareness
    """
    print(f"[DEBUG] Transforming query: '{user_query}'")
    
    # Get document preview (first few text chunks)
    preview_chunks = []
    for item in items:
        if item.get('type') == 'text':
            text = item.get('text', '').strip()
            if text:
                preview_chunks.append(text)
        if len(preview_chunks) >= 2:
            break
    
    doc_preview = '\n---\n'.join(preview_chunks)[:800]
    
    # Build recent conversation history
    history_str = ""
    if chat_history:
        recent_history = chat_history[-2:]  # Last 2 exchanges
        for h in recent_history:
            history_str += f"Human: {h['user']}\nAssistant: {h['bot'][:1000]}\n\n"
    
    # Improved query transformation prompt
    prompt =  f"""You are a helpful assistant for search in RAG retrieval. 
            Rewrite the user's question to be as clear, consice, 
            **do not add new context or information, just rewrite it**
            use the conversation history and the following document preview to understand the subject matter. 
            **Do NOT introduce unrelated topics. If the question is already clear, return it unchanged. **
            If the query is a compliment or greeting, acknowledge it politely.
            Only return the rewritten question, nothing else.\n\n
            Document preview:\n{doc_preview}\n\n
            Conversation history:\n{history_str}\n\n
            User question: {user_query}\n\n
            Rewritten question:"""
    
    try:
        output = llm.complete(prompt)
        transformed = output.text.strip()
        
        # Fallback to original query if transformation seems to have failed
        if not transformed:
            print(f"[DEBUG] Query transformation failed, using original")
            return user_query
        
        print(f"[DEBUG] Query transformed to: '{transformed}'")
        return transformed
    except Exception as e:
        print(f"[ERROR] Query transformation failed: {e}")
        return user_query


if __name__ == '__main__':
    filepath = input("Enter the path to the PDF file: ").strip()
    base_dir = "data"
    
    # Check if PDF exists
    if not os.path.exists(filepath):
        print(f"[ERROR] PDF file not found: {filepath}")
        exit(1)
    
    create_directories(base_dir=base_dir)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, length_function=len)
    items = []

    # Open both pymupdf and pdfplumber docs
    doc = pymupdf.open(filepath)
    import pdfplumber as _pdfplumber
    pdfplumber_doc = _pdfplumber.open(filepath)
    num_pages = len(doc)
    print(f"[DEBUG] Processing {num_pages} pages from {filepath}")

    # Process each page of the PDF
    for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
        page = doc[page_num]
        text = page.get_text()
        print(f"[DEBUG] Page {page_num} has {len(text)} characters of text")
        
        process_tables(pdfplumber_doc, page_num, base_dir, items, filepath)
        process_links(pdfplumber_doc, page_num, items)
        process_text_chunks(text, text_splitter, page_num,
                            base_dir, items, filepath)
        process_images(page, page_num, base_dir, items, filepath, doc)
        process_page_images(page, page_num, base_dir, items)
    
    pdfplumber_doc.close()
    doc.close()
    
    print(f"[DEBUG] Total items processed: {len(items)}")

    # --- Embedding generation ---
    item_counts = {
        'text': sum(1 for item in items if item['type'] == 'text'),
        'table': sum(1 for item in items if item['type'] == 'table'),
        'image': sum(1 for item in items if item['type'] == 'image'),
        'page': sum(1 for item in items if item['type'] == 'page'),
        'image_ocr': sum(1 for item in items if item['type']=='image_ocr')
    }
    
    print(f"[DEBUG] Item counts: {item_counts}")
    
    counters = dict.fromkeys(item_counts.keys(), 0)
    with tqdm(
        total=len(items),
        desc="Generating embeddings",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
    ) as pbar:
        for item in items:
            item_type = item['type']
            if item_type not in counters:
                # skip types not tracked for embeddings (e.g., 'link')
                continue
            counters[item_type] += 1
            
            try:
                if item_type in ['text', 'table']:
                    text_content = item.get('text', '').strip()
                    if text_content:
                        item['embedding'] = get_text_embedding(text_content)
                    else:
                        item['embedding'] = None
                elif item_type in ['image', 'page']:
                    image_content = item.get('image', '')
                    if image_content:
                        item['embedding'] = get_image_embedding(image_content)
                    else:
                        item['embedding'] = None
                elif item_type == 'image_ocr':
                    txt = item.get('text', '').strip()
                    item['embedding'] = get_text_embedding(txt) if txt else None
                else:
                    item['embedding'] = None
            except Exception as e:
                print(f"[ERROR] Failed to generate embedding for {item_type}: {e}")
                item['embedding'] = None
            
            pbar.set_postfix_str(
                f"Text: {counters['text']}/{item_counts['text']}, "
                f"Table: {counters['table']}/{item_counts['table']}, "
                f"Image: {counters['image']}/{item_counts['image']} ,"
                f"ImageOCR: {counters.get('image_ocr',0)}/{item_counts.get('image_ocr',0)}, "
            )
            pbar.update(1)

    # Check if we have any embeddings
    valid_embeddings = [item for item in items if item.get('embedding') is not None]
    print(f"[DEBUG] Generated {len(valid_embeddings)} valid embeddings")
    
    if not valid_embeddings:
        print("[ERROR] No valid embeddings generated! Cannot proceed.")
        exit(1)

    # Initialize Pinecone
    result = PINECONE_INIT(items)
    if result is None:
        print("[ERROR] Failed to initialize Pinecone")
        exit(1)
    
    vIndex, indexed_items = result

    # --- Local Llama 3.2 LLM setup ---
    llama_model_path = "C:/Models/llama-3.2-1b-instruct-q8_0.gguf"
    
    if not os.path.exists(llama_model_path):
        print(f"[ERROR] Llama model not found: {llama_model_path}")
        exit(1)
    
    llm = LlamaCPP(
        model_path=llama_model_path,
        temperature=0.1,
        max_new_tokens=512,
        context_window=3900,
        verbose=False,
        model_kwargs={"n_gpu_layers": -1}
    )

    # --- Interactive RAG loop with chat history ---
    memory = ConversationBufferMemory()
    chat_history = []
    print("RAG System initialized. Type 'exit' or 'quit' to end the session.")
    print("=" * 50)
    
    try:
        while True:
            user_prompt = input("\nYou: ").strip()
            if user_prompt.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not user_prompt:
                continue
                
            # Transform user query
            improved_query = transform_query(llm, user_prompt, chat_history, items)
            print(f"[Refined query: {improved_query}]")
            
            # Get embeddings and search
            try:
                query_emb = get_text_embedding(improved_query)
                matched_items = query_pinecone(vIndex, indexed_items, query_emb, top_k=5)
                reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

                # Compute rerank scores
                rerank_scores = reranker.predict([(improved_query, item['text']) for item in matched_items])
                for item, score in zip(matched_items, rerank_scores):
                    item['rerank_score'] = float(score)

                matched_items = sorted(matched_items, key=lambda x: x['rerank_score'], reverse=True)[:3]
                # Debug print before sorting
                print("\n🔍 Rerank Scores:")
                for i, item in enumerate(matched_items):
                    print(f"{i+1}. Score: {item['rerank_score']:.4f} | Text: {item['text'][:80]}...")

        
                # Generate response
                response = invoke_local_llama(llm, improved_query, matched_items, chat_history)
                print(f"\nAssistant: {response}")
                
                # Show references
                if matched_items:
                    print(f"\n[References:")
                    for i, ref_item in enumerate(matched_items, 1):
                        path = ref_item.get('path', '[no path]')
                        item_type = ref_item.get('type', 'unknown')
                        score = ref_item.get('similarity_score', 0)
                        print(f"  {i}. {item_type} (score: {score:.3f}): {path}")
                    print("]")
                
                # Update chat history
                chat_history.append({"user": user_prompt, "bot": response})
                if len(chat_history) > MAX_TURNS:
                    chat_history[:] = chat_history[-MAX_TURNS:]
                memory.save_context({"user": user_prompt}, {"bot": response})
                
            except Exception as e:
                print(f"[ERROR] Failed to process query: {e}")
                import traceback
                traceback.print_exc()
            
    except KeyboardInterrupt:
        print("\nSession interrupted. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Explicitly release llama-cpp resources to avoid destructor bug
        if hasattr(llm, 'release'):
            llm.release()
        print("Resources cleaned up.")
