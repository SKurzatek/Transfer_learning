import os
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import tqdm

CHECKPOINT_INTERVAL = 320
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = ["A photo taken inside a building", "A photo taken outside the building"]

INPUT_PARQUET = "train-00000-of-00330.parquet"
OUTPUT_DIR = "CLIP_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === CLIP MODEL SETUP ===
def load_clip():
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    return model, processor

# === LLaMA CHAIN SETUP ===
def load_llama_chain():
    prompt_template = ChatPromptTemplate.from_template(
        """Classify the following image caption as either "inside" or "outside".

Caption: {caption}
Answer:"""
    )
    model = ChatOllama(model="llama3.1:8b")  # Change if you use a different model name
    chain = {"caption": RunnablePassthrough()} | prompt_template | model
    return chain

# === BATCHED CLIP INFERENCE ===
def classify_clip_batch(df_batch, model, processor):
    results = []
    for i in tqdm.tqdm(range(0, len(df_batch), BATCH_SIZE)):
        batch = df_batch.iloc[i:i+BATCH_SIZE]
        images = []
        for row in batch.itertuples():
            try:
                if not str(row.mime_type).startswith(("image/jpeg", "image/png")):
                    images.append(None)
                    continue
                img_bytes = row.image['bytes']
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                images.append(img)
            except Exception:
                images.append(None)

        # Replace failed images with blank RGBs to avoid model crash
        safe_images = [img if img else Image.new("RGB", (224, 224)) for img in images]
        inputs = processor(images=safe_images, text=PROMPTS, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().tolist()

        for j, prob in enumerate(probs):
            if images[j] is None:
                results.append({"clip_label": None, "clip_confidence": 0.0})
            else:
                label = "inside" if prob[0] > prob[1] else "outside"
                results.append({"clip_label": label, "clip_confidence": max(prob)})
    return pd.DataFrame(results)

# === LLaMA INFERENCE ===
def classify_llama_batch(df_batch, chain):
    results = []
    for row in tqdm.tqdm(df_batch.itertuples()):
        caption = row.caption_attribution_description
        try:
            if not caption or not isinstance(caption, str):
                results.append({"llama_label": None, "llama_confidence": 0.0})
                continue

            result = chain.invoke({"caption": caption}).content.strip().lower()
            if "inside" in result and "outside" not in result:
                results.append({"llama_label": "inside", "llama_confidence": 1.0})
            elif "outside" in result and "inside" not in result:
                results.append({"llama_label": "outside", "llama_confidence": 1.0})
            elif "inside" in result and "outside" in result:
                results.append({"llama_label": None, "llama_confidence": 0.5})
            else:
                results.append({"llama_label": None, "llama_confidence": 0.0})
        except Exception:
            results.append({"llama_label": None, "llama_confidence": 0.0})
    return pd.DataFrame(results)

# === MAIN PROCESSING ===
def process_full_parquet():
    df = pd.read_parquet(INPUT_PARQUET)
    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")
    # total_rows
    for start in tqdm.tqdm(range(0, CHECKPOINT_INTERVAL+1, CHECKPOINT_INTERVAL)):
        end = min(start + CHECKPOINT_INTERVAL, total_rows)
        chunk = df.iloc[start:end].copy()
        print(f"\nProcessing rows {start} to {end}")

        # Run CLIP on chunk
        clip_model, clip_processor = load_clip()
        clip_results = classify_clip_batch(chunk, clip_model, clip_processor)
        del clip_model, clip_processor
        torch.cuda.empty_cache()

        chunk = pd.concat([chunk.reset_index(drop=True), clip_results.reset_index(drop=True)], axis=1)

        # Run LLaMA on chunk
        #llama_chain = load_llama_chain()
        #llama_results = classify_llama_batch(chunk, llama_chain)
        #chunk = pd.concat([chunk.reset_index(drop=True), llama_results.reset_index(drop=True)], axis=1)

        # Keep only needed columns
        keep_cols = ['image_url', 'caption_attribution_description', 'clip_label', 'clip_confidence']
        #, 'llama_label', 'llama_confidence']
        chunk = chunk[keep_cols]

        # Save checkpoint
        out_path = os.path.join(OUTPUT_DIR, f"labeled_{start}_{end}.parquet")
        chunk.to_parquet(out_path, index=False)
        print(f"Saved checkpoint: {out_path}")

if __name__ == "__main__":
    process_full_parquet()
