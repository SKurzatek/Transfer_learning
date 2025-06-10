import os
import pandas as pd
from PIL import Image
from io import BytesIO
import torch
import tqdm
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

CHECKPOINT_INTERVAL = 100
BATCH_SIZE = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSIDE_PROMPTS = [
    "living room interior with a gray sofa, wooden coffee table, and large windows",
    "modern office space with desks, swivel chairs, computers, and overhead lights",
    "cozy bedroom interior with a bed, nightstand, lamp, and patterned rug",
    "hallway inside a building with artificial lighting and tiled floor",
    "kitchen interior with countertops, cabinets, and indoor lighting"
]
OUTSIDE_PROMPTS = [
    "outdoor park with grass, trees, and a walking path",
    "mountain landscape with sky, rocks, and vegetation",
    "streetscape outside showing buildings, sidewalk, and open sky",
    "beach scene with sand, sea, and horizon in the distance",
    "forest environment with trees, underbrush, and natural light"
]

INPUT_PARQUET = "train-00001-of-00330.parquet"
OUTPUT_DIR = "FLAN_CLIP_checkpoints_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_clip():
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    return model, processor

def classify_clip_batch(df_batch, model, processor):
    results = []
    texts = INSIDE_PROMPTS + OUTSIDE_PROMPTS

    for i in tqdm.tqdm(range(0, len(df_batch), BATCH_SIZE)):
        batch = df_batch.iloc[i:i + BATCH_SIZE]
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

        # Replace invalid images with blank filler
        safe_images = [img if img else Image.new("RGB", (224, 224)) for img in images]

        inputs = processor(images=safe_images, text=texts, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1).cpu()

        for j in range(len(images)):
            if images[j] is None:
                results.append({"clip_label": None, "clip_confidence": 0.0})
            else:
                p = probs[j]
                inside_score = p[: len(INSIDE_PROMPTS)].mean().item()
                outside_score = p[len(INSIDE_PROMPTS):].mean().item()
                if inside_score > outside_score:
                    label, conf = "inside", inside_score
                else:
                    label, conf = "outside", outside_score
                results.append({"clip_label": label, "clip_confidence": conf})

    return pd.DataFrame(results)

# === FLAN-T5 SETUP ===
def load_flan():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    return model, tokenizer

def classify_flan_batch(df_batch, model, tokenizer):
    results = []
    prompts = []

    for caption in df_batch["caption_attribution_description"]:
        if not isinstance(caption, str) or not caption.strip():
            prompts.append(None)
        else:
            prompt = (
                f"Determine whether the following caption describes a photo taken inside, outside or none. "
                f"Examples for 'inside': room, building, interior, people inside. " 
                f"Examples for 'outside': landscape, street, open area, animals in the field. "
                f"ONLY ANSWER with 'inside', 'outside' or 'none'.\n\nCaption: {caption}"
            )
            prompts.append(prompt)

    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    batch_prompts = [prompts[i] for i in valid_indices]

    if not batch_prompts:
        return pd.DataFrame([{"flan_label": None, "flan_confidence": 0.0} for _ in prompts])

    encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=False
        )
        decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        scores = output.scores[0]

    top_token_ids = output.sequences[:, 1]

    for i in range(len(prompts)):
        if i not in valid_indices:
            results.append({"flan_label": None, "flan_confidence": 0.0})
        else:
            idx = valid_indices.index(i)
            label = decoded[idx].strip().lower()

            token_logits = scores[idx]
            token_probs = torch.softmax(token_logits, dim=0)
            confidence = token_probs[top_token_ids[idx]].item()

            if "inside" in label and "outside" not in label:
                results.append({"flan_label": "inside", "flan_confidence": confidence})
            elif "outside" in label and "inside" not in label:
                results.append({"flan_label": "outside", "flan_confidence": confidence})
            elif "inside" in label and "outside" in label:
                results.append({"flan_label": None, "flan_confidence": 0.5})
            else:
                results.append({"flan_label": None, "flan_confidence": 0.0})

    return pd.DataFrame(results)

# === MAIN LOOP ===
def process_full_parquet():
    df = pd.read_parquet(INPUT_PARQUET)
    total_rows = len(df)
    print(f"Total rows to process: {total_rows}")

    for start in tqdm.tqdm(range(0, total_rows, CHECKPOINT_INTERVAL)):
        end = min(start + CHECKPOINT_INTERVAL, total_rows)
        chunk = df.iloc[start:end].copy()
        print(f"\nProcessing rows {start} to {end}")

        # === Run FLAN-T5 ===
        flan_model, flan_tokenizer = load_flan()
        flan_results = classify_flan_batch(chunk, flan_model, flan_tokenizer)
        del flan_model, flan_tokenizer
        torch.cuda.empty_cache()

        chunk = pd.concat([chunk.reset_index(drop=True), flan_results.reset_index(drop=True)], axis=1)

        # === Run CLIP ===
        clip_model, clip_processor = load_clip()
        clip_results = classify_clip_batch(chunk, clip_model, clip_processor)
        del clip_model, clip_processor
        torch.cuda.empty_cache()

        chunk = pd.concat([chunk.reset_index(drop=True), clip_results.reset_index(drop=True)], axis=1)

        # === Save only needed columns ===
        keep_cols = [
            'image_url', 'caption_attribution_description',
            'clip_label', 'clip_confidence',
            'flan_label', 'flan_confidence'
        ]
        chunk = chunk[keep_cols]

        # === Save checkpoint ===
        out_path = os.path.join(OUTPUT_DIR, f"labeled_{start}_{end}.parquet")
        chunk.to_parquet(out_path, index=False)
        print(f"Saved checkpoint: {out_path}")

if __name__ == "__main__":
    process_full_parquet()
