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
BATCH_SIZE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prompts for inside/outside classification
INSIDE_PROMPTS = [
    "A photo of furniture in an interior room",
    "A photo of chairs in an interior room",
    "A photo of tables in an interior room",
    "A photo of a bookshelf in an interior room", 
    "A photo of a bookshelf in a library",       
    "A photo of paintings in a museum hall",
    "A photo of art in a museum hall",
    "A photo of an exhibition in a museum hall", 
    "A photo of walls in a room",
    "A photo of a ceiling in a room",            
    "A photo of a floor in a room",      
    "A photo of desks and chairs in an office",
    "A photo of arches in a church interior",
    "A photo of a platform in a subway station",
    "A photo of tables and a board in a classroom",
    "A photo of a university interior with tables and a board", 
    "A photo of clothes in a wardrobe",
    "A photo of couches in a living room",
    "A photo of artificial lighting in a hallway",
    "A photo of tables in a restaurant interior",
    "A photo of appliances in a kitchen",
    "A photo of seats and gates in an airport terminal",
    "A photo of tiles and a sink in a bathroom",
    "A photo of people in an audience",
    "A photo of people talking", 
    "A photo of a presentation on a stage",
    "A photo of shops and people in a shopping mall interior",
    "A photo of workout equipment in a gym inside a building"
]

OUTSIDE_PROMPTS = [
    "A photo of trees",
    "A photo of a forest",
    "A photo of a street",
    "A photo of a building from the ouside",
    "A photo of a church from the ouside",
    "A photo of a university from the ouside",
    "A photo of an office from the ouside",
    "A photo of sky",
    "A photo of rocks and mountain landscape",
    "A photo of buildings and cars on a city street",
    "A photo of grass and people in a park",
    "A photo of sand and ocean on a beach",
    "A photo of houses and roads in a village",
    "A photo of cars and a sidewalk on a street",
    "A photo of a field and goalposts in a sports ground",
    "A photo of people and buildings in a public square",
    "A photo of trees on both sides of a road",
    "A photo of crops and sky on farmland", 
    "A photo of a river and bridge in an outdoor scene",
    "A photo of a train and tracks in an open area",
    "A photo of a bus stop and road outside",
    "A photo of animals and grass in a field"
]

GARBAGE_PROMPTS = [
    "A photo of text and arrows in a diagram",
    "A photo of printed text in a document",
    "A photo of text and layout in a scanned book page",
    "A photo of bars and labels in a digital chart",
    "A photo of symbols and a white background in a logo",
    "A photo of geographic outlines on a map",
    "A photo of solid shapes in an icon",
    "A photo of lines and a sketch in a drawing",
    "A photo of a browser window in a screenshot",
    "A photo of rows and columns in a data table",
    "A photo of abstract brushstrokes in a painting",
    "A photo of an animated figure in a cartoon",
    "A photo of typed letters on a printed page", 
    "A photo of interface elements in a software screenshot",
    "A photo of handwritten notes in a diagram",
    "A photo of a map",     
    "A photo of an icon",   
    "A photo of a logo",    
    "A photo of a computer screen",
    "A photo of an empty scene",
    "A photo of a blank photo",
    "A photo of nothing"
]
n_inside = len(INSIDE_PROMPTS)
n_outside = len(OUTSIDE_PROMPTS)
n_garbage = len(GARBAGE_PROMPTS)
# Scene tags for enrichment
SCENE_TAGS = [
    "walls", "ceiling", "windows", "floor",
    "living space", "furniture",
    "office", "tables", "chairs", "computers", "screens",
    "hallway", "artificial lighting",
    "paintings", "art exhibition",
    "tiles", "bricks",
    "modern infrastructure", "machinery",
    "subway station", "shopping mall", "library", "restaurant", "museum",


    "landscape", "grass", "trees", "path", "pavement",
    "mountain", "sky", "rocks", "vegetation", "bushes",
    "vehicle", "car", "street", "buildings", "sidewalk", "city street",
    "forest", "beach", "sand", "water", "river", "lake", "horizon",
    "kitchen", "park", "garden"
]

INPUT_PARQUET = "train-00002-of-00330.parquet"
OUTPUT_DIR = "FLAN_CLIP_checkpoints_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_clip():
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    return model, processor

def classify_clip_batch(df_batch, model, processor, top_k_tags=3):
    results = []
    io_texts = INSIDE_PROMPTS + OUTSIDE_PROMPTS + GARBAGE_PROMPTS

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

        safe_images = [img if img else Image.new("RGB", (224, 224)) for img in images]

        inputs_io = processor(images=safe_images, text=io_texts, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs_io = model(**inputs_io)
            probs_io = outputs_io.logits_per_image.softmax(dim=1).cpu()
            probs_io.shape[1] == n_inside + n_outside + n_garbage

        '''
        # Scene tag scoring
        inputs_tags = processor(images=safe_images, text=SCENE_TAGS, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs_tags = model(**inputs_tags)
            probs_tags = outputs_tags.logits_per_image.softmax(dim=1).cpu()
        '''

        for j in range(len(images)):
            if images[j] is None:
                results.append({
                    "clip_label": None,
                    "clip_confidence": 0.0,
                    "clip_tags": []
                })
            else:
                p_io = probs_io[j]
                inside_score = p_io[:n_inside].sum().item()
                outside_score = p_io[n_inside:(n_inside + n_outside)].sum().item()
                garbage_score = p_io[n_inside + n_outside:].sum().item()

                if (inside_score > outside_score) and (inside_score > garbage_score):
                    label = "inside"
                    confidence = inside_score
                elif (outside_score > inside_score) and (outside_score > garbage_score):
                    label = "outside"
                    confidence = outside_score
                else:
                    label = "garbage"
                    confidence = garbage_score
                #confidence = inside_score if label == "inside" else outside_score

                top_indices = torch.topk(p_io, k=top_k_tags).indices.tolist()
                tags = [io_texts[idx] for idx in top_indices]
                tags_scores = [p_io[idx].item() for idx in top_indices]
                
                '''
                p_tags = probs_tags[j]
                top_indices = torch.topk(p_tags, k=top_k_tags).indices.tolist()
                tags = [SCENE_TAGS[idx] for idx in top_indices]
                '''
                
                results.append({
                    "clip_label": label,
                    "clip_confidence": confidence,
                    "clip_tags": tags,
                    "clip_tags_scores": tags_scores
                })

    return pd.DataFrame(results)

# === FLAN-T5 SETUP ===
def load_flan():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    return model, tokenizer

# === FLAN BATCH CLASSIFICATION ===
def classify_flan_batch(df_batch, model, tokenizer):

    prefix = (
        "You are a classifier that labels Wikipedia article images as 'inside', 'outside', or 'garbage'.\n"
        "- inside: enclosed spaces (walls, ceilings, furniture).\n"
        "- outside: open-air scenes (sky, vegetation, streets).\n"
        "- garbage: non-photographic images (diagrams, icons, scans).\n"
        "Use both the caption and the top CLIP tags with their confidences to decide. Answer with exactly one label.\n\n"
    )

    examples = (
        "Example 1:\n"
        "Caption: A photo of couches in a living room\n"
        "Tags: 'A photo of couches in a living room' (0.80), 'A photo of bookshelves in a library' (0.75), 'A photo of walls in a room' (0.60)\n"
        "Answer: inside\n\n"
        "Example 2:\n"
        "Caption: A photo of trees and a river under a blue sky\n"
        "Tags: 'A photo of trees and a trail in a forest' (0.85), 'A photo of sky and rocks in a mountain landscape' (0.70), 'A photo of grass and people in a park' (0.65)\n"
        "Answer: outside\n\n"
        "Example 3:\n"
        "Caption: A photo of text and arrows in a diagram\n"
        "Tags: 'A photo of text and arrows in a diagram' (0.90), 'A photo of bars and labels in a digital chart' (0.88), 'A photo of symbols and a white background in a logo' (0.70)\n"
        "Answer: garbage\n\n"
        "Now classify:\n"
    )

    prompts, valid_idx = [], []
    for i, row in df_batch.iterrows():
        caption = row.get('caption_attribution_description', '')
        tags = row.get('clip_tags', [])
        scores = row.get('clip_tags_scores', [])
        
        if not caption or not isinstance(caption, str):
            prompts.append(None)
            continue

        if len(tags) > 0:
            tag_str = ", ".join([f"'{t}' ({s:.2f})" for t, s in zip(tags, scores)])
        else:
            tag_str = "CLIP model couldnt provide any information."
        prompt = (
            prefix + examples +
            f"Caption: {caption}\n"
            f"Tags: {tag_str}\n"
            "Answer:"
        )
        prompts.append(prompt)
        valid_idx.append(i)

    if not valid_idx:
        return pd.DataFrame([{'flan_label': None, 'flan_confidence': 0.0} for _ in prompts])

    batch_prompts = [prompts[i] for i in valid_idx]
    enc = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=3,
            return_dict_in_generate=True,
            output_scores=True
        )
    decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    token_scores = output.scores[0]
    top_tokens = output.sequences[:, 1]

    results = []
    idx_map = {idx: pos for pos, idx in enumerate(valid_idx)}
    for i in range(len(prompts)):
        if i not in idx_map:
            results.append({'flan_label': None, 'flan_confidence': 0.0})
            continue
        pos = idx_map[i]
        label = decoded[pos].strip().lower()
        probs = torch.softmax(token_scores[pos], dim=0)
        conf = probs[top_tokens[pos]].item()
        if label in ('inside', 'outside', 'garbage'):
            results.append({'flan_label': label, 'flan_confidence': conf})
        else:
            results.append({'flan_label': None, 'flan_confidence': conf})

    return pd.DataFrame(results)

# === MAIN LOOP ===
def process_full_parquet():
    df = pd.read_parquet(INPUT_PARQUET)
    for start in range(0, len(df), CHECKPOINT_INTERVAL):
        end = min(start + CHECKPOINT_INTERVAL, len(df))
        chunk = df.iloc[start:end].copy()

        clip_model, clip_proc = load_clip()
        clip_df = classify_clip_batch(chunk, clip_model, clip_proc)
        del clip_model, clip_proc; torch.cuda.empty_cache()
        chunk = pd.concat([chunk.reset_index(drop=True), clip_df.reset_index(drop=True)], axis=1)

        flan_model, flan_tok = load_flan()
        flan_df = classify_flan_batch(chunk, flan_model, flan_tok)
        del flan_model, flan_tok; torch.cuda.empty_cache()
        chunk = pd.concat([chunk, flan_df.reset_index(drop=True)], axis=1)

        out_df = chunk[[
            'image_url', 'caption_attribution_description',
            'clip_label', 'clip_confidence', 'clip_tags', 'clip_tags_scores',
            'flan_label', 'flan_confidence'
        ]]
        out_path = os.path.join(OUTPUT_DIR, f"labeled_{start}_{end}.parquet")
        out_df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    process_full_parquet()


'''

INSIDE_PROMPTS = [
    "A photo of an inside of a building",
    "A photo of walls and ceiling",
    "A photo of windows and floor",
    "A photo of an Living space inside of a building with furniture.",
    "A photo of an Inside of an office building with tables, chairs, computers, and screens.",
    "A photo of an Hallway inside a building with artificial lighting and floor.",
    "A photo of an Museum interior with paintings on the walls or art exhibition.",
    "A photo of an Inside of a castle or church with tiles, bricks, and antique architecture.",
    "A photo of an Inside of a modern infrastructure building.",
    "A photo of an Inside of a factory or industrial facility."
]

OUTSIDE_PROMPTS = [
    "A photo of an Outside landscape with grass, trees, and a walking path.",
    "A photo of an Mountain landscape with sky, rocks, and vegetation.",
    "A photo of an Streetscape outside showing buildings, sidewalk, and open sky.",
    "A photo of an Beach scene with sand, sea, and horizon in the distance.",
    "A photo of an Forest environment with trees, underbrush, and natural light."
]

'''