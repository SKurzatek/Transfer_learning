import os
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
#import torch
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import tqdm
import requests

import aiohttp
import asyncio
import os
from tqdm import tqdm
from aiohttp import ClientSession
from urllib.parse import urlparse
import pandas as pd

PARQUET_DIR = "./FLAN_CLIP_checkpoints_3"

MAX_CONCURRENT_DOWNLOADS = 20
DATASET_DIR = "dataset_tmp"
INSIDE_DIR = os.path.join(DATASET_DIR, "inside")
OUTSIDE_DIR = os.path.join(DATASET_DIR, "outside")
os.makedirs(INSIDE_DIR, exist_ok=True)
os.makedirs(OUTSIDE_DIR, exist_ok=True)
download_success_counter = 0

CLIP_PRIMARY_CONF = 1.1
CLIP_CONF = 0.6
FLAN_CONF = 0.6

def datatset_preview(df):
    pd.set_option('display.max_colwidth', None)
    pd.set_option('expand_frame_repr', True)

    print("\n--- Sample Preview ---")
    for i in range(0, len(df)):
        row = df.iloc[i]
        if row['final_label'] == 'inside':
            print(f"\nRow {i}:")
            print(f"  URL:            {row['image_url']}")
            print(f"  FINAL label:     {row['final_label']}")
            print(f"  CLIP label:     {row['clip_label']} (confidence: {row['clip_confidence']:.4f})")
            print(f"  flan label:    {row['flan_label']} (confidence: {row['flan_confidence']:.4f})")
            print(f"  Description:    {row['caption_attribution_description']}")
            print(f"  Tags:    {row['clip_tags']}")
            print(f"  Tags scores:    {row['clip_tags_scores']}")

    
    print("\n--- Data Summary ---")
    print("Columns:", list(df.columns))
    print("Total rows:", len(df))

    label_counts = df["final_label"].value_counts(dropna=True)
    print("\n--- Label Distribution ---")
    for label in ["inside", "outside"]:
        count = label_counts.get(label, 0)
        print(f"  {label.capitalize()}: {count}")


def evaluate_image_selection(row):
    clip_label = row.get("clip_label")
    flan_label = row.get("flan_label")
    clip_conf = row.get("clip_confidence", 0.0)
    flan_conf = row.get("flan_confidence", 0.0)

    # Final Inside Label: Only CLIP decides
    if clip_label == "inside" and clip_conf >= CLIP_PRIMARY_CONF:
        return True, "inside"

    # Final Outside Label: Both must agree, both confident
    if (
        clip_label == "outside" and
        flan_label == "outside" and
        clip_conf >= CLIP_CONF and
        flan_conf >= FLAN_CONF
    ):
        return True, "outside"
    
    if (
        clip_label == "inside" and
        flan_label == "inside" and
        clip_conf >= CLIP_CONF and
        flan_conf >= FLAN_CONF
    ):
        return True, "inside"
    
    return False, None


def download_selection():
    dir_list = os.listdir(PARQUET_DIR)
    df_list = []

    for file in dir_list:
        if file.endswith(".parquet"):
            path = os.path.join(PARQUET_DIR, file)
            df_list.append(pd.read_parquet(path))

    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Apply unified selection + final label logic
    df[["selected", "final_label"]] = df.apply(
        evaluate_image_selection, axis=1, result_type="expand"
    )

    selected_df = df[df["selected"]].drop(columns=["selected"])
    print(f"Selected {len(selected_df)} images out of {len(df)} total.")
    return selected_df


async def download_image(session: ClientSession, url: str, label: str, sem: asyncio.Semaphore):
    global download_success_counter
    async with sem:
        try:
            filename = os.path.basename(urlparse(url).path)
            subdir = INSIDE_DIR if label == "inside" else OUTSIDE_DIR
            out_path = os.path.join(subdir, filename)

            # Deduplicate
            counter = 1
            base, ext = os.path.splitext(out_path)
            while os.path.exists(out_path):
                out_path = f"{base}_{counter}{ext}"
                counter += 1

            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    with open(out_path, "wb") as f:
                        f.write(content)
                    download_success_counter += 1
        except Exception as e:
            print(f"[ERROR] Failed to download {url}: {e}")

async def download_all_images_async(url_label_pairs):
    global download_success_counter
    download_success_counter = 0

    sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_DOWNLOADS)

    async with ClientSession(connector=connector) as session:
        tasks = [
            download_image(session, url, label, sem)
            for url, label in url_label_pairs
        ]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
            await f

    print(f"\n Download complete: {download_success_counter} of {len(url_label_pairs)} succeeded.")

def download_selected_images(df_selected):
    df_filtered = df_selected[["image_url", "final_label"]].dropna()
    url_label_pairs = list(df_filtered.itertuples(index=False, name=None))
    asyncio.run(download_all_images_async(url_label_pairs))

if __name__ == "__main__":
    selected_images = download_selection() 
    datatset_preview(selected_images)
    print()
    print("NUMBER OF IMAGES")
    print(len(selected_images))
    #download_selected_images(selected_images)