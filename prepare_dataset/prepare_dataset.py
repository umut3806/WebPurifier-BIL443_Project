"""
WebPurifier – Dataset Preparation (Single Script)
==================================================
Combines the two-step pipeline into one:

  Step 1 – Fetch Ground Truth:
      Uses trafilatura to download each URL and extract the "true" clean
      text (the main article body).  Saves one .txt per URL in clean_texts/.

  Step 2 – Build Feature Dataset:
      Uses Playwright to render the full page (JS included), walks every
      DOM node, extracts features, and labels each node as:
          1 = Content  (node text appears in the ground-truth file)
          0 = Noise    (ads, nav, footer, …)

Usage:
    python prepare_dataset.py                          # use default URL file
    python prepare_dataset.py --url-file urls/cnn.txt  # custom URL file
    python prepare_dataset.py --skip-fetch              # skip Step 1 if clean_texts already exist
"""

import argparse
import asyncio
import hashlib
import os
import random
import re
import sys
import time
from collections import Counter

import pandas as pd
import trafilatura
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


# ====================================================================
# CONSTANTS
# ====================================================================

TARGET_TAGS = [
    "p", "div", "li", "article", "section",
    "main", "blockquote", "span", "code",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "td", "th",
]

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "don't",
    "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y",
    "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't",
    "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven",
    "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
    "mustn't", "needn", "needn't", "shan", "shan't", "shouldn",
    "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't",
    "wouldn", "wouldn't",
}

DEFAULT_URL_FILE      = "urls/combined/medium.txt"
CLEAN_TEXTS_FOLDER    = "clean_texts"
OUTPUT_CSV            = "webpurifier_dataset.csv"
MAX_CONCURRENT_PAGES  = 3      # parallel browser tabs (safe for most sites)


# ====================================================================
# HELPERS
# ====================================================================

def normalize_text(text: str) -> str:
    """Collapse whitespace for reliable substring matching."""
    return re.sub(r"\s+", " ", text.strip())


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def load_urls(path: str) -> list[str]:
    """Read a file of URLs (one per line) and return a list."""
    with open(path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


# ====================================================================
# STEP 1 – FETCH GROUND-TRUTH CLEAN TEXTS (trafilatura)
# ====================================================================

def fetch_clean_texts(urls: list[str], output_folder: str) -> dict[str, bool]:
    """
    Downloads each URL with trafilatura, extracts the main article text,
    and saves it to  output_folder/<md5_hash>.txt.

    Returns a dict  {url: success_bool}  for reporting.
    """
    os.makedirs(output_folder, exist_ok=True)
    report: dict[str, bool] = {}

    total = len(urls)
    for idx, url in enumerate(urls, 1):
        h = url_hash(url)
        out_path = os.path.join(output_folder, f"{h}.txt")

        # Skip if already fetched
        if os.path.exists(out_path):
            print(f"  [{idx}/{total}] SKIP (exists)  {h}  ← {url}")
            report[url] = True
            continue

        print(f"  [{idx}/{total}] Fetching       {h}  ← {url}")

        try:
            html = trafilatura.fetch_url(url)
            if html is None:
                print(f"           ✗ Could not download page.")
                report[url] = False
                continue

            clean = trafilatura.extract(html)
            if not clean:
                print(f"           ✗ Downloaded but no text extracted.")
                report[url] = False
                continue

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(clean)

            print(f"           ✓ Saved ({len(clean)} chars)")
            report[url] = True

        except Exception as e:
            print(f"           ✗ Error: {e}")
            report[url] = False

    return report


# ====================================================================
# STEP 2 – BUILD FEATURE DATASET (Playwright + BeautifulSoup)
#
# Performance notes:
#   - ONE browser is launched and shared across all URLs.
#   - Up to MAX_CONCURRENT_PAGES tabs run in parallel (semaphore).
#   - Random jitter (0.5-2s) between tab starts avoids rate-limiting.
#   - Scrolls reduced from 5 to 3; sleeps shortened.
# ====================================================================

async def _render_page(context, url: str) -> str | None:
    """Open a new tab, render the page, return HTML, then close the tab."""
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await asyncio.sleep(2)

        # Scroll to trigger lazy-loaded elements (3 scrolls instead of 5)
        for _ in range(3):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(0.5)

        return await page.content()

    except Exception as e:
        print(f"           ✗ Playwright error: {e}")
        return None
    finally:
        await page.close()


def extract_node_features(html: str, ground_truth: str, h: str) -> list[dict]:
    """
    Walk every target node in `html`, compute features, and label it as
    content (1) or noise (0) based on whether the node's normalised text
    appears in the `ground_truth` string.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    for node in soup.find_all(TARGET_TAGS):
        raw = node.get_text(strip=True)
        text = normalize_text(raw)
        text_length = len(text)

        if text_length < 2:
            continue

        # --- Feature extraction ---

        # 1. Link Density
        links = node.find_all("a")
        link_text_len = sum(len(normalize_text(a.get_text())) for a in links)
        link_density = link_text_len / text_length if text_length > 0 else 0

        # 2. Text-to-Tag Ratio
        total_tags = len(node.find_all(True)) + 1
        ttr = text_length / total_tags

        # 3. Keyword Score (class/id attributes)
        attrs_str = (str(node.get("class", "")) + str(node.get("id", ""))).lower()
        keyword_score = 0
        if any(kw in attrs_str for kw in ["article", "content", "body"]):
            keyword_score += 1
        if any(kw in attrs_str for kw in ["sidebar", "ad", "menu", "footer", "nav"]):
            keyword_score -= 1

        # 4. Stop-Word Density
        words = text.lower().split()
        sw_count = sum(1 for w in words if w in STOP_WORDS)
        sw_density = sw_count / len(words) if words else 0

        # --- Label ---
        label = 1 if (text_length > 20 and text in ground_truth) else 0

        rows.append({
            "url_hash":          h,
            "tag_type":          node.name,
            "link_density":      round(link_density, 4),
            "text_to_tag_ratio": round(ttr, 4),
            "keyword_score":     keyword_score,
            "stop_word_density": round(sw_density, 4),
            "text_length":       text_length,
            "label":             label,
        })

    return rows


async def _process_single_url(
    idx: int,
    total: int,
    url: str,
    clean_folder: str,
    context,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Process one URL under the semaphore: render → extract → return rows."""
    h = url_hash(url)
    gt_path = os.path.join(clean_folder, f"{h}.txt")

    if not os.path.exists(gt_path):
        print(f"  [{idx}/{total}] SKIP (no ground truth)  {h}")
        return []

    async with semaphore:
        # Random jitter so requests don't fire in lockstep
        await asyncio.sleep(random.uniform(0.5, 2.0))

        print(f"  [{idx}/{total}] Rendering  {h}  ← {url}")

        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = normalize_text(f.read())

        html = await _render_page(context, url)
        if html is None:
            return []

        rows = extract_node_features(html, ground_truth, h)
        content_n = sum(1 for r in rows if r["label"] == 1)
        print(f"  [{idx}/{total}] ✓ {len(rows)} nodes  "
              f"(content: {content_n}, noise: {len(rows) - content_n})")
        return rows


async def build_dataset(
    urls: list[str],
    clean_folder: str,
    max_concurrent: int = MAX_CONCURRENT_PAGES,
) -> pd.DataFrame:
    """
    Process all URLs concurrently (up to `max_concurrent` at a time)
    using a single shared browser instance.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(urls)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        tasks = [
            _process_single_url(idx, total, url, clean_folder, context, semaphore)
            for idx, url in enumerate(urls, 1)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        await context.close()
        await browser.close()

    # Flatten results, skipping exceptions
    all_rows: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ⚠ Task error: {r}")
        elif isinstance(r, list):
            all_rows.extend(r)

    return pd.DataFrame(all_rows)


# ====================================================================
# MAIN
# ====================================================================

async def run(args: argparse.Namespace):
    urls = load_urls(args.url_file)
    print(f"\n{'=' * 65}")
    print(f"  WebPurifier – Dataset Preparation")
    print(f"  URL file : {args.url_file}  ({len(urls)} URLs)")
    print(f"  Output   : {args.output}")
    print(f"  Workers  : {args.workers} concurrent tabs")
    print(f"{'=' * 65}\n")

    # ---- Step 1: Fetch clean texts ----
    if args.skip_fetch:
        print("▶ STEP 1  Skipped (--skip-fetch flag)\n")
    else:
        print("▶ STEP 1  Fetching ground-truth clean texts with trafilatura…\n")
        t0 = time.time()
        report = fetch_clean_texts(urls, CLEAN_TEXTS_FOLDER)
        ok = sum(1 for v in report.values() if v)
        print(f"\n  Done in {time.time() - t0:.1f}s  —  {ok}/{len(urls)} URLs succeeded.\n")

    # ---- Step 2: Build feature dataset ----
    print("▶ STEP 2  Rendering pages & building feature dataset…\n")
    t0 = time.time()
    df = await build_dataset(urls, CLEAN_TEXTS_FOLDER, max_concurrent=args.workers)

    if df.empty:
        print("\n✗ No data extracted. Check that clean_texts/ has ground truth files.")
        sys.exit(1)

    df.to_csv(args.output, index=False)
    elapsed = time.time() - t0

    # ---- Summary ----
    print(f"\n{'=' * 65}")
    print(f"  DATASET READY")
    print(f"{'=' * 65}")
    print(f"  Rows          : {len(df)}")
    print(f"  Unique URLs   : {df['url_hash'].nunique()}")
    print(f"  Content (1)   : {(df['label'] == 1).sum()}  "
          f"({(df['label'] == 1).mean() * 100:.1f}%)")
    print(f"  Noise   (0)   : {(df['label'] == 0).sum()}  "
          f"({(df['label'] == 0).mean() * 100:.1f}%)")
    print(f"  Saved to      : {args.output}")
    print(f"  Step 2 time   : {elapsed:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(
        description="WebPurifier – Prepare the training dataset from a list of URLs."
    )
    parser.add_argument(
        "--url-file",
        default=DEFAULT_URL_FILE,
        help=f"Path to a text file with one URL per line (default: {DEFAULT_URL_FILE})",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip Step 1 (trafilatura). Use when clean_texts/ already has all ground truth files.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_CONCURRENT_PAGES,
        help=f"Max concurrent browser tabs for Step 2 (default: {MAX_CONCURRENT_PAGES})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.url_file):
        print(f"Error: URL file not found: {args.url_file}")
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
