import asyncio
import re
import pandas as pd
import joblib
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import os
import hashlib

# Load the trained Random Forest model pipeline
try:
    MODEL = joblib.load('trained_models/webpurifier_XGBoost_SMOTE.pkl')
except FileNotFoundError:
    print("Error: webpurifier_model.pkl not found. Train and save your model first.")
    exit()

# Attributes from your proposal 
TARGET_TAGS = [
    'p', 'div', 'li', 'article', 'section', 
    'main', 'blockquote', 'span', 'code',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
    'td', 'th'
]
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
    't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip())

async def get_rendered_html(url):
    """Fetches the dynamically rendered HTML using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
        )
        page = await context.new_page()
        
        try:
            print(f"Navigating to {url}...")
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await asyncio.sleep(3) # Wait for initial JS
            
            # Scroll to trigger lazy-loaded elements
            for i in range(3):
                await page.evaluate("window.scrollBy(0, window.innerHeight)")
                await asyncio.sleep(1)
                
            return await page.content()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None
        finally:
            await browser.close()

def extract_features_and_predict(html_content):
    """Parses HTML, extracts features for each node, and predicts if it is content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    features_list = []
    nodes_list = []

    # 1. Extract Features
    for node in soup.find_all(TARGET_TAGS):
        raw_node_text = node.get_text(strip=True)
        normalized_node_text = normalize_text(raw_node_text)
        text_length = len(normalized_node_text)
        
        if text_length < 2: 
            continue 

        # Link Density
        links = node.find_all('a')
        link_text_len = sum(len(normalize_text(a.get_text())) for a in links)
        link_density = link_text_len / text_length if text_length > 0 else 0

        # Text-to-Tag Ratio
        total_tags = len(node.find_all(True)) + 1
        ttr = text_length / total_tags

        # Class/ID Keywords
        attrs_str = str(node.get('class', '')) + str(node.get('id', ''))
        attrs_str = attrs_str.lower()
        keyword_score = 0
        if any(kw in attrs_str for kw in ['article', 'content', 'body']): keyword_score += 1
        if any(kw in attrs_str for kw in ['sidebar', 'ad', 'menu', 'footer', 'nav']): keyword_score -= 1

        # Stop Word Density
        words = normalized_node_text.lower().split()
        sw_count = sum(1 for w in words if w in STOP_WORDS)
        sw_density = sw_count / len(words) if len(words) > 0 else 0

        features_list.append({
            'tag_type': node.name,
            'link_density': link_density,
            'text_to_tag_ratio': ttr,
            'keyword_score': keyword_score,
            'stop_word_density': sw_density,
            'text_length': text_length
        })
        nodes_list.append(normalized_node_text)

    if not features_list:
        return "No processable text found."

    # 2. Convert to DataFrame
    df_features = pd.DataFrame(features_list)

    # 3. Predict using the loaded model
    predictions = MODEL.predict(df_features)

    # 4. Reconstruct Purified Text
    # 4. Reconstruct Purified Text with Global Deduplication
    purified_text_blocks = []
    seen_blocks = set() # This set will remember every block we add
    
    for text_block, prediction in zip(nodes_list, predictions):
        if prediction == 1:
            # Check if this exact text has already been saved
            if text_block not in seen_blocks:
                purified_text_blocks.append(text_block)
                seen_blocks.add(text_block)

    return "\n\n".join(purified_text_blocks)

async def main():
    test_url = input("Enter a URL to purify: ")
    print("Purifying... This may take a few seconds.")
    
    html = await get_rendered_html(test_url)
    
    if html:
        clean_article = extract_features_and_predict(html)
        
        # 1. Create a directory for the outputs if it doesn't exist
        output_dir = "purified_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Generate a unique filename using the URL hash
        url_hash = hashlib.md5(test_url.encode('utf-8')).hexdigest()
        file_path = os.path.join(output_dir, f"{url_hash}.txt")
        
        # 3. Save the clean text to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_article)
            
        print("\n" + "="*50)
        print(" PURIFICATION COMPLETE ")
        print("="*50)
        print(f"Successlly saved purified content to:\n{file_path}\n")

if __name__ == "__main__":
    asyncio.run(main())