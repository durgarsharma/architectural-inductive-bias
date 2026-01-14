"""
Wikipedia Data Downloader - Hindi Version
Step 0, Part 1: Download Hindi Wikipedia Articles
"""

import requests
import time
import json
import os
from pathlib import Path

class WikipediaDownloaderHindi:
    """Downloads Hindi Wikipedia articles for the cipher project"""
    
    def __init__(self, output_dir="data/wikipedia_hindi"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Changed to Hindi Wikipedia API
        self.api_url = "https://hi.wikipedia.org/w/api.php"
        
        # User agent to avoid being blocked
        self.headers = {
            'User-Agent': 'CipherResearchBot/1.0 (Educational Research Project)'
        }
    
    def get_popular_articles(self, limit=1000):
        """
        Get list of popular Hindi Wikipedia articles using Special:Random
        """
        print(f"Fetching list of {limit} random Hindi articles...")
        
        articles = []
        batch_size = 50  # Fetch 50 at a time
        
        for batch_num in range((limit + batch_size - 1) // batch_size):
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": "0",
                "rnlimit": min(batch_size, limit - len(articles))
            }
            
            try:
                response = requests.get(
                    self.api_url, 
                    params=params, 
                    headers=self.headers,
                    timeout=10
                )
                
                # Check if response is successful
                if response.status_code != 200:
                    print(f"  Warning: HTTP {response.status_code}")
                    continue
                
                data = response.json()
                
                if "query" in data and "random" in data["query"]:
                    for article in data["query"]["random"]:
                        articles.append({
                            "id": article["id"],
                            "title": article["title"]
                        })
                    print(f"  Fetched {len(articles)}/{limit} article titles...")
                else:
                    print(f"  Unexpected response format")
                
                time.sleep(1)  # Be nice to Wikipedia servers
                
            except requests.exceptions.RequestException as e:
                print(f"  Network error: {e}")
                time.sleep(2)
            except json.JSONDecodeError as e:
                print(f"  JSON parsing error: {e}")
                time.sleep(2)
        
        print(f"Total articles found: {len(articles)}")
        return articles[:limit]
    
    def download_article_text(self, title):
        """
        Download the text content of a Hindi Wikipedia article
        """
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain"
        }
        
        try:
            response = requests.get(
                self.api_url, 
                params=params, 
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            if not pages:
                return None
            
            page_id = list(pages.keys())[0]
            
            if page_id == "-1":  # Article not found
                return None
            
            return pages[page_id].get("extract", None)
                
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def download_corpus(self, num_articles=1000, min_length=500):
        """
        Download a corpus of Hindi Wikipedia articles
        """
        print(f"\n=== Downloading Hindi Wikipedia Corpus ===")
        print(f"Target: {num_articles} articles")
        print(f"Minimum length: {min_length} characters\n")
        
        # Get article list (fetch 2x to account for short articles)
        article_list = self.get_popular_articles(num_articles * 2)
        
        if not article_list:
            print("❌ Failed to fetch article list. Check your internet connection.")
            return []
        
        corpus = []
        articles_metadata = []
        skipped = 0
        
        for idx, article_info in enumerate(article_list):
            if len(corpus) >= num_articles:
                break
            
            title = article_info["title"]
            print(f"[{len(corpus)+1}/{num_articles}] {title[:60]}...", end=" ")
            
            text = self.download_article_text(title)
            
            if text and len(text) >= min_length:
                corpus.append(text)
                articles_metadata.append({
                    "id": len(corpus),
                    "title": title,
                    "length": len(text),
                    "word_count": len(text.split())
                })
                
                # Save individual article
                filename = f"article_{len(corpus):04d}.txt"
                filepath = self.output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {title}\n\n{text}")
                
                print(f"✓ ({len(text):,} chars)")
            else:
                skipped += 1
                print(f"✗ (too short or failed)")
            
            time.sleep(1)  # Rate limiting
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(articles_metadata, f, indent=2, ensure_ascii=False)
        
        # Save combined corpus
        combined_path = self.output_dir / "combined_corpus.txt"
        with open(combined_path, 'w', encoding='utf-8') as f:
            separator = "\n\n" + "="*50 + " ARTICLE BREAK " + "="*50 + "\n\n"
            f.write(separator.join(corpus))
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"✓ Successfully downloaded: {len(corpus)} articles")
        print(f"✗ Skipped (too short/failed): {skipped}")
        print(f"✓ Saved to: {self.output_dir}")
        print(f"✓ Total characters: {sum(len(text) for text in corpus):,}")
        print(f"✓ Average article length: {sum(len(text) for text in corpus) // len(corpus):,} chars")
        print(f"{'='*60}\n")
        
        return corpus


def quick_download(num_articles=50):
    """Download a small sample for testing"""
    print("Starting QUICK download (50 Hindi articles)...\n")
    downloader = WikipediaDownloaderHindi(output_dir="data/wikipedia_hindi_sample")
    corpus = downloader.download_corpus(num_articles=num_articles)
    
    if corpus:
        print("✅ Success! Check the 'data/wikipedia_hindi_sample' folder.")
    else:
        print("❌ Download failed. Please check your internet connection.")


def full_download():
    """Download the full 1000 article corpus"""
    print("Starting FULL download (1000 Hindi articles)...\n")
    print("⚠️  This will take 1-2 hours. Make sure you have stable internet.\n")
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return
    
    downloader = WikipediaDownloaderHindi(output_dir="data/wikipedia_hindi")
    corpus = downloader.download_corpus(num_articles=1000)
    
    if corpus:
        print("✅ Full Hindi corpus downloaded successfully!")
    else:
        print("❌ Download failed.")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Hindi Wikipedia Corpus Downloader for Cipher Research")
    print("="*60 + "\n")
    print("Options:")
    print("  1. Quick test (50 articles, ~10-15 minutes)")
    print("  2. Full corpus (1000 articles, ~1-2 hours)")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        quick_download(50)
    elif choice == "2":
        full_download()
    else:
        print("\nInvalid choice. Running quick test by default...\n")
        quick_download(50)