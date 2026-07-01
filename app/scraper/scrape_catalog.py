import json
import time
from pathlib import Path
 
import httpx
from bs4 import BeautifulSoup
 
BASE = "https://www.shl.com/products/product-catalog/"
PAGE_SIZE = 12
TYPE = 1  # Individual Test Solutions
MAX_START = 372  # discovered from pagination; adjust if SHL adds more entries
OUT_PATH = Path("data/raw/catalog_listing.json")
 
TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.shl.com/",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
}
 
 
def parse_listing_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    entries = []
 
    # The catalog table(s) on the page. We specifically want the table whose
    # header row contains "Individual Test Solutions".
    tables = soup.find_all("table")
    for table in tables:
        header_text = table.get_text(" ", strip=True)
        if "Individual Test Solutions" not in header_text:
            continue
 
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue  # header or malformed row
 
            name_cell, remote_cell, adaptive_cell, type_cell = cells[:4]
            link = name_cell.find("a")
            if not link:
                continue
 
            name = link.get_text(strip=True)
            url = link.get("href")
            if url and url.startswith("/"):
                url = "https://www.shl.com" + url
 
            remote_testing = bool(remote_cell.find(["span", "i", "svg"])) or "yes" in remote_cell.get_text(strip=True).lower()
            adaptive_irt = bool(adaptive_cell.find(["span", "i", "svg"])) or "yes" in adaptive_cell.get_text(strip=True).lower()
 
            type_letters = type_cell.get_text(" ", strip=True).split()
 
            entries.append(
                {
                    "name": name,
                    "url": url,
                    "remote_testing": remote_testing,
                    "adaptive_irt": adaptive_irt,
                    "test_type": type_letters,
                }
            )
    return entries
 
 
def scrape_all() -> list[dict]:
    all_entries = []
    seen_urls = set()
 
    with httpx.Client(headers=HEADERS, timeout=30, http2=True, follow_redirects=True) as client:
        start = 0
        while start <= MAX_START:
            params = {"start": start, "type": TYPE} if start > 0 else {"type": TYPE}
            print(f"Fetching start={start} ...")
            resp = client.get(BASE, params=params)
            resp.raise_for_status()
 
            entries = parse_listing_page(resp.text)
            if not entries:
                print(f"  No entries found at start={start}, stopping.")
                break
 
            new_count = 0
            for e in entries:
                if e["url"] not in seen_urls:
                    seen_urls.add(e["url"])
                    all_entries.append(e)
                    new_count += 1
 
            print(f"  Found {len(entries)} rows, {new_count} new.")
            start += PAGE_SIZE
            time.sleep(0.5)  # be polite
 
    return all_entries
 
 
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = scrape_all()
    with open(OUT_PATH, "w") as f:
        json.dump(
            {"legend": TEST_TYPE_LEGEND, "entries": entries},
            f,
            indent=2,
        )
    print(f"Saved {len(entries)} entries to {OUT_PATH}")
 
 
if __name__ == "__main__":
    main()