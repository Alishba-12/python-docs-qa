"""Scrape the Python 3 Tutorial and split it into one chunk per section."""
import json
import os
import re
import time
from urllib.parse import urldefrag, urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://docs.python.org/3/tutorial/"
OUTPUT_PATH = os.path.join("data", "sections.json")
HEADERS = {"User-Agent": "python-docs-qa/1.0 (internship task)"}
MIN_TEXT_LENGTH = 40
HEADING_TAGS = ("h1", "h2", "h3", "h4")

FALLBACK_PAGES = [
    "appetite.html", "interpreter.html", "introduction.html",
    "controlflow.html", "datastructures.html", "modules.html",
    "inputoutput.html", "errors.html", "classes.html", "stdlib.html",
    "stdlib2.html", "venv.html", "whatnow.html", "interactive.html",
    "floatingpoint.html", "appendix.html",
]


def fetch(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def get_page_urls():
    """Read the table of contents and return every chapter page once."""
    soup = BeautifulSoup(fetch(BASE_URL), "html.parser")
    urls = []
    for link in soup.select("div.toctree-wrapper a.reference.internal"):
        page_url, _ = urldefrag(urljoin(BASE_URL, link["href"]))
        if page_url not in urls:
            urls.append(page_url)
    if not urls:
        urls = [BASE_URL + name for name in FALLBACK_PAGES]
    return urls


def section_title(section):
    heading = section.find(list(HEADING_TAGS), recursive=False)
    if heading is None:
        return ""
    for link in heading.select("a.headerlink"):
        link.decompose()
    return heading.get_text(" ", strip=True)


def own_text(section):
    """Text that belongs directly to this section, not to its subsections."""
    parts = []
    for child in section.find_all(recursive=False):
        if child.name == "section" or child.name in HEADING_TAGS:
            continue
        for link in child.select("a.headerlink"):
            link.decompose()
        text = child.get_text()
        text = "\n".join(line.rstrip() for line in text.splitlines())
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def walk(section, page_url, parents, results):
    title = section_title(section)
    path = parents + [title]
    text = own_text(section)
    anchor = section.get("id", "")
    page_name = page_url.split("/")[-1]
    if len(text) >= MIN_TEXT_LENGTH:
        results.append({
            "id": f"{page_name}#{anchor}",
            "url": f"{page_url}#{anchor}",
            "page": page_name,
            "title": title,
            "path": " > ".join(path),
            "text": text,
        })
    for sub in section.find_all("section", recursive=False):
        walk(sub, page_url, path, results)


def scrape_page(url):
    soup = BeautifulSoup(fetch(url), "html.parser")
    body = soup.select_one("div.body")
    top = body.find("section") if body else None
    results = []
    if top is not None:
        walk(top, url, [], results)
    return results


def main():
    os.makedirs("data", exist_ok=True)
    all_sections = []
    for url in get_page_urls():
        sections = scrape_page(url)
        print(f"{url.split('/')[-1]}: {len(sections)} sections")
        all_sections.extend(sections)
        time.sleep(0.5)  # be polite to the docs server
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(all_sections)} sections saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
