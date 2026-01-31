from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup


def _extract_next_data_json(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")  # on Mac, html.parser is usually fine
    tag = soup.find("script", id="__NEXT_DATA__")

    if tag is not None:
        raw = tag.get_text(strip=True)  # IMPORTANT: not tag.string
        if raw:
            return json.loads(raw)

    # Fallback: regex extraction (works even if the HTML parser chokes)
    m = re.search(
        r'<script[^>]+id="__NEXT_DATA__"[^>]*>\s*(\{.*?\})\s*</script>',
        html,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError("Could not locate __NEXT_DATA__ JSON in the provided HTML.")
    return json.loads(m.group(1))


def extract_country_flag_table(page_source_html: str) -> List[Tuple[str, str]]:
    data = _extract_next_data_json(page_source_html)

    try:
        content = data["props"]["pageProps"]["pageData"]["content"]
        block = next(b for b in content if isinstance(b, dict) and "countryListProps" in b)
        countries = block["countryListProps"]
    except Exception as e:
        raise KeyError(f"Could not locate countryListProps in __NEXT_DATA__: {e}") from e

    rows: List[Tuple[str, str]] = []
    for c in countries:
        name = (c.get("name") or "").strip()
        flag_src = ((c.get("flag") or {}).get("src") or "").strip()
        if name and flag_src:
            rows.append((name, flag_src))

    # de-dupe preserving order
    seen = set()
    out: List[Tuple[str, str]] = []
    for name, src in rows:
        if (name, src) not in seen:
            seen.add((name, src))
            out.append((name, src))
    return out


if __name__ == "__main__":
    # Recommended: read the HTML from a file you saved from "View Page Source"
    # e.g. save as associations.html and point at it here.
    with open("generate/names_and_flags.html", "r", encoding="utf-8") as f:
        html = f.read()

    out = open("fifa_members_with_flag_sources.csv", "w")

    table = extract_country_flag_table(html)

    print("country,flag_src")
    out.write("country,flag_source\n")
    for country, flag_src in table:
        c = country.replace('"', '""')
        u = flag_src.replace('"', '""')
        print(f'"{c}","{u}"')
        out.write(f"\"{c}\",{u}\n")
    out.close()
