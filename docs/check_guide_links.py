#!/usr/bin/env python3
"""Check every external link in the method guides (docs/guides/*.py) and the primer.

  python3 docs/check_guide_links.py

Wikipedia: one batched API query per 40 titles (existence + not a disambiguation
page; redirects reported). DOIs: resolved through the doi.org handle API (publisher
sites often 403 a script). Everything else: plain GET. Exit code 1 if anything fails.
Wikipedia rate-limits fast clients (HTTP 429 = slow down, not "missing").
"""
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "UnderstandingVCS-linkcheck/1.0"}
URL_RE = re.compile(r'https?://(?:[^\s"\'<>()]|\([^\s"\'<>()]*\))+')


def get(url, tries=4):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20)
                continue
            return e.code, b""
        except Exception as e:  # noqa: BLE001
            return str(e)[:60], b""
    return 429, b""


def main():
    srcs = glob.glob(os.path.join(HERE, "guides", "*.py")) + [os.path.join(HERE, "primer.py"),
                                                               os.path.join(HERE, "build_pages.py")]
    urls = set()
    for f in srcs:
        urls |= {u.rstrip(".,;") for u in URL_RE.findall(open(f, encoding="utf-8").read())}
    wiki = sorted(u for u in urls if "en.wikipedia.org/wiki/" in u)
    doi = sorted(u for u in urls if "doi.org/" in u and "/api/" not in u)
    other = sorted(u for u in urls if u not in wiki and u not in doi and "github.com/akmaier" not in u
                   and "cdn.jsdelivr" not in u and "%s" not in u and "akmaier.github.io" not in u)
    bad = []
    titles = {urllib.parse.unquote(u.split("/wiki/", 1)[1]).replace("_", " "): u for u in wiki}
    names = sorted(titles)
    for i in range(0, len(names), 40):
        q = ("https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops"
             "&ppprop=disambiguation&redirects=1&titles=" + urllib.parse.quote("|".join(names[i:i + 40])))
        code, body = get(q)
        if code != 200:
            bad.append(("wikipedia-api", code, q)); continue
        res = json.loads(body)["query"]
        for r in res.get("redirects", []):
            print("redirect: %s -> %s" % (r["from"], r["to"]))
        norm = {n["to"]: n["from"] for n in res.get("normalized", [])}
        for p in res["pages"].values():
            t = norm.get(p["title"], p["title"])
            if "missing" in p:
                bad.append(("missing", t, titles.get(t, t)))
            elif "pageprops" in p:
                bad.append(("disambiguation", t, titles.get(t, t)))
    for u in doi:
        code, body = get("https://doi.org/api/handles/" + u.split("doi.org/", 1)[1])
        if code != 200 or json.loads(body or b"{}").get("responseCode") != 1:
            bad.append(("doi", code, u))
    for u in other:
        code, _ = get(u)
        if code != 200:
            bad.append(("http", code, u))
        time.sleep(1)
    print("checked %d wikipedia, %d doi, %d other links" % (len(wiki), len(doi), len(other)))
    for b in bad:
        print("BAD", *b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
