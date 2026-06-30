#!/usr/bin/env python3
"""Render the results-audit site from manifest.py into docs/*.html.

Pure standard library. Run from anywhere:  python3 docs/build_pages.py
Re-run after editing manifest.py or regenerating assets.
"""
import html
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import manifest as M  # noqa: E402


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "main"


HEAD = git_head()
SHORT = HEAD[:8]
BLOB = M.REPO_URL + "/blob/main/"   # links track main so they survive file moves


def src(path, label=None):
    """A link to a source file in the GitHub repo."""
    if not path or path == "—":
        return html.escape(label or "—")
    label = label or path
    return '<a href="%s%s"><code>%s</code></a>' % (BLOB, path, html.escape(label))


def esc(s):
    return html.escape(str(s))


PAGES = [
    ("index.html", "Overview"),
    ("paper1.html", "Paper 1"),
    ("paper2.html", "Paper 2"),
    ("provenance.html", "Provenance"),
    ("environment.html", "Environment"),
    ("reproduce.html", "Reproduce"),
]


def nav(active):
    items = ['<span class="brand">UnderstandingVCS · results audit</span>']
    for href, label in PAGES:
        cls = ' class="active"' if href == active else ""
        items.append('<a href="%s"%s>%s</a>' % (href, cls, esc(label)))
    return '<nav><div class="wrap">%s</div></nav>' % "".join(items)


def page(active, title, body):
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title>
<meta name="description" content="Reproducibility and provenance audit for the UnderstandingVCS papers.">
<link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
%s
%s
<footer><div class="wrap">
Built from <a href="%s/commit/%s"><code>%s</code></a> ·
source of truth: <code>docs/manifest.py</code> ·
regenerate with <code>python3 docs/build_assets.py &amp;&amp; python3 docs/build_pages.py</code> ·
<a href="%s">repository</a>
</div></footer>
</body>
</html>
""" % (esc(title), nav(active), body, M.REPO_URL, HEAD, SHORT, M.REPO_URL)


def render_ledger(claims):
    rows = []
    for c in claims:
        st = c["status"]
        meta_rows = [
            ("Script", src(c["script"])),
            ("Command", '<code class="cmd">%s</code>' % esc(c["command"])),
            ("Artifact", src(c["artifact"])),
            ("Runtime", esc(c["runtime"])),
            ("Hardware", esc(c["hardware"])),
            ("Verified by", c["verified_by"]),  # may contain inline html/code refs
        ]
        meta = "".join("<dt>%s</dt><dd>%s</dd>" % (k, v) for k, v in meta_rows)
        rows.append("""
<div class="row">
  <div class="head">
    <span class="claim">%s</span>
    <span class="value">%s</span>
    <span class="badge %s">%s</span>
  </div>
  <div class="body">
    <p class="detail">%s</p>
    <dl class="meta">%s</dl>
  </div>
</div>""" % (esc(c["claim"]), esc(c["value"]), st, st, esc(c["detail"]), meta))
    return '<div class="ledger">%s</div>' % "".join(rows)


def render_figures(figs):
    cells = []
    for base, title, desc in figs:
        cells.append("""
<div class="fig">
  <a href="assets/img/%s.png" target="_blank"><img src="assets/img/%s.png" alt="%s"></a>
  <h4>%s</h4><p>%s</p>
</div>""" % (base, base, esc(title), esc(title), esc(desc)))
    return '<div class="figrid">%s</div>' % "".join(cells)


def render_videos(vids):
    cells = []
    for base, title, desc in vids:
        cells.append("""
<div class="vid">
  <video controls preload="metadata" playsinline>
    <source src="assets/video/%s.mp4" type="video/mp4">
  </video>
  <h4>%s</h4><p>%s</p>
</div>""" % (base, esc(title), esc(desc)))
    return '<div class="vidgrid">%s</div>' % "".join(cells)


# ---------------------------------------------------------------------------
def build_index():
    o = M.ORACLE
    pillars = "".join(
        '<div class="pillar"><b>%s</b><small>%s — %s</small></div>'
        % (esc(t), d, src(p)) for t, d, p, _ in o["pillars"])
    cards = ""
    for P in (M.PAPER1, M.PAPER2):
        cards += """
<a class="card" href="%s.html" style="text-decoration:none;color:inherit">
  <span class="venue">%s</span>
  <h3>%s</h3>
  <p>%s</p>
  <p style="color:var(--accent)">Open the evidence ledger →</p>
</a>""" % (P["id"], esc(P["venue"]), esc(P["title"]), esc(P["subtitle"]))

    body = """
<header class="hero"><div class="wrap">
  <h1>Audit every result, down to the script that made it</h1>
  <p class="lead">A reproducibility and provenance dashboard for the two papers built on the
  differentiable Atari&nbsp;2600 VCS. Every claim links to the exact code, command, artifact,
  runtime and hardware — and to the external reference that proves the emulator is real.</p>
  <div class="media">
    <img src="assets/gif/si_compare.gif" alt="Space Invaders: xitari vs jutari vs pixel difference">
    <p class="caption">Space Invaders — <b>xitari</b> (reference C++) · <b>jutari</b> (our Julia port) ·
    <b>pixel difference</b>. The difference panel is solid black: byte-for-byte identical output.</p>
  </div>
</div></header>

<section><div class="wrap">
  <h2>Why you can trust this</h2>
  <p class="sub">%s</p>
  <p>%s</p>
  <div class="pillars">%s</div>
</div></section>

<section><div class="wrap">
  <h2>The two papers</h2>
  <p class="sub">This site covers Paper&nbsp;1 and Paper&nbsp;2 only.</p>
  <div class="cards">%s</div>
</div></section>

<section><div class="wrap">
  <h2>How to read a claim</h2>
  <p class="sub">Each row in a paper’s ledger is one auditable result.</p>
  <table class="tbl">
    <tr><th>Field</th><th>What it gives the reviewer</th></tr>
    <tr><td>Value</td><td>the measured number, read from the committed result file</td></tr>
    <tr><td>Script</td><td>link to the exact source that produces it</td></tr>
    <tr><td>Command</td><td>the verbatim invocation to reproduce it</td></tr>
    <tr><td>Artifact</td><td>the committed output (results table / JSON / figure) it wrote</td></tr>
    <tr><td>Runtime &amp; Hardware</td><td>wall-clock and the machine it ran on</td></tr>
    <tr><td>Verified by</td><td>the test / oracle gate that guards it</td></tr>
    <tr><td>Status</td><td><span class="badge measured">measured</span> vs
      <span class="badge deferred">deferred</span> — we never dress a plan up as a result</td></tr>
  </table>
</div></section>
""" % (esc(o["headline"]), o["body"], pillars, cards)
    return page("index.html", "UnderstandingVCS — results audit", body)


def build_paper(P):
    extra = ""
    if P["id"] == "paper2":
        h = P["headline"]
        extra = """
<section><div class="wrap">
  <h2>Headline finding</h2>
  <p class="sub">Causal methods recover the truth; gradient/correlational methods collapse on the
  discrete sprite-position outputs whose naive gradient is exactly zero.</p>
  <div class="bignum">
    <div class="b"><strong>%s</strong><small>faithfulness gap (position regime)</small></div>
    <div class="b"><strong>%s</strong><small>causal / intervention mean (±%s CI95, n=%s)</small></div>
    <div class="b"><strong>%s</strong><small>gradient / correlational mean (±%s CI95, n=%s)</small></div>
    <div class="b"><strong>%s</strong><small>methods on the leaderboard</small></div>
    <div class="b"><strong>%s</strong><small>per-game records aggregated</small></div>
  </div>
  <p class="caption">From <a href="%stools/xai_study/compare/out/leaderboard.json"><code>leaderboard.json</code></a>
  · <code>headline_contrast.position_regime</code>.</p>
</div></section>""" % (h["gap"], h["causal"], h["causal_ci"], h["causal_n"],
                        h["grad"], h["grad_ci"], h["grad_n"], h["n_methods"], h["n_records"], BLOB)

    links = '<a href="%s%s">paper PDF</a>' % (BLOB, P["pdf"])
    if P.get("supplement_pdf"):
        links += ' · <a href="%s%s">supplement PDF</a>' % (BLOB, P["supplement_pdf"])

    vids = ""
    if P["videos"]:
        vids = """
<section><div class="wrap">
  <h2>Videos</h2>
  <p class="sub">The supplement divergence clip and the narrated overview.</p>
  %s
</div></section>""" % render_videos(P["videos"])

    gallery = ""
    if P.get("gallery"):
        cells = []
        for game, title in P["gallery"]:
            cells.append("""
<div class="vid">
  <video controls preload="none" playsinline poster="assets/img/poster_%s.jpg">
    <source src="assets/video/cmp_%s.mp4" type="video/mp4">
  </video>
  <h4>%s</h4>
</div>""" % (game, game, esc(title)))
        gallery = """
<section><div class="wrap">
  <h2>Conformance gallery</h2>
  <p class="sub">Each clip is three panels — <b>xitari</b> (reference C++) ·
  <b>jutari</b> (our port) · <b>per-pixel difference</b>. The difference panel stays black
  for the whole clip (only the “DIFFERENCE” header label is lit): byte-for-byte identical
  output. Verified across the full length of all 64 games; a representative selection follows.</p>
  <div class="vidgrid">%s</div>
  <p class="caption">All 64 games render pixel-exact — see the
  <a href="%stools/rom_sweep/results_jutari_screen.md">screen sweep</a>.</p>
</div></section>""" % ("".join(cells), BLOB)

    body = """
<header class="hero"><div class="wrap">
  <span class="venue" style="color:var(--accent-2);font-weight:600">%s</span>
  <h1>%s</h1>
  <p class="lead">%s</p>
  <p style="margin-top:14px">%s · %s</p>
</div></header>

%s

<section><div class="wrap">
  <h2>Evidence ledger</h2>
  <p class="sub">Every claim → script → command → artifact → runtime → hardware → verifying gate.</p>
  %s
</div></section>

<section><div class="wrap">
  <h2>Figures</h2>
  <p class="sub">Rasterised from the paper’s committed PDFs; click to enlarge.</p>
  %s
</div></section>

%s
%s
""" % (esc(P["venue"]), esc(P["title"]), P["blurb"], links, esc(P["subtitle"]),
       extra, render_ledger(P["claims"]), render_figures(P["figures"]), gallery, vids)
    return page(P["id"] + ".html", P["title"], body)


def build_provenance():
    PR = M.PROVENANCE
    def lst(items, three=False):
        out = []
        for it in items:
            if three:
                a, b, p = it
                out.append("<tr><td><b>%s</b></td><td>%s</td><td>%s</td></tr>"
                           % (esc(a), b, src(p)))
            else:
                a, b = it
                out.append("<tr><td><b>%s</b></td><td>%s</td></tr>" % (esc(a), b))
        return "".join(out)

    tests = "".join("<tr><td><b>%s</b></td><td>%s</td><td>%s</td></tr>"
                    % (esc(a), b, src(p)) for a, b, p in PR["tests"])
    repro = "".join("<p><b>%s</b></p><pre><code>%s</code></pre>" % (esc(a), esc(b))
                    for a, b in PR["reproduce"])

    body = """
<header class="hero"><div class="wrap">
  <h1>Provenance &amp; reproducibility</h1>
  <p class="lead">The infrastructure a skeptical reviewer can use to confirm the code and
  experiments are real: an external oracle, conformance harnesses, a large test suite,
  continuous integration, and an open development log.</p>
</div></header>

<section><div class="wrap">
  <h2>The external oracle</h2>
  <p>%s The reference emulator lives at %s with its determinism patch at %s.</p>
</div></section>

<section><div class="wrap">
  <h2>Conformance harnesses</h2>
  <table class="tbl"><tr><th>Harness</th><th>What it asserts</th><th>Source</th></tr>%s</table>
</div></section>

<section><div class="wrap">
  <h2>Comparison tooling</h2>
  <table class="tbl"><tr><th>Tool</th><th>Purpose</th><th>Source</th></tr>%s</table>
</div></section>

<section><div class="wrap">
  <h2>Test suite</h2>
  <table class="tbl"><tr><th>Port</th><th>Count</th><th>Location</th></tr>%s</table>
  <div class="note">%s</div>
</div></section>

<section><div class="wrap">
  <h2>Continuous integration</h2>
  <table class="tbl"><tr><th>Workflow</th><th>Runs</th><th>Source</th></tr>%s</table>
</div></section>

<section><div class="wrap">
  <h2>Development log</h2>
  <table class="tbl"><tr><th>Document</th><th>Role</th><th>Source</th></tr>%s</table>
</div></section>

<section><div class="wrap">
  <h2>Reproduce the conformance numbers</h2>
  %s
</div></section>
""" % (M.ORACLE["body"], src("xitari/"), src("tools/xitari_conformance_seed.patch"),
       lst(PR["harnesses"], three=True), lst(PR["tools"], three=True), tests,
       PR["tests_note"], lst(PR["ci"], three=True), lst(PR["logs"], three=True), repro)
    return page("provenance.html", "Provenance", body)


def build_environment():
    E = M.ENVIRONMENT
    hw = "".join("<tr><td><b>%s</b></td><td>%s</td><td>%s</td></tr>"
                 % (esc(a), esc(b), esc(c)) for a, b, c in E["hardware"])
    sw = "".join("<tr><td><b>%s</b></td><td>%s</td><td>%s</td></tr>"
                 % (esc(a), esc(b), esc(c)) for a, b, c in E["software"])
    det = "".join("<li>%s</li>" % d for d in E["determinism"])
    roms = "".join("<tr><td><code>%s</code></td><td>%s</td><td><code>%s</code></td></tr>"
                   % (esc(g), esc(s), esc(h)) for g, s, h in E["roms"]["rows"])

    body = """
<header class="hero"><div class="wrap">
  <h1>Environment &amp; hardware</h1>
  <p class="lead">Where every number was produced: machines, pinned software, seeds and
  byte-verified inputs.</p>
</div></header>

<section><div class="wrap">
  <h2>Hardware</h2>
  <table class="tbl"><tr><th>Role</th><th>Machine</th><th>Used for</th></tr>%s</table>
</div></section>

<section><div class="wrap">
  <h2>Software (pinned)</h2>
  <table class="tbl"><tr><th>Component</th><th>Version</th><th>Pin</th></tr>%s</table>
</div></section>

<section><div class="wrap">
  <h2>Determinism &amp; seeds</h2>
  <ul>%s</ul>
</div></section>

<section><div class="wrap">
  <h2>ROM provenance</h2>
  <p class="sub">%s</p>
  <table class="tbl"><tr><th>Game</th><th>Size</th><th>SHA-256 (prefix)</th></tr>%s</table>
  <p class="caption">Full table: %s · verify with
  <code>python3 tools/xai_study/repro/make_hash_tables.py --verify</code>.</p>
</div></section>
""" % (hw, sw, det, esc(E["roms"]["note"]), roms, src(E["roms"]["table"]))
    return page("environment.html", "Environment", body)


def build_reproduce():
    PR = M.PROVENANCE
    repro = "".join("<p><b>%s</b></p><pre><code>%s</code></pre>" % (esc(a), esc(b))
                    for a, b in PR["reproduce"])
    # gather every command from both papers
    rows = []
    for P in (M.PAPER1, M.PAPER2):
        for c in P["claims"]:
            if c["command"] and c["command"] != "—":
                rows.append("<tr><td>%s</td><td><code class=\"cmd\">%s</code></td><td>%s</td></tr>"
                            % (esc(c["claim"]), esc(c["command"]), src(c["artifact"])))
    table = "".join(rows)

    body = """
<header class="hero"><div class="wrap">
  <h1>Reproduce</h1>
  <p class="lead">Clone the repo, run the gates, then regenerate any single number from the
  command in its ledger row. ROMs come from AutoROM and are SHA-256 verified.</p>
</div></header>

<section><div class="wrap">
  <h2>Conformance gates</h2>
  %s
</div></section>

<section><div class="wrap">
  <h2>Per-claim commands</h2>
  <p class="sub">The exact invocation behind every ledger row, and the artifact it writes.</p>
  <table class="tbl"><tr><th>Claim</th><th>Command</th><th>Artifact</th></tr>%s</table>
</div></section>

<section><div class="wrap">
  <h2>Rebuild this site</h2>
  <pre><code>python3 docs/build_assets.py   # PDFs/MP4s -> web img/gif/mp4
python3 docs/build_pages.py    # manifest.py -> docs/*.html</code></pre>
</div></section>
""" % (repro, table)
    return page("reproduce.html", "Reproduce", body)


def main():
    outputs = {
        "index.html": build_index(),
        "paper1.html": build_paper(M.PAPER1),
        "paper2.html": build_paper(M.PAPER2),
        "provenance.html": build_provenance(),
        "environment.html": build_environment(),
        "reproduce.html": build_reproduce(),
    }
    for name, content in outputs.items():
        with open(os.path.join(HERE, name), "w") as f:
            f.write(content)
        print("wrote docs/%s (%d bytes)" % (name, len(content)))
    # .nojekyll so GitHub Pages serves our paths verbatim
    open(os.path.join(HERE, ".nojekyll"), "w").close()
    print("built at commit", SHORT)


if __name__ == "__main__":
    main()
