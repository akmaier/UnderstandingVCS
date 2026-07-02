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


def srcln(path, line, label=None):
    """A GitHub blob link to a specific line of a source file (verified line no)."""
    label = label or ("%s:%d" % (path.rsplit("/", 1)[-1], line))
    return '<a href="%s%s#L%d"><code>%s</code></a>' % (BLOB, path, line, esc(label))


def link(href, label):
    """A link that is external (http), an internal page/anchor, or a repo path."""
    if href.startswith("http") or href.endswith(".html") or href.startswith("#") \
            or ".html#" in href:
        return '<a href="%s">%s</a>' % (href, label)
    return src(href, label)


PAGES = [
    ("index.html", "Overview"),
    ("paper1.html", "Paper 1"),
    ("conformance.html", "Conformance"),
    ("paper2.html", "Paper 2"),
    ("methods.html", "P2 Methods"),
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
        ]
        # optional: the upstream scripts that produce this row's input data
        if c.get("inputs"):
            items = "".join(
                "<li>%s — %s</li>" % (src(p), desc) for p, desc in c["inputs"])
            meta_rows.append(("Input data", '<ul class="inputs">%s</ul>' % items))
        meta_rows += [
            ("Runtime", esc(c["runtime"])),
            ("Hardware", esc(c["hardware"])),
            ("Verified by", c["verified_by"]),  # may contain inline html/code refs
        ]
        meta = "".join("<dt>%s</dt><dd>%s</dd>" % (k, v) for k, v in meta_rows)
        # optional: a short "where this comes from" comment under the meta
        note = '<p class="rownote">%s</p>' % c["note"] if c.get("note") else ""
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
    %s
  </div>
</div>""" % (esc(c["claim"]), esc(c["value"]), st, st, c["detail"], meta, note))
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
        % (esc(t), d, link(href, lbl)) for t, d, lbl, href in o["pillars"])
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
  <p class="sub">Across all regimes, causal/intervention methods stay well above gradient and
  correlational methods — a robust faithfulness gap whose confidence interval excludes zero.
  On the discrete sprite-position outputs the naive gradient is exactly zero; the emulator's
  bilinear sampler restores a non-zero position gradient, but its faithfulness stays low, so the
  position-only gap is directional and not significant at six games.</p>
  <div class="bignum">
    <div class="b"><strong>%s</strong><small>all-regime faithfulness gap · CI [%s, %s] (excludes 0)</small></div>
    <div class="b"><strong>%s</strong><small>causal / intervention mean (all regimes, n=%s)</small></div>
    <div class="b"><strong>%s</strong><small>gradient / correlational mean (all regimes, n=%s)</small></div>
    <div class="b"><strong>%s</strong><small>methods on the leaderboard</small></div>
    <div class="b"><strong>%s</strong><small>per-game records aggregated</small></div>
  </div>
  <p class="caption">Primary contrast from
  <a href="%stools/xai_study/compare/out/leaderboard.json"><code>leaderboard.json</code></a>
  · <code>headline_contrast.all_regimes</code> (CI from <code>leaderboard_ci.csv</code>).</p>
  <p class="sub" style="margin-top:18px">Position-only regime — shown as directional, <b>not</b>
  significant at six games:</p>
  <div class="bignum">
    <div class="b"><strong>%s</strong><small>position-regime gap · CI [%s, %s] (includes 0)</small></div>
    <div class="b"><strong>%s</strong><small>causal / intervention mean (position, n=%s)</small></div>
    <div class="b"><strong>%s</strong><small>gradient / correlational mean (position, n=%s)</small></div>
  </div>
  <p class="caption"><code>headline_contrast.position_regime</code>. The testbed was
  redesigned to score every method on shared random-action gameplay states — see the
  <a href="%sxai_paper/xai_2_interpretability/experiment_redesign.md">experiment redesign note</a>.</p>
</div></section>""" % (h["gap"], h["gap_ci_lo"], h["gap_ci_hi"],
                        h["causal"], h["causal_n"], h["grad"], h["grad_n"],
                        h["n_methods"], h["n_records"], BLOB,
                        h["pos_gap"], h["pos_gap_ci_lo"], h["pos_gap_ci_hi"],
                        h["pos_causal"], h["pos_causal_n"], h["pos_grad"], h["pos_grad_n"], BLOB)

    links = '<a href="%s%s">paper PDF</a>' % (BLOB, P["pdf"])
    if P.get("supplement_pdf"):
        links += ' · <a href="%s%s">supplement PDF</a>' % (BLOB, P["supplement_pdf"])
    tour = {"paper1": ('conformance.html', "Conformance code tour"),
            "paper2": ('methods.html', "Method catalogue &amp; execution stack")}.get(P["id"])
    if tour:
        links += ' · <a href="%s"><b>%s →</b></a>' % tour

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
  <p class="sub">For a line-by-line walk of each harness's call stack, see the
  <a href="conformance.html">conformance code tour</a>.</p>
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


def _stack(rows):
    """rows: list of (step_html, computes_html) -> a 2-col call-stack table."""
    body = "".join("<tr><td>%s</td><td>%s</td></tr>" % (s, c) for s, c in rows)
    return ('<table class="tbl"><tr><th>Step in the call stack</th>'
            '<th>What is actually computed</th></tr>%s</table>' % body)


def build_conformance():
    # Every link below points at a line number verified against the source.
    T_PXC1 = "jaxtari/tests/test_pxc1_conformance.py"
    CHK = "tools/check_trace.py"
    CHKJL = "tools/check_trace.jl"
    ENV = "jaxtari/jaxtari/env/stella_environment.py"
    CON = "jaxtari/jaxtari/console.py"
    CPU = "jaxtari/jaxtari/cpu/m6502.py"
    T_PXC2 = "jaxtari/tests/test_pxc2_jaxtari_vs_jutari.py"
    T_SCR = "jaxtari/tests/test_screen_conformance.py"
    T_K = "jaxtari/tests/test_pxc4_klaus_dormann.py"
    TD = "tools/trace_dump.cpp"
    JDUMP = "tools/jutari_trace_dump.jl"

    intro = """
<header class="hero"><div class="wrap">
  <h1>Conformance harness — a guided code tour</h1>
  <p class="lead">A reviewer should be able to confirm what every conformance test computes
  without reading the whole codebase. This page walks the function stack of each harness —
  from the test assertion, through the replay driver, down into the emulator core where the
  compared bytes are produced — with a line-anchored link at every step. The documentation in
  the code explains <i>how</i> each function works; this page explains <i>what is being
  checked and why it is evidence</i>.</p>
  <p style="margin-top:12px"><a href="#oracle">The oracle pipeline</a> ·
  <a href="#pxc1">PXC1 (RAM/CPU)</a> · <a href="#pxc2">PXC2 (dual-port)</a> ·
  <a href="#pxcs">PXC-S (screen)</a> · <a href="#pxc4">PXC4 (6502)</a> ·
  <a href="#core">the emulator core</a></p>
</div></header>"""

    oracle = """
<section id="oracle"><div class="wrap">
  <h2>How conformance is established</h2>
  <p>Every test compares one of our ports against <b>xitari</b>, the external
  C&plus;&plus; reference emulator (<a href="https://github.com/google-deepmind/xitari">google-deepmind/xitari</a>).
  The comparison runs in two stages. First, xitari is driven by a fixed action stream and its
  per-frame state is dumped to a JSONL trace. Then a port is driven by the <i>same</i> action
  stream and its state is diffed against that trace, frame by frame.</p>
  <pre><code>  fixed action stream
        │
        ▼
  %s  ──▶  JSONL trace (per frame: ram, optional cpu / screen)
   (drives the real xitari C++ emulator: ale.act, ale.getRAM, ale.getScreen)
        │                                         committed under tools/fixtures/
        ▼
  %s  ──▶  steps a port with the same actions, diffs each frame
        │
        ▼
  PXC1 (RAM+CPU)   PXC2 (port≡port)   PXC-S (screen)   PXC4 (6502 ISA)</code></pre>
  <p>The JSONL record for one frame holds the absolute frame index, the action applied, the
  128 B of RIOT RAM as 256 hex chars, and — when the relevant flag is set — the six CPU
  registers and the 210&times;160 framebuffer. The RAM bytes are read out of xitari at
  %s and hex-encoded at %s; the CPU registers are tapped through a
  <code>friend class CpuDebug</code> at %s so xitari's headers are never modified.</p>
</div></section>""" % (
        srcln(TD, 132, "trace_dump.cpp · main"),
        srcln(CHK, 123, "check_trace.py · check_trace"),
        srcln(TD, 256, "trace_dump.cpp:256 · ale.getRAM()"),
        srcln(TD, 67, "trace_dump.cpp:67 · hex_encode"),
        srcln(TD, 54, "trace_dump.cpp:54 · CpuDebug"),
    )

    pxc1 = """
<section id="pxc1"><div class="wrap">
  <h2>PXC1 — RAM &amp; CPU trace replay</h2>
  <p class="sub">Does a port reproduce xitari's RAM (and CPU registers) frame-for-frame?</p>
  <p>The test loads an xitari trace, replays the same actions through the port, and asserts the
  port's 128 B RIOT RAM hex-matches the trace at every frame. A mismatch raises at the first
  diverging frame with a byte-level diff.</p>
  %s
  <p><b>How to read it.</b> The assertion that matters is the hex-string compare at
  %s: the port's RAM is read at %s, hex-encoded, and compared to the trace's
  <code>ram</code> field. If they differ the harness reports the frame and the differing byte
  addresses, then raises. The Julia port is checked by the mirror harness %s.</p>
</div></section>""" % (
        _stack([
            (srcln(T_PXC1, 61, "test_pxc1_conformance.py:61 · test_jaxtari_matches_xitari…"),
             "Entry point. Builds the port from the ROM, replays the trace, asserts all 10 "
             "frames matched (%s)." % srcln(T_PXC1, 80, "L80 · assert matched == 10")),
            (srcln(CHK, 123, "check_trace.py:123 · check_trace(rom, trace)"),
             "Constructs the environment, boots it the same way xitari boots (60 NOOP + 4 RESET), "
             "then loops the trace's actions."),
            (srcln(CHK, 175, "check_trace.py:175 · ram_hex != ref['ram']"),
             "<b>The core check.</b> Per frame: encode the port RAM, compare to the reference hex; "
             "on mismatch emit a byte diff and raise <code>ConformanceError</code>."),
            (srcln(ENV, 375, "stella_environment.py:375 · step(action)"),
             "Applies the action and runs one frame to the next VSYNC — the port advance."),
            (srcln(ENV, 492, "stella_environment.py:492 · get_ram()"),
             "Returns the 128 B RIOT RAM (<code>console.bus.ram</code>) — the exact bytes compared."),
        ]),
        srcln(CHK, 175, "check_trace.py:175"),
        srcln(ENV, 492, "get_ram()"),
        srcln(CHKJL, 61, "check_trace.jl:61 · check_trace"),
    )

    pxc2 = """
<section id="pxc2"><div class="wrap">
  <h2>PXC2 — dual-port cross-check</h2>
  <p class="sub">Do the JAX port and the Julia port diverge from xitari <i>identically</i>?</p>
  <p>This is the claim featured on the front page. Two independently written ports — one in
  JAX, one in Julia — are required to produce <b>byte-identical RIOT RAM at every frame</b>.
  Where they still differ from xitari, they must differ in the <i>same</i> bytes by the
  <i>same</i> amount. Two separate implementations agreeing to the bit is strong evidence of a
  shared, correct mechanism rather than two coincidental bugs.</p>
  %s
  <p><b>How to read it.</b> The headline test runs jaxtari <i>live</i> and loads the jutari
  result from a committed fixture trace (produced by %s). The compare at
  %s is a plain byte-equality of the two ports' RAM per frame; on any mismatch it
  fails at %s with the frame index and differing addresses. A third test
  (%s) locks in the residual divergence-from-xitari count per ROM, so a real
  emulation fix must move <i>both</i> ports in the same commit — they can never drift apart
  silently.</p>
</div></section>""" % (
        _stack([
            (srcln(T_PXC2, 281, "test_pxc2…py:281 · test_jaxtari_matches_jutari_per_frame_ram"),
             "The headline cross-check. Steps jaxtari live; loads the jutari fixture; compares "
             "RAM per frame."),
            (srcln(T_PXC2, 240, "test_pxc2…py:240 · _jaxtari_ram_per_frame"),
             "Boots jaxtari (60 NOOP + 4 RESET) and records the 128 B RAM after every action — "
             "the live side of the comparison."),
            (srcln(T_PXC2, 299, "test_pxc2…py:299 · if jax_ram != jul_ram"),
             "<b>The core check.</b> Byte-equality of the two ports' RAM at frame <code>i</code>; "
             "on mismatch builds a per-address diff and fails (%s)."
             % srcln(T_PXC2, 306, "L306")),
            (srcln(T_PXC2, 310, "test_pxc2…py:310 · …divergence_pattern_unchanged"),
             "Regression guard: the count of bytes still differing from xitari must equal the "
             "value pinned per ROM (all 0 today)."),
            (srcln(JDUMP, 132, "jutari_trace_dump.jl:132 · main"),
             "Produces the jutari fixture: same boot, same actions, records RAM per frame — the "
             "Julia side, generated once and committed."),
        ]),
        srcln(JDUMP, 132, "jutari_trace_dump.jl"),
        srcln(T_PXC2, 299, "L299"),
        srcln(T_PXC2, 306, "L306"),
        srcln(T_PXC2, 310, "the third test"),
    )

    pxcs = """
<section id="pxcs"><div class="wrap">
  <h2>PXC-S — screen conformance</h2>
  <p class="sub">Does the rendered 210&times;160 framebuffer match, pixel for pixel?</p>
  <p>RAM equality does not imply screen equality — the TIA renders pixels from register writes
  whose timing must be exact. PXC-S diffs the framebuffer per frame, both xitari↔jutari (from
  fixtures) and xitari↔jaxtari (rendered live), counting differing pixels.</p>
  %s
  <p><b>How to read it.</b> %s counts differing pixels per frame; each test asserts
  the worst frame is within a pinned threshold (%s — set to 0 once a game is
  pixel-exact). The live framebuffer comes from %s, which crops the TIA's internal
  buffer to the visible window.</p>
</div></section>""" % (
        _stack([
            (srcln(T_SCR, 219, "test_screen_conformance.py:219 · test_jutari_screen_matches_xitari"),
             "xitari fixture vs jutari fixture, per frame."),
            (srcln(T_SCR, 235, "test_screen_conformance.py:235 · test_jaxtari_screen_matches_xitari"),
             "xitari fixture vs jaxtari rendered live."),
            (srcln(T_SCR, 213, "test_screen_conformance.py:213 · _per_frame_diffs"),
             "<b>The core check.</b> Per frame: <code>(a[i] != b[i]).sum()</code> — the count of "
             "differing palette-index pixels; the test asserts the worst is ≤ the pin (%s)."
             % srcln(T_SCR, 227, "L227")),
            (srcln(ENV, 473, "stella_environment.py:473 · get_screen()"),
             "Crops <code>console.bus.tia.framebuffer</code> to the visible rows — the rendered "
             "output being diffed."),
        ]),
        srcln(T_SCR, 213, "_per_frame_diffs"),
        srcln(T_SCR, 75, "max_screen_diff"),
        srcln(ENV, 473, "get_screen()"),
    )

    pxc4 = """
<section id="pxc4"><div class="wrap">
  <h2>PXC4 — 6502 functional test</h2>
  <p class="sub">Is the CPU core a correct 6502, independent of any game?</p>
  <p>This runs Klaus Dormann's widely used 6502 functional test — an external ROM that
  exercises every opcode, addressing mode and flag — on a flat 64 KB memory, bypassing the TIA,
  RIOT and cartridge entirely. Success is a hard, unambiguous signal: the program only reaches
  its success address if every sub-test passed.</p>
  %s
  <p><b>How to read it.</b> The loop steps the CPU at %s and watches the program
  counter: reaching <code>KLAUS_SUCCESS_PC = 0x3469</code> (%s) means every sub-test
  passed; a PC stuck on a trap loop means a specific opcode is wrong, and the failure names the
  address to disassemble.</p>
</div></section>""" % (
        _stack([
            (srcln(T_K, 80, "test_pxc4_klaus_dormann.py · test_klaus_dormann…passes"),
             "Loads the ROM into flat 64 KB memory, sets PC to 0x0400, steps until success or stuck."),
            (srcln(T_K, 99, "test_pxc4_klaus_dormann.py:99 · _step_inner(state, memory)"),
             "Executes one 6502 instruction with no TIA/RIOT side effects — pure ISA."),
            (srcln(T_K, 102, "test_pxc4_klaus_dormann.py:102 · cur_pc == KLAUS_SUCCESS_PC"),
             "<b>The pass condition.</b> PC reaches 0x3469 (%s) ⇒ all opcodes/flags correct."
             % srcln(T_K, 49, "L49")),
            (srcln(CPU, 410, "m6502.py:410 · _step_inner"),
             "The shared instruction decoder/executor used by both the game path and this test."),
        ]),
        srcln(T_K, 99, "L99"),
        srcln(T_K, 49, "test_pxc4_klaus_dormann.py:49"),
    )

    core = """
<section id="core"><div class="wrap">
  <h2>Where the compared bytes come from — the emulator core</h2>
  <p>Each harness above bottoms out in the same port internals. This is the stack a single
  <code>step()</code> runs through, so a reviewer can follow a compared byte back to the CPU
  instruction and TIA tick that produced it.</p>
  %s
  <p>The Julia port mirrors this structure (<code>env_step!</code> / <code>get_ram</code> /
  <code>run_until_frame</code>); its fixtures are produced by %s. Because the two ports
  share neither code nor language, PXC2 agreement is independent corroboration of this stack.</p>
</div></section>""" % (
        _stack([
            (srcln(ENV, 87, "stella_environment.py:87 · reset(...)") + " → "
             + srcln(ENV, 191, "_boot_burn"),
             "Resets the console and burns the boot frames (60 NOOP + 4 RESET + any per-game "
             "starting actions) so the port starts where xitari starts."),
            (srcln(ENV, 375, "stella_environment.py:375 · step(action)"),
             "Applies the action to the bus, then runs one video frame."),
            (srcln(CON, 97, "console.py:97 · run_until_frame"),
             "Steps the CPU (up to a 25,000-instruction budget) until the TIA frame counter "
             "advances — one frame, including the partial/grey-frame case."),
            (srcln(CPU, 365, "m6502.py:365 · step") + " → "
             + srcln(CPU, 410, "_step_inner") + " + " + srcln(CPU, 382, "_tia_post_step"),
             "Fetch/decode/execute one 6502 instruction, then advance the TIA and RIOT by the "
             "cycles it consumed and resolve any WSYNC stall."),
            (srcln(CON, 49, "console.py:49 · console_reset"),
             "Reads the reset vector at $FFFC/$FFFD into PC — the boot entry the whole run hangs off."),
        ]),
        srcln(JDUMP, 132, "jutari_trace_dump.jl"),
    )

    reproduce = """
<section><div class="wrap">
  <h2>Reproduce</h2>
  <pre><code># regenerate an xitari reference trace
./tools/trace_dump --rom xitari/roms/pong.bin \\
  --actions tools/fixtures/actions/pong_noop_10.txt &gt; /tmp/pong.jsonl

# replay it against each port
python tools/check_trace.py --rom xitari/roms/pong.bin --trace /tmp/pong.jsonl
julia --project=jutari tools/check_trace.jl --rom xitari/roms/pong.bin --trace /tmp/pong.jsonl

# run the gates (PXC1/PXC2/PXC-S/PXC4 live)
cd jaxtari &amp;&amp; .venv/bin/python -m pytest -q tests/test_pxc1_conformance.py \\
  tests/test_pxc2_jaxtari_vs_jutari.py tests/test_pxc4_klaus_dormann.py
jaxtari/.venv/bin/pytest jaxtari/tests/test_screen_conformance.py   # ~23 min</code></pre>
</div></section>"""

    body = (intro + oracle + pxc1 + pxc2 + pxcs + pxc4 + core + reproduce)
    return page("conformance.html", "Conformance harness — code tour", body)


def build_methods():
    XS = "tools/xai_study/"

    def catalogue(rows, recdir, reccount):
        # rows: (method_html, script_basename, score_html)
        def key_of(s):
            return s.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        def audit(s):
            af = audit_faith(key_of(s))
            if not af:
                return "<span class='cite'>excluded</span>"
            faith, ci, ng, nr = af
            return ("<b>%.3f</b> <span class='cite'>±%.3f · %s games</span>" % (faith, ci, ng))
        body = "".join(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (m, srcln_or_src(XS + s), sc, audit(s),
               '<a href="m_%s.html">explain &amp; figure →</a>' % key_of(s)) for m, s, sc in rows)
        head = ('<table class="tbl"><tr><th>Method (reference)</th>'
                '<th>Implementation</th><th>Score description</th>'
                '<th><a href="#e6">Audit faithfulness</a></th>'
                '<th>Page</th></tr>'
                '%s</table>' % body)
        rec = ('<p class="caption">Records: %s — <b>%d</b> committed §R JSON+npz.</p>'
               % (src(XS + recdir, recdir), reccount))
        return head + rec

    def srcln_or_src(path):  # link a file by its basename label
        return src(path, path.rsplit("/", 1)[-1])

    PHASE_A = [
        ("<b>A1</b> connectomics / data-flow graph", "phaseA_kording/A1_connectomics.jl",
         "precision/recall + graph-edit-distance vs the true read/write graph"),
        ("<b>A2</b> single-unit lesions", "phaseA_kording/A2_lesions.jl",
         "rank-correlation of importance with the unit's true role; spurious-specific count"),
        ("<b>A3</b> tuning curves", "phaseA_kording/A3_tuning.jl",
         "spurious-tuning rate (strongly-tuned units whose tuning ≠ true role)"),
        ("<b>A4</b> spike-word / pairwise correlations", "phaseA_kording/A4_correlations.jl",
         "weak-pairwise / strong-global structure reproduced vs true coupling"),
        ("<b>A5</b> local field potentials", "phaseA_kording/A5_lfp.jl",
         "%-variance that is the known clocks (frame/scanline) → epiphenomenal"),
        ("<b>A6</b> Granger causality", "phaseA_kording/A6_granger.jl",
         "false-edge / missed-edge rate vs the true data-flow"),
        ("<b>A7</b> dim-reduction (NMF/PCA)", "phaseA_kording/A7_dimred.jl",
         "matched-component fraction vs known signals (clock, R/W, vsync)"),
        ("<b>A8</b> whole-state recording", "phaseA_kording/A8_wholestate.jl",
         "descriptive baseline"),
    ]
    PHASE_B = [
        ("Vanilla gradient <span class='cite'>Simonyan et al. 2014</span>",
         "phaseB_attribution/saliency.jl",
         "corr + deletion/insertion AUC + precision@k vs true causal top-k"),
        ("Grad×Input / DeepLIFT <span class='cite'>Shrikumar et al. 2017</span>",
         "phaseB_attribution/gradxinput.jl", "as above + completeness where defined"),
        ("Guided Backprop <span class='cite'>Springenberg et al. 2015</span>",
         "phaseB_attribution/guided_backprop.jl", "as above + the Adebayo et al. 2018 sanity check"),
        ("SmoothGrad <span class='cite'>Smilkov et al. 2017</span>",
         "phaseB_attribution/smoothgrad.jl", "noise-averaged saliency; corr + del/ins"),
        ("<b>Integrated Gradients</b> <span class='cite'>Sundararajan et al. 2017</span>",
         "phaseB_attribution/ig_baseline_sweep.jl", "corr + del/ins + completeness; baseline sweep"),
        ("Expected Gradients <span class='cite'>Erion et al. 2021 (NMI)</span>",
         "phaseB_attribution/expected_gradients.jl", "baseline-averaged IG; as above"),
        ("Occlusion <span class='cite'>Zeiler &amp; Fergus 2014</span>",
         "phaseB_attribution/occlusion.jl", "del/ins AUC ≈ coarse intervention oracle"),
        ("Extremal / meaningful perturbation <span class='cite'>Fong &amp; Vedaldi 2017; Fong et al. 2019</span>",
         "phaseB_attribution/perturbation.jl", "learned minimal mask; IoU vs true causal set"),
        ("RISE <span class='cite'>Petsiuk et al. 2018</span>",
         "phaseB_attribution/rise.jl", "randomized-mask saliency (N=500 masks); corr + del/ins"),
        ("LIME <span class='cite'>Ribeiro et al. 2016</span>",
         "phaseB_attribution/lime.jl", "local linear weights; corr vs true; stability"),
        ("KernelSHAP / Shapley <span class='cite'>Lundberg &amp; Lee 2017; Štrumbelj &amp; Kononenko 2014</span>",
         "phaseB_attribution/kernelshap.jl", "Shapley values; corr vs true; convergence vs compute"),
        ("On-distribution counterfactual <span class='cite'>cf. Olson 2021; Atrey 2020</span>",
         "phaseB_attribution/counterfactual.jl", "minimal valid edit: validity + minimality vs true minimal set"),
        ("<b>N/A audit</b>: Grad-CAM/++, attention rollout, VIPER "
         "<span class='cite'>Selvaraju 2017; Abnar &amp; Zuidema 2020; Bastani 2018</span>",
         "phaseB_attribution/na_audit.jl", "recorded as <i>does not apply</i> (needs NN layers / a policy)"),
    ]
    PHASE_C = [
        ("Activation patching / causal mediation <span class='cite'>Vig et al. 2020; ROME, Meng et al. 2022</span>",
         "phaseC_mechanistic/activation_patching.jl", "recovered effect vs the exact patch; site P/R vs true data-flow"),
        ("Interchange interventions / DAS <span class='cite'>Geiger et al. 2021, 2023</span>",
         "phaseC_mechanistic/das.jl", "interchange accuracy; alignment vs the true variable"),
        ("Attribution / edge patching <span class='cite'>Nanda 2023; Syed et al. 2023</span>",
         "phaseC_mechanistic/attribution_patching.jl", "approx error vs true patching; edge P/R"),
        ("Path patching / IOI circuit <span class='cite'>Wang et al. 2022; Goldowsky-Dill et al. 2023</span>",
         "phaseC_mechanistic/path_patching.jl", "circuit precision/recall vs the true routine"),
        ("ACDC — automatic circuit discovery <span class='cite'>Conmy et al. 2023</span>",
         "phaseC_mechanistic/acdc.jl", "edge P/R + scrubbing-preserved performance vs true data-flow"),
        ("Sparse autoencoders <span class='cite'>Cunningham et al. 2023; Bricken et al. 2023; Templeton et al. 2024</span>",
         "phaseC_mechanistic/sae.jl", "feature↔known-variable match (F1/MI) + causal use + monosemanticity"),
        ("NMF/PCA dictionaries", "phaseC_mechanistic/dictionaries.jl",
         "matched-component fraction vs known variables"),
        ("Causal scrubbing <span class='cite'>Chan et al. 2022</span>",
         "phaseC_mechanistic/causal_scrubbing.jl", "scrubbing-preserved performance vs the true routine"),
        ("Linear probing + control tasks <span class='cite'>Alain &amp; Bengio 2017; Hewitt &amp; Liang 2019</span>",
         "phaseC_mechanistic/linear_probing.jl", "accuracy <b>and</b> selectivity (probe − control) → present-vs-used gap"),
        ("Logit / tuned lens <span class='cite'>nostalgebraist 2020; Belrose et al. 2023</span>",
         "phaseC_mechanistic/logit_lens.jl", "readout fidelity vs the true intermediate value"),
    ]
    ORACLE = [
        ("Intervention oracle — the primary ground truth", "ground_truth/oracle_intervene.jl",
         "occlude/clamp/resample a cause <code>u</code>, re-run bit-exact, record exact |Δy(u)|"),
        ("Gradient oracle (content path)", "ground_truth/oracle_grad.jl",
         "∂y/∂u + Integrated Gradients through the differentiable substrate (content outputs only)"),
        ("Cross-check", "ground_truth/oracle_xcheck.jl",
         "correlation of intervention vs gradient; disagreement is reported, not hidden"),
    ]
    T3 = [
        ("Import candidate labels", "t3/import_labels.py",
         "OCAtari / AtariARI RAM→concept candidates"),
        ("Verify by intervention", "t3/verify_labels.jl",
         "set the byte, re-render, confirm the object moves → upgrades to verified-causal"),
        ("Discover new labels", "t3/discover_labels.jl",
         "RAM↔framebuffer correlation + intervention sweeps"),
    ]
    INFRA = [
        ("common/results.py", "tools/xai_study/common/results.py", "writes/reads the §R record schema"),
        ("common/jutari_oracle.jl", "tools/xai_study/common/jutari_oracle.jl", "the intervention/gradient oracle on the jutari substrate"),
        ("common/replay.py", "tools/xai_study/common/replay.py", "deterministic replay-to-state (the <code>f&lt;start&gt;+&lt;window&gt;</code> encoding)"),
        ("common/seeds.py", "tools/xai_study/common/seeds.py", "the single seed=0 source"),
        ("common/game_set.json", "tools/xai_study/common/game_set.json", "the fixed 6-game core set"),
        ("repro/make_hash_tables.py", "tools/xai_study/repro/make_hash_tables.py", "SHA-256 ROM + action-stream hashes (verify with --verify)"),
    ]

    diagram = """  ROM  (AutoROM, SHA-256 verified — not redistributed)
    │   jutari substrate · seed 0 · state f120+30 · games: core (6)
    ▼
  ground_truth/oracle_intervene.jl ── exact |Δy(u)| ──▶  the §1 oracle  (T1 causal truth)
    │
    ├─ phaseA_kording/A1..A8.jl       neuroscience battery
    ├─ phaseB_attribution/*.jl        12 XAI methods + N/A audit
    └─ phaseC_mechanistic/*.jl        10 mechanistic-interp methods
         each runner scores its output vs the oracle on the triad  F ∧ S ∧ M
         └─▶ writes a §R record   out/<phase>/<exp>_<game>.json  (+ .npz)
    ▼
  compare/leaderboard.py    ── pure read of every record ──▶  faithfulness × plausibility
  compare/benchmark/run.py  ── ROM-free scoring of a new method against the committed oracle"""

    body = """
<header class="hero"><div class="wrap">
  <h1>Paper 2 — method catalogue &amp; execution stack</h1>
  <p class="lead">Every interpretability method in the benchmark, with a link to its
  implementation, its reference, and the score it is graded on — plus how a run executes,
  what it writes, and where the records live. This is the Paper-2 counterpart of the
  <a href="conformance.html">conformance code tour</a>.</p>
  <p style="margin-top:12px"><a href="#stack">Execution stack</a> ·
  <a href="#oracle">Oracle &amp; T3</a> · <a href="#phaseA">Phase A</a> ·
  <a href="#phaseB">Phase B (attribution)</a> · <a href="#phaseC">Phase C (mechanistic)</a> ·
  <a href="#e6">Leaderboard &amp; benchmark</a></p>
</div></header>

<section id="stack"><div class="wrap">
  <h2>How a measurement runs</h2>
  <p>Because the VCS is fully known and exactly intervenable, every explanation is scored
  against the truth. The <b>intervention oracle</b> records the exact causal effect
  |Δy(u)| of each candidate cause by clamping it and re-running bit-exact. Each method
  runner then produces its own attribution/circuit and is graded against that oracle on the
  correctness triad — <b>F</b> faithful (true causes), <b>S</b> sufficient (predicts held-out
  interventions), <b>M</b> minimal/right-level. Runners are Julia on the jutari substrate
  (jaxtari eager is ≈205× slower), <code>seed = 0</code>, on the fixed 6-game core set,
  inside the Paper-1 bit-exact horizon.</p>
  <pre><code>%s</code></pre>
  <p>Every runner writes a self-describing <b>§R record</b> per game/regime —
  <code>{paper, phase, method, game, state, target_output, metric_name, value, ci, n,
  seed, where, commit, oracle_ref, timestamp}</code> plus an <code>extra{}</code> block
  carrying the exact <code>oracle_abs_delta_per_cause</code> map and the triad
  <code>{F,S,M}</code>. That is what makes the leaderboard a <i>pure read</i> and the
  benchmark ROM-free.</p>
  <h3>Shared infrastructure</h3>
  <table class="tbl"><tr><th>Module</th><th>Role</th></tr>%s</table>
</div></section>

<section id="oracle"><div class="wrap">
  <h2>E1 · the ground-truth oracle &amp; E2 · T3 labels</h2>
  <p class="sub">The instrument everything is scored against, and the game-concept labels.</p>
  %s
  <p class="caption">Oracle records: %s — <b>3</b>. T3 records: %s — <b>17</b>.</p>
</div></section>

<section id="phaseA"><div class="wrap">
  <h2>Phase A — the Jonas &amp; Kording battery (quantified)</h2>
  <p class="sub">Classical neuroscience methods, scored against the true register-transfer
  account. The calibration baseline: rich structure, low faithfulness.</p>
  %s
</div></section>

<section id="phaseB"><div class="wrap">
  <h2>Phase B — attribution / XAI methods</h2>
  <p class="sub">One runner per method; each saliency/attribution map scored against the
  oracle. References are the methods' original papers.</p>
  %s
</div></section>

<section id="phaseC"><div class="wrap">
  <h2>Phase C — mechanistic interpretability</h2>
  <p class="sub">The state trajectory is the “activations”, the program's data-flow is the
  “circuit” — both known exactly, so recovered structure is scored against the truth.</p>
  %s
</div></section>

<section id="e6"><div class="wrap">
  <h2>E6 — leaderboard &amp; the ROM-free benchmark</h2>
  <table class="tbl"><tr><th>Step</th><th>Implementation</th><th>Output</th></tr>
    <tr><td>Cross-tradition leaderboard (pure read of all records)</td><td>%s</td>
      <td>%s</td></tr>
    <tr><td>Headline faithful-vs-plausible demo</td><td>%s</td><td>%s</td></tr>
    <tr><td>Packaged benchmark — score a new method, no ROM needed</td><td>%s</td>
      <td>%s</td></tr>
  </table>
  <p>The leaderboard re-orients each record's faithfulness onto the headline plot
  (faithfulness X vs a transparent plausibility proxy Y) and runs an embedded self-check;
  it never re-runs an experiment. The benchmark scores one method end-to-end against the
  committed oracle records — a third party needs no ROM.</p>
</div></section>

<section><div class="wrap">
  <h2>Reproduce</h2>
  <pre><code># the oracle, then any method runner (Julia on the jutari substrate, seed 0)
julia --project=jutari tools/xai_study/ground_truth/oracle_intervene.jl --game pong
julia --project=jutari tools/xai_study/phaseB_attribution/ig_baseline_sweep.jl --games core
julia --project=jutari tools/xai_study/phaseC_mechanistic/activation_patching.jl --games core

# aggregate (pure reads — no ROM) and score a new method
python3 tools/xai_study/compare/leaderboard.py
python3 tools/xai_study/compare/benchmark/run.py --method magnitude_proxy</code></pre>
  <p class="caption">Full per-phase command list:
  %s.</p>
</div></section>
""" % (
        diagram,
        "".join("<tr><td>%s</td><td>%s</td></tr>" % (src(p, n), d) for n, p, d in INFRA),
        ('<table class="tbl"><tr><th>Step</th><th>Implementation</th><th>What it computes</th></tr>'
         + "".join("<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
                   % (n, src(XS + s, s.rsplit("/", 1)[-1]), d) for n, s, d in (ORACLE + T3))
         + "</table>"),
        src(XS + "ground_truth/out", "ground_truth/out"),
        src(XS + "t3/out", "t3/out"),
        catalogue(PHASE_A, "phaseA_kording/out", 54),
        catalogue(PHASE_B, "phaseB_attribution/out", 166),
        catalogue(PHASE_C, "phaseC_mechanistic/out", 72),
        src(XS + "compare/leaderboard.py", "compare/leaderboard.py"),
        src(XS + "compare/out/leaderboard.json", "compare/out/leaderboard.json"),
        src(XS + "compare/faithful_demo.py", "compare/faithful_demo.py"),
        src(XS + "compare/out/faithful_demo.json", "compare/out/faithful_demo.json"),
        src(XS + "compare/benchmark/run.py", "compare/benchmark/run.py"),
        src(XS + "compare/benchmark/out", "compare/benchmark/out (14)"),
        link("https://github.com/akmaier/UnderstandingVCS/blob/main/xai_paper/xai_2_interpretability/REPRODUCIBILITY.md",
             "REPRODUCIBILITY.md §3"),
    )
    return page("methods.html", "Paper 2 — method catalogue", body)


import json as _json

# Load the actual cross-method audit (leaderboard.json) once, keyed by method name.
_LEADER = {}
try:
    _lb = _json.load(open(os.path.join(REPO, "tools", "xai_study", "compare", "out",
                                        "leaderboard.json")))
    _LEADER = {r["method"]: r for r in _lb.get("rows", [])}
except Exception:
    pass


def audit_row(key):
    return _LEADER.get(M.P2_LEADER.get(key, ""), None)


def audit_faith(key):
    """(faithfulness, ci95, n_games, n_records) from the leaderboard, or None."""
    r = audit_row(key)
    if not r or r.get("faithfulness") is None:
        return None
    return (r["faithfulness"], r.get("faithfulness_ci95") or 0.0,
            r.get("n_games"), r.get("n_records"))


_PHASE_LABEL = {"A": "Phase A · neuroscience battery", "B": "Phase B · attribution / XAI",
                "C": "Phase C · mechanistic interpretability", "NA": "recorded as not-applicable"}
_PHASE_ANCHOR = {"A": "phaseA", "B": "phaseB", "C": "phaseC", "NA": "phaseB"}
_RECDIR = {"A": "phaseA_kording", "B": "phaseB_attribution", "NA": "phaseB_attribution",
           "C": "phaseC_mechanistic"}


_MATRIX = {"A1_connectomics", "A4_correlations", "A6_granger", "path_patching", "acdc"}
_SCATTER = {"activation_patching", "attribution_patching", "das"}


def _hexlab(i):
    return "$%02X" % int(i)


_FP_BAND = {}
def _fp_band():
    """Per pong cell: (% of footprint pixels in the top score band, total px)."""
    if _FP_BAND:
        return _FP_BAND
    try:
        import numpy as np
        A = os.path.join(HERE, "assets", "methods")
        cells = [int(x) for x in open(os.path.join(A, "fp_pong_cells.txt"))]
        fp = np.frombuffer(open(os.path.join(A, "fp_pong.raw"), "rb").read(),
                           np.float32).reshape(len(cells), 210, 160)
        for i, c in enumerate(cells):
            tot = int((fp[i] > 0).sum())
            _FP_BAND[c] = (round(100 * int((fp[i][:33] > 0).sum()) / max(tot, 1)), tot)
    except Exception:
        pass
    return _FP_BAND


def _phaseB_reading(meth):
    """A per-method, data-driven explanation of what its example figure's causal
    region shows (which output, which cells, why the score/paddle appear), plus
    the shared footprint-proxy caveat. Returns '' for non-Phase-B methods."""
    if meth["phase"] != "B":
        return ""
    import numpy as np
    import re as _re
    b = os.path.join(REPO, "tools", "xai_study", "phaseB_attribution", "out", meth["record"])
    try:
        rec = _json.load(open(b + ".json"))
        npz = dict(np.load(b + ".npz", allow_pickle=True))
    except Exception:
        return ""
    ex = rec.get("extra", {})
    note = ex.get("output_note", "")
    ci = ex.get("content_ram_index")
    is_score = "score@" in note
    if is_score:
        mm = _re.search(r"score@ram\[(\d+)\]", note)
        oc = int(mm.group(1)) if mm else None
        outdesc = ("the <b>score</b> (score@RAM&nbsp;%s)" % _hexlab(oc)) if oc else "the <b>score</b>"
    elif ci is not None:
        outdesc = ("the content of <b>RAM&nbsp;%s</b> (byte&nbsp;%d) — the most causally-active "
                   "concept byte at this state" % (_hexlab(ci), ci))
    else:
        outdesc = "a single content byte"
    orac = np.abs(np.asarray(npz.get("oracle_abs_delta", []), float))
    names = ex.get("cause_names", [])
    per = {}
    for k, nm in enumerate(names):
        m = _re.search(r"ram\[(\d+)\]", str(nm))
        if m and k < len(orac):
            per[int(m.group(1))] = per.get(int(m.group(1)), 0.0) + float(orac[k])
    top = [c for c, v in sorted(per.items(), key=lambda x: -x[1]) if v > 0][:3]
    band = _fp_band()

    def phrase(c):
        s = band.get(c, (0, 0))[0]
        if s >= 5:
            return ("RAM&nbsp;%s (%d%% of its footprint sits in the score band, the rest in the "
                    "play area)" % (_hexlab(c), s))
        return "RAM&nbsp;%s (the play area — ball / paddles)" % _hexlab(c)

    tops = "; ".join(phrase(c) for c in top) if top else "no candidate cell"
    score_cells = [c for c in top if band.get(c, (0, 0))[0] >= 5]
    bleed = ""
    if score_cells and not is_score:
        who = "RAM&nbsp;" + "/".join(_hexlab(c) for c in score_cells)
        bleed = (" The <b>score digits</b> appear in the region because %s reach%s them: perturbing "
                 "%s over the 30-frame NOOP window changes the game outcome, and hence the score — a "
                 "<i>downstream</i> effect, not direct rendering." %
                 (who, "es" if len(score_cells) == 1 else "", "it" if len(score_cells) == 1 else "them"))
    missing = ""
    if is_score:
        missing = (" Because it explains the score, its only true-cause is %s, so the region covers "
                   "the score and the ball but <b>not the paddle</b>: the paddle cell (RAM&nbsp;$36) "
                   "is not causal for the score, so it is legitimately absent." %
                   ("RAM&nbsp;" + "/".join(_hexlab(c) for c in top)))
    return """
<section><div class="wrap">
  <h2>Reading this example's causal region</h2>
  <p>This example explains %s. Its strongest true-cause%s: %s.%s%s</p>
</div></section>""" % (outdesc, "s are" if len(top) != 1 else " is", tops, bleed, missing)


def _method_caption(meth):
    ph, k = meth["phase"], meth["key"]
    if ph == "B":
        return ("<b>Top row</b> (image domain, as in Paper 1): the game frame, then the oracle's "
                "<b>true causal region</b> and this method's <b>attributed region</b> — each painted "
                "onto the frame through the screen footprint of the RAM cells it implicates "
                "(brighter = more important). A faithful method's heat matches the oracle's. "
                "<b>Bottom</b>: per-cell importance — oracle "
                "(<span style='color:var(--accent-2)'>green</span>) vs method "
                "(<span style='color:var(--accent)'>blue</span>) — and the deletion/insertion "
                "faithfulness curves (perturb the ranked causes and watch the output move). "
                "<i>Note:</i> the image-domain overlay footprints are illustrative, computed on the "
                "pre-redesign boot frame; the bars, curves and all reported numbers come from the "
                "re-run records on the shared gameplay states.")
    if k in _MATRIX:
        return ("Two adjacency matrices over the candidate RAM cells: the <b>true</b> data-flow "
                "graph and the graph this method <b>recovered</b>. A bright cell (row = cause, "
                "column = effect) is an edge; the difference between the two panels is the error.")
    if k in _SCATTER:
        return ("Each point is one intervention site: its <b>exact causal effect</b> from the "
                "oracle (x) against the method's <b>recovered/approximate effect</b> (y). Points on "
                "the dashed diagonal mean the method recovered the true effect.")
    if k == "A3_tuning":
        return ("Each point is a RAM cell: its <b>true causal importance</b> (x) against its "
                "<b>tuning strength</b> to a game variable (y). "
                "<span style='color:var(--bad)'>Red</span> points are strongly tuned yet not "
                "causal — the trap the metric measures.")
    if k == "A2_lesions":
        return ("Left: the frame. Middle: each cell's lesion importance painted on the screen via "
                "its footprint. Right: lesion importance vs the cell's true causal role.")
    if k == "linear_probing":
        return ("Per labelled RAM cell, the probe's <b>selectivity</b> (accuracy minus a control "
                "task). <span style='color:var(--bad)'>Red</span> bars are cells that are "
                "decodable but <i>not causally used</i> — present ≠ used.")
    if ph == "A":
        return ("Left: the frame. Right: this method's per-cell/per-component result against the "
                "ground-truth importance, with the RAM cells labelled.")
    if ph == "C":
        return ("Left: the frame. Right: the recovered structure/effect against the ground truth "
                "(matched components or preserved behaviour), with the RAM cells labelled.")
    return "Left: a game frame for context. Right: why these methods do not apply to the VCS."


def build_method_page(meth):
    rec = {}
    rpath = os.path.join(REPO, "tools", "xai_study", _RECDIR[meth["phase"]], "out",
                         meth["record"] + ".json")
    try:
        rec = _json.load(open(rpath))
    except Exception:
        pass
    v = rec.get("value")
    vs = ("%.3f" % v) if isinstance(v, (int, float)) else str(v)
    score = ("<b>%s = %s</b> — this example only (%s, state %s); the audit aggregate is below."
             % (esc(rec.get("metric_name", "score")), esc(vs), esc(meth["game"]),
                esc(rec.get("state", "—")))) if rec else ""
    recdir = "tools/xai_study/%s/out" % _RECDIR[meth["phase"]]
    meta = [
        ("Implementation", src(meth["script"])),
        ("Reference", esc(meth["ref"])),
        ("Record", src(recdir + "/" + meth["record"] + ".json", meth["record"] + ".json")),
        ("All records", src(recdir, _RECDIR[meth["phase"]] + "/out")),
    ]
    metahtml = "".join("<dt>%s</dt><dd>%s</dd>" % (k, val) for k, val in meta)

    # "In the audit" — the method's real entry in the cross-method leaderboard
    af = audit_faith(meth["key"])
    ar = audit_row(meth["key"])
    lb = "tools/xai_study/compare/out/leaderboard.json"
    if af:
        faith, ci, ng, nr = af
        plaus = ar.get("plausibility_proxy")
        trad = ar.get("tradition")
        bign = ('<div class="bignum"><div class="b"><strong>%.3f</strong>'
                '<small>faithfulness vs oracle (mean over %s games, ±%.3f CI95)</small></div>'
                '<div class="b"><strong>%s</strong><small>committed records aggregated</small></div>'
                '<div class="b"><strong>%.2f</strong><small>human-plausibility proxy</small></div>'
                '</div>' % (faith, ng, ci, nr, plaus if plaus is not None else 0))
        audit_html = """
<section><div class="wrap">
  <h2>In the audit</h2>
  <p>This is the method's entry in the actual cross-method audit — its faithfulness is the
  <b>mean over all %s core games</b> (%s committed §R records), not the single example shown above.
  Tradition: <b>%s</b>. The example figure (Pong) is one of those records.</p>
  %s
  <p class="caption">Source: %s · the whole leaderboard is on the
  <a href="methods.html#e6">methods page</a> and the
  <a href="paper2.html">Paper&nbsp;2 audit</a>.</p>
</div></section>""" % (ng, nr, esc(str(trad)), bign, src(lb, "leaderboard.json"))
    else:
        audit_html = """
<section><div class="wrap">
  <h2>In the audit</h2>
  <p>Recorded but <b>excluded from the faithfulness leaderboard</b>: these methods have no
  applicable causes on the VCS, so a faithfulness score would be meaningless. See the
  <a href="methods.html#e6">leaderboard</a> and the <a href="paper2.html">Paper&nbsp;2 audit</a>.</p>
  <p class="caption">Source: %s.</p>
</div></section>""" % src(lb, "leaderboard.json")
    body = """
<header class="hero"><div class="wrap">
  <span class="venue" style="color:var(--accent-2);font-weight:600">%s</span>
  <h1>%s</h1>
  <p class="lead">%s</p>
  <p style="margin-top:10px"><a href="methods.html#%s">← back to the method catalogue</a></p>
</div></header>

<section><div class="wrap">
  <div class="fig" style="max-width:1040px;margin:0 auto">
    <a href="assets/methods/%s.png" target="_blank"><img src="assets/methods/%s.png" alt="%s result"></a>
  </div>
  <p class="caption">%s %s</p>
</div></section>

%s

<section><div class="wrap">
  <h2>What it does</h2>
  <p>%s</p>
</div></section>

<section><div class="wrap">
  <h2>How it's scored</h2>
  <p>%s</p>
  <p class="caption">The score is measured against the §1 intervention oracle — never against
  another interpretability method. F (faithful) is always vs the oracle; see the
  <a href="methods.html#stack">execution stack</a>.</p>
</div></section>

%s

<section><div class="wrap">
  <dl class="meta" style="margin-top:0">%s</dl>
  <p class="caption" style="margin-top:14px">The figure is generated from the committed record by
  <a href="%sdocs/gen_method_figures.py"><code>docs/gen_method_figures.py</code></a>; the game frame
  and each RAM cell's screen footprint are produced by
  <a href="%sdocs/render_scenes.jl"><code>render_scenes.jl</code></a> /
  <a href="%sdocs/cell_footprints.jl"><code>cell_footprints.jl</code></a>.</p>
</div></section>
""" % (esc(_PHASE_LABEL[meth["phase"]]), esc(meth["title"]), esc(meth["ref"]),
       _PHASE_ANCHOR[meth["phase"]], meth["key"], meth["key"], esc(meth["title"]),
       _method_caption(meth), score, _phaseB_reading(meth), esc(meth["what"]),
       M.P2_METHOD_SCORED.get(meth["key"], ""), audit_html, metahtml, BLOB, BLOB, BLOB)
    return page("m_%s.html" % meth["key"], meth["title"] + " — Paper 2 method", body)


def main():
    outputs = {
        "index.html": build_index(),
        "paper1.html": build_paper(M.PAPER1),
        "paper2.html": build_paper(M.PAPER2),
        "conformance.html": build_conformance(),
        "methods.html": build_methods(),
        "provenance.html": build_provenance(),
        "environment.html": build_environment(),
        "reproduce.html": build_reproduce(),
    }
    for meth in M.P2_METHODS:
        outputs["m_%s.html" % meth["key"]] = build_method_page(meth)
    for name, content in outputs.items():
        with open(os.path.join(HERE, name), "w") as f:
            f.write(content)
        print("wrote docs/%s (%d bytes)" % (name, len(content)))
    # .nojekyll so GitHub Pages serves our paths verbatim
    open(os.path.join(HERE, ".nojekyll"), "w").close()
    print("built at commit", SHORT)


if __name__ == "__main__":
    main()
