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
    ("paper2.html", "Paper 2"),
    ("conformance.html", "Conformance"),
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


def main():
    outputs = {
        "index.html": build_index(),
        "paper1.html": build_paper(M.PAPER1),
        "paper2.html": build_paper(M.PAPER2),
        "conformance.html": build_conformance(),
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
