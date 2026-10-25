"""Beginner's guides for the Paper-2 method pages (docs/m_<key>.html).

One module per method, ``docs/guides/<key>.py``, each defining ``GUIDE = dict(...)``.
``build_pages.build_guide_section`` renders it as the "Learn the method" block.
The site doubles as an introduction to explainable AI: every guide explains the
method the way it is used on neural networks, then shows how the same method is
applied to the Atari 2600 VCS, where — unlike for a network — the ground truth is
known, and links the exact lines of both implementations (the XAI runner and the
emulator substrate).

Schema (HTML strings; KaTeX math with \\( \\) inline and \\[ \\] display):

  key        method key (must match manifest.P2_METHODS)
  level      short tag line, e.g. "Gradient family · the starting point"
  wiki       [(label, en.wikipedia.org URL)]   background concepts, checked live
  papers     [(citation html, URL)]            the original paper(s) + one critique
  idea       html  — plain-language intuition for a deep-learning beginner
  nn         html  — how the method works on a neural network (formula + toy example)
  recipe     [html] — the algorithm as numbered steps
  atari_map  [(neural-net concept, VCS counterpart)] — the translation table
  atari      html  — how we apply it to the Atari, concretely
  code       [dict(path, find, role, desc)] — `find` is a literal substring of the
             line to link; build_pages resolves it to a line number at build time and
             FAILS if it is missing, so links cannot rot silently. role: xai | substrate
  reading    html  — how to read this page's figure / what result to expect
  pitfalls   [html] — known failure modes, in general and on the VCS
  try_it     shell command that reproduces the record
  try_note   html  — what the command writes / how long it takes
  next       [method keys] — what to read next
"""
import importlib
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_all():
    out = {}
    for fn in sorted(os.listdir(_HERE)):
        if fn.endswith(".py") and not fn.startswith("_"):
            mod = importlib.import_module("guides." + fn[:-3])
            g = getattr(mod, "GUIDE", None)
            if g:
                out[g["key"]] = g
    return out
