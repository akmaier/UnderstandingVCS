# P2-E9-2 — Reference no-hallucination verification

**Scope:** every bib key in `paper/references.bib` that is actually `\cite`-d in
`paper/sections/*.tex` (55 keys). Each was checked via WebSearch (and WebFetch on a
canonical source where needed) to confirm the work exists with the cited
title / authors / year / venue. Metadata corrections applied directly in
`references.bib`. Standard: Nature submission — a hallucinated citation is fatal.

**Verdict legend:** OK = exists, metadata correct as cited · FIXED = real paper,
metadata corrected in the bib · INTERNAL = self/companion/data artifact, not an
external publication (not a hallucination risk) · SUSPECT = needs PO attention.

Run date: 2026-06-23.

| key | claimed (title / authors / year / venue) | verdict | source URL | action |
|---|---|---|---|---|
| jonas2017could | Could a Neuroscientist Understand a Microprocessor? / Jonas, Kording / 2017 / PLOS Comput. Biol. 13(1) e1005268 | OK | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268 | none |
| selvaraju2017gradcam | Grad-CAM / Selvaraju et al. / 2020 / IJCV 128(2) 336–359, doi 10.1007/s11263-019-01228-7 | OK | https://link.springer.com/article/10.1007/s11263-019-01228-7 | none (journal version; ICCV-2017 orig noted) |
| chattopadhyay2018gradcampp | Grad-CAM++ / Chattopadhyay et al. / 2018 / WACV, 839–847, doi 10.1109/WACV.2018.00097 | OK | https://www.semanticscholar.org/paper/2c1b79a13087a8e9bc2a4446384145e6f85d4820 | none |
| greydanus2018visualizing | Visualizing and Understanding Atari Agents / Greydanus et al. / 2018 / ICML, PMLR 80:1792–1801 | OK | https://proceedings.mlr.press/v80/greydanus18a.html | none (pages confirmed) |
| atrey2020exploratory | Exploratory Not Explanatory / Atrey, Clary, Jensen / 2020 / ICLR | OK | https://arxiv.org/abs/1912.05743 | none |
| such2019atarizoo | An Atari Model Zoo / Such et al. / 2019 / IJCAI, 3260–3267, doi 10.24963/ijcai.2019/452 | OK | https://www.ijcai.org/Proceedings/2019/0452.pdf | none |
| bellemare2013arcade | The Arcade Learning Environment / Bellemare et al. / 2013 / JAIR 47:253–279, doi 10.1613/jair.3912 | OK | https://jair.org/index.php/jair/article/view/10819 | none |
| xitari | Xitari: an ALE fork / DeepMind / 2016 / github.com/google-deepmind/xitari | OK | https://github.com/google-deepmind/xitari | none (repo confirmed) |
| jang2017gumbel | Categorical Reparameterization with Gumbel-Softmax / Jang, Gu, Poole / 2017 / ICLR | OK | https://openreview.net/forum?id=rkE3y85ee | none |
| bengio2013estimating | Estimating or Propagating Gradients through Stochastic Neurons / Bengio, Léonard, Courville / 2013 / arXiv:1308.3432 | OK | https://arxiv.org/abs/1308.3432 | none |
| jaderberg2015spatial | Spatial Transformer Networks / Jaderberg et al. / 2015 / NeurIPS 28 | OK | https://proceedings.neurips.cc/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf | none |
| sundararajan2017axiomatic | Axiomatic Attribution for Deep Networks / Sundararajan, Taly, Yan / 2017 / ICML, PMLR 70:3319–3328 | OK | https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf | none |
| delfosse2023ocatari | OCAtari / Delfosse et al. / 2023 / arXiv:2306.08649 | OK | https://arxiv.org/abs/2306.08649 | none |
| anand2019unsupervised | Unsupervised State Representation Learning in Atari / Anand et al. / 2019 / NeurIPS 32 | OK | https://arxiv.org/abs/1906.08226 | none |
| hewitt2019designing | Designing and Interpreting Probes with Control Tasks / Hewitt, Liang / 2019 / EMNLP, 2733–2743 | OK | https://aclanthology.org/D19-1275/ | none |
| jacovi2020towards | Towards Faithfully Interpretable NLP Systems / Jacovi, Goldberg / 2020 / ACL, 4198–4205, doi 10.18653/v1/2020.acl-main.386 | OK | https://aclanthology.org/2020.acl-main.386/ | none |
| yang2019benchmarking | Benchmarking Attribution Methods with Relative Feature Importance / Yang, Kim / 2019 / arXiv:1907.09701 | OK | https://arxiv.org/abs/1907.09701 | none (see SUSPECT note: dup of yang2019bim) |
| lindner2023tracr | Tracr: Compiled Transformers as a Laboratory for Interpretability / Lindner et al. / 2023 / NeurIPS 36, arXiv:2301.05062 | OK | https://papers.nips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract-Conference.html | none |
| gupta2024interpbench | InterpBench / Gupta, Arcuschin, Kwa, Garriga-Alonso / 2024 / NeurIPS D&B Track, arXiv:2407.14494 | OK | https://proceedings.neurips.cc/paper_files/paper/2024/hash/a8f7d43ae092d9a5295775eb17f3f4f7-Abstract-Datasets_and_Benchmarks_Track.html | none |
| barbiero2025neural | Neural Interpretable Reasoning / Barbiero et al. / 2025 / arXiv:2502.11639 | FIXED | https://arxiv.org/abs/2502.11639 | author list corrected: removed non-author "Lio, Pietro", added "Espinosa Zarlenga, Mateo", fixed Diligenti/Giannini order |
| lazebnik2002radio | Can a Biologist Fix a Radio? / Lazebnik / 2002 / Cancer Cell 2(3) 179–182, doi 10.1016/S1535-6108(02)00133-2 | OK | https://pubmed.ncbi.nlm.nih.gov/12242150/ | none (dup of lazebnik2002can) |
| lazebnik2002can | Can a Biologist Fix a Radio? / Lazebnik / 2002 / Cancer Cell 2(3) 179–182 | OK | https://pubmed.ncbi.nlm.nih.gov/12242150/ | none (dup of lazebnik2002radio) |
| marr1982vision | Vision / Marr / 1982 / W. H. Freeman | OK | https://direct.mit.edu/books/monograph/3299 | none |
| simonyan2014deep | Deep Inside Convolutional Networks / Simonyan, Vedaldi, Zisserman / 2014 / ICLR Workshop | OK | https://dblp.org/rec/journals/corr/SimonyanVZ13.html | none |
| zeiler2014visualizing | Visualizing and Understanding Convolutional Networks / Zeiler, Fergus / 2014 / ECCV, 818–833 | OK | https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53 | none |
| fong2017interpretable | Interpretable Explanations by Meaningful Perturbation / Fong, Vedaldi / 2017 / ICCV, 3429–3437 | OK | https://openaccess.thecvf.com/content_iccv_2017/html/Fong_Interpretable_Explanations_of_ICCV_2017_paper.html | none |
| petsiuk2018rise | RISE / Petsiuk, Das, Saenko / 2018 / BMVC | OK | https://dblp.org/rec/conf/bmvc/PetsiukDS18.html | none |
| ribeiro2016why | "Why Should I Trust You?" (LIME) / Ribeiro, Singh, Guestrin / 2016 / KDD, 1135–1144 | OK | https://dl.acm.org/doi/10.1145/2939672.2939778 | none |
| lundberg2017unified | A Unified Approach to Interpreting Model Predictions (SHAP) / Lundberg, Lee / 2017 / NeurIPS | OK | https://papers.nips.cc/paper/7062 | none |
| abnar2020quantifying | Quantifying Attention Flow in Transformers / Abnar, Zuidema / 2020 / ACL, 4190–4197 | OK | https://aclanthology.org/2020.acl-main.385/ | none |
| vig2020investigating | Investigating Gender Bias … Causal Mediation Analysis / Vig et al. / 2020 / NeurIPS | OK | https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html | none |
| meng2022locating | Locating and Editing Factual Associations in GPT (ROME) / Meng et al. / 2022 / NeurIPS | OK | https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html | none |
| geiger2021causal | Causal Abstractions of Neural Networks / Geiger, Lu, Icard, Potts / 2021 / NeurIPS | OK | https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html | none |
| geiger2023finding | Finding Alignments … Distributed Neural Representations (DAS) / Geiger et al. / 2024 / CLeaR, PMLR 236:160–187, arXiv:2303.02536 | OK | https://proceedings.mlr.press/v236/geiger24a.html | none |
| nanda2023attribution | Attribution Patching: Activation Patching at Industrial Scale / Nanda / 2023 / neelnanda.io (blog) | OK | https://www.neelnanda.io/mechanistic-interpretability/attribution-patching | none |
| wang2022interpretability | Interpretability in the Wild (IOI) / Wang et al. / 2023 / ICLR, arXiv:2211.00593 | OK | https://iclr.cc/virtual/2023/poster/11341 | none |
| conmy2023towards | Towards Automated Circuit Discovery (ACDC) / Conmy et al. / 2023 / NeurIPS, arXiv:2304.14997 | OK | https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html | none |
| cunningham2023sparse | Sparse Autoencoders Find Highly Interpretable Features / Cunningham et al. / 2023 / arXiv:2309.08600 | OK | https://arxiv.org/abs/2309.08600 | none |
| chan2022causal | Causal Scrubbing / Chan et al. / 2022 / AI Alignment Forum | OK | https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN | none |
| alain2017understanding | Understanding Intermediate Layers Using Linear Classifier Probes / Alain, Bengio / 2017 / ICLR Workshop | OK | https://openreview.net/pdf?id=HJ4-rAVtl | none |
| belrose2023eliciting | Eliciting Latent Predictions … Tuned Lens / Belrose et al. / 2023 / arXiv:2303.08112 | OK | https://arxiv.org/abs/2303.08112 | none |
| shrikumar2017learning | Learning Important Features … (DeepLIFT) / Shrikumar, Greenside, Kundaje / 2017 / ICML, PMLR 70:3145–3153 | OK | https://proceedings.mlr.press/v70/shrikumar17a.html | none |
| springenberg2015striving | Striving for Simplicity: The All Convolutional Net / Springenberg et al. / 2015 / ICLR Workshop | OK | https://dblp.org/rec/journals/corr/SpringenbergDBR14.html | none |
| smilkov2017smoothgrad | SmoothGrad / Smilkov et al. / 2017 / ICML Workshop on Vis. for DL | OK | https://arxiv.org/abs/1706.03825 | none |
| erion2021improving | Improving Performance … Expected Gradients / Erion et al. / 2021 / Nat. Mach. Intell. 3(7) 620–631 | OK | https://www.nature.com/articles/s42256-021-00343-w | none |
| fong2019understanding | Understanding Deep Networks via Extremal Perturbations / Fong, Patrick, Vedaldi / 2019 / ICCV, 2950–2958 | OK | https://openaccess.thecvf.com/content_ICCV_2019/html/Fong_Understanding_Deep_Networks_via_Extremal_Perturbations_and_Smooth_Masks_ICCV_2019_paper.html | none |
| strumbelj2014explaining | Explaining Prediction Models … Feature Contributions / Štrumbelj, Kononenko / 2014 / KAIS 41(3) 647–665 | OK | https://link.springer.com/article/10.1007/s10115-013-0679-x | none |
| olson2021counterfactual | Counterfactual State Explanations for RL / Olson et al. / 2021 / Artif. Intell. 295:103455 | OK | https://www.sciencedirect.com/science/article/pii/S0004370221000060 | none |
| adebayo2018sanity | Sanity Checks for Saliency Maps / Adebayo et al. / 2018 / NeurIPS 31 | OK | https://dblp.org/rec/conf/nips/AdebayoGMGHK18.html | none |
| bastani2018verifiable | Verifiable RL via Policy Extraction / Bastani, Pu, Solar-Lezama / 2018 / NeurIPS 31 | OK | https://proceedings.neurips.cc/paper/2018/hash/e6d8545daa42d5ced125a4bf747b3688-Abstract.html | none (see SUSPECT note: page range varies by index) |
| yang2019bim | BIM: Towards Quantitative Evaluation … Ground Truth / Yang, Kim / 2019 / arXiv:1907.09701 | OK | https://dblp.org/rec/journals/corr/abs-1907-09701.html | none (see SUSPECT note: dup of yang2019benchmarking) |
| ieee610 | IEEE Std 610.12-1990 Glossary of SW Eng. Terminology / IEEE / 1990 | OK | https://ieeexplore.ieee.org/document/159342 | none |
| maier2025vcs | A Bit-Exact, Differentiable Atari 2600 (Paper 1) / Maier, Bayer, Krauss / 2025 / companion paper | INTERNAL | — | none (companion paper in this program) |
| leaderboardP2 | Cross-tradition leaderboard data artifact / Paper 2 authors / 2026 | INTERNAL | — | none (committed analysis record) |
| faithfuldemoP2 | Faithful-method demonstration data artifact / Paper 2 authors / 2026 | INTERNAL | — | none (committed analysis record) |

## SUSPECT / PO attention (no fabrication found; dedup / minor-metadata only)

1. **Duplicate of one real paper (arXiv:1907.09701), cited under TWO keys with TWO
   titles** — `yang2019benchmarking` ("Benchmarking Attribution Methods with
   Relative Feature Importance") and `yang2019bim` ("BIM: Towards Quantitative
   Evaluation of Interpretability Methods with Ground Truth"). Both titles are
   genuine alternate titles of the *same* Yang & Kim 2019 arXiv paper (v2 retitle;
   dblp/Semantic Scholar still index it as "BIM…"). Not a hallucination, but the
   paper is cited twice under different keys. **PO decision:** keep both only if the
   sections deliberately reference the "BAM/relative-feature-importance" framing vs
   the "BIM/ground-truth" framing separately; otherwise merge to one key. DoD
   requires "BIM (Yang & Kim 2019, arXiv:1907.09701)" be present and verified — it is
   (`yang2019bim`).

2. **`bastani2018verifiable` page range** — cited as 2494–2504; NeurIPS-2018 indexes
   variously list 2494–2504 and 2499–2509 (Curran printed vs other indexes). Title,
   authors, year, venue all confirmed. Pages are source-dependent and not
   load-bearing; left as cited rather than substitute an equally-unverified number.
   **PO option:** drop the `pages` field to avoid an index-dependent value.

3. **Duplicate radio cite** — `lazebnik2002radio` and `lazebnik2002can` are the same
   paper (Cancer Cell 2002, 2(3):179–182) under two keys. Both verified real. **PO
   decision:** dedup to one key if only one is needed.

(Several other bib entries — e.g. `vig2020causalmediation`, `meng2022rome`,
`geiger2021abstraction`, `conmy2023acdc`, `wang2022ioi`, `nanda2023attributionpatching`,
`alain2017probing`, `hewitt2019control`, `belrose2023tunedlens`, `anand2019atariari` —
are duplicate variants present in the file but **not `\cite`-d** anywhere, so they are
out of this verification's scope. They are dead entries; PO may prune them, but they
cannot reach the rendered bibliography.)

## Summary
- 55 cited keys checked.
- **OK: 51** (incl. 3 internal data/companion artifacts, not externally verifiable).
- **FIXED: 1** (`barbiero2025neural` author list).
- **SUSPECT (dedup / minor metadata, no fabrication): 3 issues** across
  `yang2019benchmarking`+`yang2019bim`, `bastani2018verifiable` pages,
  `lazebnik2002radio`+`lazebnik2002can`.
- **No hallucinated / non-existent citation found.** Every cited work resolves to a
  genuine publication or a committed internal artifact.
