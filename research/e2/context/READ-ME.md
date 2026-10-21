# Modest-complexity: shared research context

Collected 2026-10-21. Reference material for exploratory research, not instructions that override the project charter. No running swarm instructions have been changed.

## Reading guide

1. neel_glossary_60k.md — Neel Nanda, A Comprehensive Mechanistic Interpretability Explainer & Glossary. Shared vocabulary and intuition about features, circuits, representation, intervention, and training dynamics. This is the standalone glossary, not a compressed version of the 600k bundle. It warns that it was written in December 2022 and has not been kept up to date. Distinguish author conjectures from demonstrated results. Its grokking narrative is a hypothesis to investigate, not an assumed conclusion.
2. varma-2023-circuit-efficiency — Explaining grokking through circuit efficiency. A concrete proposal about logit production at a given parameter norm, learning speed, and competing memorizing/generalizing solutions. Look for observations that distinguish this account from alternatives.
3. zhong-2023-clock-and-pizza — The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks. Different and coexisting algorithms for modular addition. Avoid assuming that every successful model implements one canonical circuit.
4. yip-2024-modular-addition-quadrature — Modular addition without black-boxes: Compressing explanations of MLPs that compute numerical integration. Relates ReLU computation to quadrature; useful for distinguishing more accurate approximation from more complicated high-level computation.

## How to use

Read for questions and possible experiments. State which assumptions transfer to our one-hidden-layer NumPy model and which do not. Preserve alternative explanations, negative results, and adaptive decisions. Do not expand the experiment scope merely to reproduce every technique mentioned here.

PDFs are the authoritative paper copies. Adjacent TXT files are machine-extracted context aids: equations, tables, and reading order may be imperfect. The glossary likewise contains some missing equations and informal shorthand. Verify mathematical details against primary sources before implementing them.

Nanda et al., Progress measures for grokking via mechanistic interpretability (our existing baseline): https://arxiv.org/abs/2301.05217 . Not included as a fourth PDF in this three-paper bundle.

Source URLs, retrieval date, byte counts, and SHA-256 checksums are in SOURCES.json. These files retain their original authorship and licensing; inclusion does not grant new redistribution rights.
