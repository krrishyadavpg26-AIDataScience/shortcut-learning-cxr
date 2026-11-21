# Generative AI usage log

The project guidelines ask for Generative / Agentic AI tools to be used with appropriate
prompts and context. This file records how they were used, so the contribution is transparent.
Add a row every time you use an AI tool, and keep the prompt wording.

| # | Date | Tool | Purpose | Prompt (summary or exact) | What I kept / changed / verified |
|---|------|------|---------|---------------------------|----------------------------------|
| 1 | 2026-11-05 | Claude | Topic selection | "Need a college level project which could be made in a day or two. Research paper check of recent years." (with the course guideline PDF attached) | Chose project 7 (shortcut learning) from the suggested list |
| 2 | 2026-11-06 | Claude | Project plan | "Provide the steps from start to finish to do this project" | Used the phase plan; changed ___ |
| 3 | 2026-11-07 | Claude | Literature search | Asked for recent papers on shortcut learning in chest X-rays | Read both key papers myself (Sourget 2025, Boland 2025) and checked the gap |
| 4 | 2026-11-07 | Claude | Code scaffolding | "Give the steps to run on my laptop ... provide the research material, project proposal ... steps for running the experiment" | Reviewed every file; ran smoke test; changed ___ |
| 5 |  |  | Debugging |  |  |
| 6 |  |  | Results interpretation |  |  |

| 7 | 2026-11-21 | Singularity Alpha | Run diagnosis | Asked it to run JTT with up-weight 50; it traced a segfault to the GPU being switched off (Eco mode) | Switched GPU mode back, verified with nvidia-smi and a torch check; results/ untouched |
| 8 | 2026-11-21 | Singularity Alpha | JTT analysis + experiment | Asked for per-seed JTT error sets by group and the up-weight 50 vs 20 comparison | Checked group counts sum to error-set sizes; corrected its claim that every metric worsened (ΔTPR and Grad-CAM improved slightly) |

## My own contributions

- Ran all experiments on my RTX 5080 laptop and checked the numbers.
- Choices I made or changed: ___ (e.g. marker size, ρ values, JTT up-weight)
- Analysis and conclusions in the slides are my own reading of `results/summary.md`.

## Verification practice

- Every claim taken from a paper was checked against the paper itself.
- Code produced with AI help was read line by line and tested with
  `python main.py --fake-data --quick --no-pretrained` before the real run.
