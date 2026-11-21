# Generative AI usage log

The project guidelines ask for Generative / Agentic AI tools to be used with appropriate
prompts and context. This file records how they were used, so the contribution is transparent.

| # | Date | Tool | Purpose | Prompt (summary or exact) | What I kept / changed / verified |
|---|------|------|---------|---------------------------|----------------------------------|
| 1 | 2026-11-05 | Claude | Topic selection | "Need a college level project which could be made in a day or two. Research paper check of recent years." (with the course guideline PDF attached) | Chose project 7 (shortcut learning) from the suggested options |
| 2 | 2026-11-06 | Claude | Project plan | "Provide the steps from start to finish to do this project" | Used the phase plan as the project structure |
| 3 | 2026-11-07 | Claude | Literature search | Asked for recent papers on shortcut learning in chest X-rays | Read both key papers (Sourget 2025, Boland 2025) and checked the research gap |
| 4 | 2026-11-07 | Claude | Code, proposal, runbook | "Give the steps to run on my laptop ... provide the research material, project proposal ... steps for running the experiment" | Ran the smoke test and a quick run before the full experiment |
| 5 | 2026-11-08 | Claude | Setup debugging | Shared the pytorch.org install page showing CUDA 13.2; told it I have Python 3.14 | Installed Python 3.14 + PyTorch 2.15 (CUDA 13.2) instead of the suggested 3.11 / CUDA 12.8; check_setup.py confirmed the RTX 5080 |
| 6 | 2026-11-08 | Claude | Quick-run check | Shared results_quick/summary.md and figures | Confirmed sanity checks (clean AUC 0.987, gap near 0 at ρ = 50%) before running the full experiment |
| 7 | 2026-11-08 | Claude | Results analysis + slides | Shared results/summary.md, all_runs.csv and figures from the full run | All numbers in the deck come from my results files |
| 8 | 2026-11-13 | Claude | Interpretation + speaker notes | "Provide the interpretation of the results and speaker notes of the slides. Also mention my contribution and how it will help further research" | Reviewed the interpretation and notes before presenting |
| 9 | 2026-11-21 | Singularity Alpha | Run diagnosis | Asked it to run JTT with up-weight 50; it traced a segfault to the GPU being switched off (Eco mode) | Switched GPU mode back; verified with nvidia-smi and a torch check; results/ untouched |
| 10 | 2026-11-21 | Singularity Alpha | JTT analysis + experiment | Asked for per-seed JTT error sets by group and the up-weight 50 vs 20 comparison | Checked group counts sum to error-set sizes; corrected its claim that every metric worsened (ΔTPR and Grad-CAM improved slightly) |
| 11 | 2026-11-21 | Claude | GitHub + proposal | Asked how to push to GitHub; asked to fit the proposal on one page | Fixed the remote URL error and pushed; proposal checked as one A4 page |

## My own contributions

- Chose the project, the research gap and the dataset from the suggested options.
- Set up the environment myself (Python 3.14, PyTorch with CUDA 13.2) and ran all 27 training runs on my RTX 5080 laptop.
- Ran the sanity checks (smoke test, quick run) before the full experiment.
- Directed the JTT follow-up with Singularity Alpha and checked its output against the result files.
- Reviewed the analysis, slides and speaker notes, and will present the work.

## Verification practice

- Claims taken from papers were checked against the papers themselves.
- Code produced with AI help was tested with `python main.py --fake-data --quick --no-pretrained` and a quick real run before the full experiment.
- AI-generated analysis was checked against `results/summary.md` and `results/all_runs.csv`.