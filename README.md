# LLM-Assisted Thematic Analysis Pipeline (Gemini)

A Python pipeline for coding open-ended survey responses with an LLM and refining the resulting codebook through structured discussion with the model and comparison against human coding. I built it for faculty-led research at Carleton College that analyzes 20 years of student responses to the question "What is computer science?", collected before and after an introductory CS course.

It adapts the framework from *LLM-in-the-loop: Leveraging Large Language Model for Thematic Analysis* (Dai, Xiong, and Ku, EMNLP 2023) to the Gemini API. 
Our changes include batched coding of hundreds of responses, merging codebooks across batches, and a discussion step for resolving disagreements between my coding and the model's.

**No data is included.** This repo contains code and prompts only. The exemplar file contains fake exemplars I wrote myself instead of student data.

## How the pipeline works

`ThematicAnalysis.ipynb` runs the steps in order:

1. **Initial coding.** Few-shot prompting with exemplars, in batches, with each code justified by a quote from the response.
2. **Summary.** The coded output is converted into structured JSON of response, code, and definition.
3. **Themes.** Codes are grouped into named themes with short definitions.
4. **More data.** Additional batches are coded against the current codebook, new codes are proposed, and codebooks are merged.
5. **Discussion.** I propose changes to the codebook (merging, renaming, moving codes) and the model responds with agreements and objections before it produces a revised version.
6. **Final codebook.** Codes get numeric IDs and definitions.
7. **Labeling.** The full dataset is labeled in small batches (with a pause between calls for rate limits), then exported as a table.
8. **Human validation.** A random subset is pulled out for hand coding. Disagreements between my labels and the model's feed back into step 5 to sharpen definitions.

## Files

- `gemini_chat_mods.py`: the pipeline functions and Gemini API wrapper
- `ThematicAnalysis.ipynb`: the workflow, run cell by cell
- `init_coding_task.txt`: instructions for the initial coding prompt
- `exemplars.txt`: Shows the exact format and "thinking" process we want the model to follow.  Contains fake exemplars I wrote myself instead of student data.

