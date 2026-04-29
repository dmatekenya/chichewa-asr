## Female Test Dataset Construction

### Overview

The female test dataset was constructed as part of the benchmarking pipeline for Automatic Speech Recognition (ASR) model evaluation. The objective was to create a **high-quality, representative, and balanced subset of female speech data** to support fair and robust model comparison.

Given variability in audio quality and metadata completeness, a combination of **manual curation** and **rule-based processing** was used.

This dataset will later be combined with a corresponding male/other subset to form the **final benchmark test dataset**.

---

### Step 1: Manual Identification of Female Speakers

A subset of audio files was manually curated to identify **female speakers**. This was done through:

- manual listening of audio clips  
- identification of speaker characteristics  
- grouping of files believed to belong to the same speaker  

This curated subset served as the source pool for constructing the female benchmark dataset.

---

### Step 2: Duration Filtering and Bucketing

Audio clips were filtered and categorized based on duration:

- clips shorter than **1 second** were excluded  
- clips longer than **30 seconds** were retained but handled separately  

The remaining clips were assigned to non-overlapping duration buckets:

- **1–5 seconds**
- **5–10 seconds**
- **10–15 seconds**
- **15–20 seconds**
- **20–30 seconds**
- **>30 seconds** (retained without sampling constraints)

This ensures coverage across different utterance lengths.

---

### Step 3: Target-Based Sampling

A target of approximately **45 minutes** of female speech was defined. This target intentionally exceeds the minimum required duration to account for potential removal of low-quality clips during later validation.

Sampling was performed independently within each duration bucket using predefined targets:

- **1–5 seconds** → 5 minutes  
- **5–10 seconds** → 7.5 minutes  
- **10–15 seconds** → 7.5 minutes  
- **15–20 seconds** → 12.5 minutes  
- **20–30 seconds** → 12.5 minutes  
- **>30 seconds** → all available clips included  

Within each bucket:

- clips were randomly sampled until the target duration was reached  
- if the available audio was less than the target, **all clips were included**  

This approach ensures both **duration diversity** and **robust coverage**.

---

### Step 4: Speaker Identification

Since explicit speaker metadata was not available, speaker identities were inferred using filename-based heuristics.

The following rules were applied:

- files containing known names (e.g., *Priscilla*) were assigned that name as `speaker_id`  
- files with the prefix **"AU"** were grouped under a single speaker  
- files with suffixes such as `_1`, `_2`, etc., were treated as belonging to the same speaker  
  - e.g., `mofaya_1`, `mofaya_2` → `mofaya`  
- all other files used the cleaned base filename as a proxy speaker identifier  

This provides a **practical approximation of speaker identity**, sufficient for sampling and evaluation.

---

### Step 5: Dataset Assembly

The sampled audio files were:

- copied into a single output directory (flat structure, no subfolders)  
- accompanied by a metadata file containing key attributes  

This produces a **clean and easy-to-use dataset** for benchmarking and manual review.

---

## Dataset Columns

The final dataset includes the following columns:

### `audio_filename`
- Name of the audio file  
- Used to locate and reference the corresponding audio clip  

---

### `transcript_filename`
- Name of the transcript file associated with the audio  
- Links to the text transcription of the spoken content  

---

### `duration_seconds`
- Duration of the audio clip in seconds  
- Used for filtering, bucketing, and sampling  

---

### `transcript`
- Text transcription of the audio clip  
- Serves as the ground truth for ASR evaluation  

---

### `speaker_id`
- Estimated speaker identifier derived from filename heuristics  
- Used to approximate speaker-level grouping and diversity  
- Not guaranteed to represent true unique speakers, but sufficient for:
  - sampling control  
  - speaker diversity analysis  
  - reducing over-representation  

---

## Notes

- Speaker identification is **approximate** and based on heuristic rules  
- The dataset is **intentionally oversampled** (45 minutes target) to allow for later quality filtering  
- Duration buckets ensure **balanced representation across utterance lengths**  
- This dataset forms the **female component** of the final benchmark dataset and will be combined with a corresponding male/other subset  
- The dataset is intended for **evaluation and benchmarking**, not for model training  