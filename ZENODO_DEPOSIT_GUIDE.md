# Complete Guide: Depositing CA-CRS+ on Zenodo

This guide provides step-by-step instructions for publishing the **CA-CRS+** research paper, source code, and benchmark artifacts to **Zenodo** (https://zenodo.org).

---

## 1. Why Zenodo?

- **Permanent Digital Object Identifier (DOI):** Gives you a citable, immutable reference (e.g., `10.5281/zenodo.XXXXXXX`) that can be cited in research papers, CVs, and technical reports.
- **Indexed Globally:** Zenodo records are automatically indexed by Google Scholar, Semantic Scholar, OpenAlex, and DBLP.
- **Open Access & Long-Term Preservation:** Hosted by CERN on behalf of OpenAIRE, ensuring at least 20+ years of guaranteed preservation.
- **High Capacity:** Allows uploads up to 50 GB per dataset free of charge.
- **Versioning:** Allows uploading new revisions (v1.1, v2.0) while keeping the original DOI intact and linked.

---

## 2. What to Prepare for the Deposit

Prepare the following 3 files to upload:

| File Name | Description | Mandatory? |
|---|---|---|
| `CA-CRS-Plus-Research-Paper.pdf` | The complete 5-page revised research paper manuscript. | **Yes** |
| `TENCON-2026-Review-Responses.pdf` | Detailed author responses addressing all peer review comments. | **Recommended** |
| `cacrsplus-v1.0-code-archive.zip` | Complete clean source code, tests, calibration scripts, and configs. | **Yes** |

### How to Create `cacrsplus-v1.0-code-archive.zip`
From the root of this repository, run:
```bash
zip -r cacrsplus-v1.0-code-archive.zip . \
  -x "*.git*" \
  -x "*.venv*" \
  -x "*__pycache__*" \
  -x "*.DS_Store" \
  -x "models/*.pth" \
  -x "models/*.pt" \
  -x "outputs/tables/all_datasets.csv" \
  -x "outputs/tables/ucf_qnrf.csv" \
  -x "outputs/tables/shanghaitech_both.csv"
```
*(This produces a lean ~5–10 MB bundle containing all code, reproduction scripts, parameter JSONs, and test suites).*

---

## 3. Step-by-Step Deposit Instructions (Web Interface)

### Step 1: Sign In to Zenodo
1. Visit **[https://zenodo.org](https://zenodo.org)**.
2. Click **Log in** at the top right.
3. Sign in using your **GitHub account** (`Priyanshu8yadav`) or **ORCID**.

### Step 2: Create a New Upload
1. Click the green **+ New Upload** button in the top navigation bar (or visit [https://zenodo.org/deposit/new](https://zenodo.org/deposit/new)).

### Step 3: Upload Files
1. Drag and drop:
   - `CA-CRS-Plus-Research-Paper.pdf`
   - `TENCON-2026-Review-Responses.pdf` (optional but valuable for provenance)
   - `cacrsplus-v1.0-code-archive.zip`
2. Click **Start upload** to upload all files.

### Step 4: Reserve a DOI (Important!)
- In the **Basic information** section under **Digital Object Identifier (DOI)**, click **Reserve DOI**.
- Zenodo will immediately generate your permanent DOI string (e.g. `10.5281/zenodo.14XXXXXX`).
- *You can copy this DOI into your paper or README before pressing Publish!*

### Step 5: Fill in the Metadata

Follow these exact entries:

#### 1. Resource Type
- **Resource type:** `Publication`
- **Publication type:** `Preprint` *(or `Working paper` / `Software`)*

#### 2. Title
```
CA-CRS+: A Zone-Aware, Resource-Informed Crowd Risk Scoring Framework with Cause-Specific Gate Intervention Recommendation for Stampede Prevention
```

#### 3. Authors / Creators
- **Author 1:**
  - Family name: `Yadav`
  - Given name: `Priyanshu Kumar`
  - Affiliation: `Department of Computer Science and Engineering, Rajkiya Engineering College, Mainpuri, Uttar Pradesh, India`
  - Email: `priyanshu322006@gmail.com`
- **Author 2:**
  - Family name: `Mishra`
  - Given name: `Abhinav Kumar`
  - Affiliation: `Department of Computer Science and Engineering, Rajkiya Engineering College, Mainpuri, Uttar Pradesh, India`
  - Email: `mishrakabhinav34@gmail.com`

#### 4. Description (Abstract & Summary)
Paste the following:
```html
<p><b>CA-CRS+</b> is a zone-aware, resource-informed crowd-risk scoring framework operating on standard RGB video without thermal cameras or cloud dependency. It addresses four critical limitations of prior systems: generic alerts without cause, absence of spatial resolution, lack of gate-specific actionable recommendations, and lack of dynamic emergency resource validation.</p>

<h3>Key Contributions:</h3>
<ul>
  <li><b>Benchmark-Validated Crowd Counting:</b> Density-map regression (DM-Count VGG-19; validated at MAE 48.7 on ShanghaiTech Part A, 14.4 on Part B, and 0.55 against multi-object-tracking ground truth on PETS2009 S2.L1).</li>
  <li><b>Circular-Variance Conflict Index:</b> Normalized directional conflict metric via magnitude-weighted circular variance of optical flow that does not degenerate under balanced counter-flow.</li>
  <li><b>Non-Linear Crush Scoring:</b> Solves the zero-velocity gridlock paradox where dense stationary crowds are scored as safe by linear models. On dense imagery, CA-CRS+ flags 55% of scenes as DANGER where a matched linear baseline flags 0%.</li>
  <li><b>Cause-to-Gate Mapping with Ripple-Effect Downgrade:</b> Emits named gate commands (Density &rarr; OPEN, Speed &rarr; CLOSE, Conflict &rarr; REDIRECT, MIXED &rarr; HOLD) with a safety check preventing crowd redirection from overwhelming downstream zones.</li>
  <li><b>Dynamic Resource-Aware Alerting:</b> Sizes marshal demand logarithmically with crowd population (D_mar = ceil(alpha * log10(1 + N))).</li>
  <li><b>Multi-Camera Fusion:</b> Fuses multiple real CCTV camera streams into an urgency-weighted (2:1) Global Risk Score (GRS).</li>
  <li><b>Edge Performance:</b> Runs at 4.0 frames/s on a single commodity NVIDIA Tesla T4 GPU, supporting real-time 1–2 Hz risk-update cadences.</li>
</ul>

<p><i>Provenance note: This research paper was initially prepared for IEEE Region 10 Conference (TENCON 2026), thoroughly upgraded after peer review to include ground-truth quantitative benchmarking, sensitivity analyses, and multi-camera fusion, and subsequently withdrawn from TENCON by the authors. It is deposited here as a permanent, open-access research preprint and reproducible artifact archive.</i></p>
```

#### 5. License
- Select: **Creative Commons Attribution 4.0 International (CC-BY-4.0)** for the paper/data, or **MIT License** for the code.

#### 6. Keywords
Add these tags:
- `crowd safety`
- `stampede prevention`
- `zone-aware risk scoring`
- `crowd counting`
- `optical flow`
- `causal attribution`
- `gate intervention`
- `multi-camera fusion`
- `low-infrastructure public safety`

#### 7. Related Identifiers
- Add the GitHub repository:
  - **Identifier:** `https://github.com/Priyanshu8yadav/cacrsplus`
  - **Relationship:** `isSupplementTo`
  - **Resource type:** `Software`

---

## 4. Finalizing and Publishing

1. Click **Save draft** to save all entries.
2. Review the preview page. Check that all author names, affiliations, and uploaded files are correct.
3. Click **Publish**.
4. A warning will appear: *"Once published, you will not be able to delete your files."* Click **OK**.
5. Your record is now live! Your DOI is permanent and globally active.

---

## 5. Adding the Zenodo Badge to GitHub

The Markdown badge has been linked in your repository `README.md` and `CITATION.cff`:
```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23691416.svg)](https://doi.org/10.5281/zenodo.23691416)
```
Permanent DOI URL: [https://doi.org/10.5281/zenodo.23691416](https://doi.org/10.5281/zenodo.23691416)
