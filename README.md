# VNA Parameter Extraction Algorithm

This repository contains an in-house **MATLAB** toolkit developed at the **University of Birmingham** for extracting material (dielectric) parameters from **VNA-based** measurements. The workflow is designed to be **robust and fully interactive**: you configure the run via `config_VNA.json`, then perform parameter extraction through a **GUI**—without modifying the source code.

## 🚀 Key Features

- **GUI-based extraction workflow:** Interactive, step-by-step parameter extraction with visual feedback.
- **Config-driven operation:** All key settings are defined in **`config_VNA.json`** (paths, frequency range, model/fit options, output settings, etc.).
- **Robust optimization:** Designed for stable convergence on real measurement data.
- **Initial condition control:** Set / refine **initial guesses** and related settings directly through the GUI to improve convergence.
- **No-code execution:** Run the full extraction process **without editing MATLAB code**.
- **Strict file structure:** The code expects a specific directory layout; an example is provided in **`folder_struct/`**.

---

## 📥 Get the Code (Git)

### Option A — Clone with HTTPS (recommended)

```bash
git clone https://github.com/alperensar/THz_TDS_UoB.git
cd THz_TDS_UoB
```

### Option B — Clone with SSH

```bash
git clone git@github.com:alperensar/THz_TDS_UoB.git
cd THz_TDS_UoB
```

> Alternatively, you can download a ZIP from GitHub (**Code → Download ZIP**), then extract it.

---

## ✅ Requirements

- MATLAB (with standard toolboxes typically used for data processing and GUI operation)

---

## ⚙️ Setup in MATLAB (no install step)

1. **Open MATLAB**
2. **Set your working directory** to the repository root (the folder you cloned)
3. **Add the repository to your MATLAB path** (recommended)

Run this once in the MATLAB Command Window:

```matlab
cd('PATH/TO/THz_TDS_UoB');        % <-- update this
addpath(genpath(pwd));
savepath;
```

---

## 🧭 Typical Workflow

1. **Prepare your dataset folder structure**
   - Follow the required directory hierarchy.
   - Use the provided example in **`folder_struct/`** as a template.

2. **Configure the run**
   - Edit **`config_VNA.json`** to point to your measurement root directory and set extraction options.

3. **Launch the GUI**
   - Start the GUI using the MATLAB entry-point script included in this repo (the `.m` file that launches the GUI).

   Example (replace with your actual GUI launcher filename):
   ```matlab
   run('launch_VNA_GUI.m')
   ```

4. **Run extraction**
   - Load/select the dataset in the GUI.
   - Adjust **initial conditions** if needed.
   - Run the extraction and save/export results from the GUI.

---

## 🗂️ Required Data Structure

The extraction expects a strict directory hierarchy. A working example is provided under:

- `folder_struct/` (example dataset + template layout)

> If your measured data does not match this structure, the GUI may not find inputs automatically.

---

## ⚙️ Configuration (`config_VNA.json`)

`config_VNA.json` is the single source of truth for running the tool. It is used to define things like:

- Measurement root directory and dataset selection
- Frequency range / processing options
- Model and fitting/optimization options
- Output paths and saving behavior

> The goal is: **change configuration, not code**.

---

## 📚 Citations & References

If you utilize this algorithm in your research, please cite:

- Publication 1: A. Sari and others, "Interlaboratory mmWave and THz Quasi-Optical Characterization of Commercial Conventional and 3-D Printable Substrates," *IEEE Transactions on Terahertz Science and Technology*, 2026 (Under Review).

**BibTeX snippet:**
```bibtex
@article{sar2026thz,
  title   = {Interlaboratory mmWave and THz Quasi-Optical Characterization of Commercial Conventional and 3-D Printable Substrates},
  author  = {Sari, Alperen and others.},
  journal = {IEEE Transactions on Terahertz Science and Technology},
  year    = {2026}
}

```

---

## 👤 Author

**Alperen Sari — PhD candidate, University of Birmingham**  
GitHub: **@alperensar**
