# VNA Parameter Extraction Algorithm

![MATLAB](https://img.shields.io/badge/MATLAB-R2025b-0076A8)
![Optimization Toolbox](https://img.shields.io/badge/Optimization%20Toolbox-required-0076A8)
![Workflow](https://img.shields.io/badge/workflow-interactive-6f42c1)

**Material permittivity and loss tangent extraction from VNA measurements using an interactive MATLAB workflow.**

**Initialisation Correction.** The entry point now resolves its configuration relative to its own location, checks that Optimization Toolbox is available, and asks for the measurement directory when none is configured. These changes address startup failures when MATLAB opens the repository from different working directories. The extraction calculations are unchanged.

The software was developed at the University of Birmingham. It supports generalized reflection line (GRL) and thru reflect line (TRL) workflows, CSV and Touchstone inputs, and interactive control of initial fit conditions.

## Requirements

- MATLAB with a graphical desktop. The startup check was exercised with R2025b on macOS; other releases and operating systems have not yet been verified.
- Optimization Toolbox, including `fmincon`. In MATLAB, run `license('test','Optimization_Toolbox')` to check the licence.
- A measurement directory arranged as shown in [folder_struct](folder_struct).

Python, Poetry, and a compiler are not required for this MATLAB repository.

## Download and start

Download this repository using **Code → Download ZIP** and extract it, or clone `https://github.com/alperensar/UoB_VNA_Extraction.git`. Open the extracted repository folder in MATLAB. Keep `src/runAllMaterials.m` and `src/config_VNA.json` together.

In the MATLAB Command Window, run:

```matlab
addpath(fullfile(pwd, 'src'));
runAllMaterials
```

Use this same command on Windows, macOS, or Linux after opening the repository folder in MATLAB. No permanent MATLAB path change is needed. If you prefer to start from another folder, replace `pwd` with the absolute path to the extracted repository. For example:

```matlab
addpath(fullfile('C:', 'Users', 'YourName', 'Documents', 'UoB_VNA_Extraction', 'src'));
runAllMaterials
```

The second example shows the Windows path form; replace every example directory with the one on your computer. On macOS or Linux, use the same command with a path under your home directory.

When prompted, select the measurement root directory. The default `FilePatterns.rootDir` in `src/config_VNA.json` is empty so the software asks you to select your own data. If you set it to an existing directory, that directory is used directly. Select CSV or S2P input, then use the interactive controls to review and export the fit. You may also call `runAllMaterials('path/to/config_VNA.json')` to select a separate configuration.

## Measurement data and configuration

The example [folder_struct](folder_struct) shows a material directory with measurement-system and frequency-band subdirectories. The material name must match an entry in `Materials` in [src/config_VNA.json](src/config_VNA.json). The configuration defines material thickness and surface roughness, input patterns, GRL/TRL selection, frequency bands, and output settings. Edit these values for your own measurements before extraction. Keep the supplied configuration structure intact.

If the program reports `VNA:ConfigurationNotFound`, check that `src/config_VNA.json` exists or provide its path explicitly. If it reports `VNA:MissingOptimizationToolbox`, install or license Optimization Toolbox. If no files are found, check the selected root directory, the material names, and the CSV/S2P format selected in the first dialog.

## Citation

If you use this software in research, please cite:

A. Sari et al., “Interlaboratory mmWave and THz Quasi-Optical Characterization of Commercial Conventional and 3-D Printable Substrates,” *IEEE Transactions on Terahertz Science and Technology*, 2026 (under review).

## Author

Alperen Sari — University of Birmingham
GitHub: [@alperensar](https://github.com/alperensar)
