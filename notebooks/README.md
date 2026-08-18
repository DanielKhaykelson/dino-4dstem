# DINO-4DSTEM tutorial notebooks

Prefer notebooks to the GUI? These walk through the whole workflow on a
**synthetic WS2 phantom** — a simulated 4D-STEM scan where the correct answer is
known, so you can *measure* how well the model recovers it.

## Run them

Double-click **`launch_notebooks.bat`** (in the project root). It opens Jupyter
Notebook in this folder using the DINO-4DSTEM environment — installing Jupyter
itself the first time if needed. Then open a notebook and run the cells
top-to-bottom.

> Prefer the terminal? `conda activate dino4dstem`, then `jupyter notebook` from
> this folder.

## The four notebooks

| # | Notebook | What it covers |
|---|---|---|
| 01 | `01_make_WS2_phantom.ipynb` | build the WS2 phantom, its classes/domains, the virtual image, the diffracting domains, and the output data |
| 02 | `02_load_preprocess_train.ipynb` | load the cube, pre-process it, and train the DINO model with the default recipe |
| 03 | `03_analyze_classmap.ipynb` | the class map, IoU / ARI / NMI vs the ground truth, the confusion & cosine matrices, per-class diffraction |
| 04 | `04_NMF_kmeans.ipynb` | the classical NMF + KMeans baseline, compared to DINO and the ground truth |

Run them **in order** — each uses what the previous one produced.

`ws2_utils.py` holds the shared helpers (phantom generation, virtual images, a
live drag-to-select ROI, and the metrics). Import it with
`import ws2_utils as wu`.

> The **live ROI** needs the widget backend: put `%matplotlib widget` at the top
> of the cell (already noted in the notebooks).
