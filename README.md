# Crack segmentation: U-Net vs. a tuned classical pipeline

Pixel-wise crack segmentation in photos of concrete and asphalt, trained and tested on the
public **DeepCrack** dataset. The project compares a compact **U-Net (PyTorch)** with a
**classical image-processing pipeline** (black-hat filter + thresholding). Both methods are
tuned only on training data and scored with the same metric on the same 237 test images.

![Metrics on the test set](images/metrics.png)

## Results (DeepCrack test set, 237 images)

| Method | Precision | Recall | **F1 (2 px tol.)** | F1 (strict) | mean IoU |
|---|---|---|---|---|---|
| Classical: CLAHE → black-hat → mean + 1.5·std → blob filter | 0.869 | 0.866 | 0.868 | 0.766 | 0.629 |
| U-Net (1.94 M parameters, CPU-trained) | **0.957** | 0.865 | **0.909** | **0.798** | **0.658** |

- **U-Net minus classical, F1 (2 px tol.):** +0.041, 95% CI [+0.023, +0.059]. This is a paired bootstrap over test images, 2,000 resamples.
- The U-Net is better on **167 of 237** test images. Median per-image F1 is 0.953 for the U-Net and 0.918 for the classical pipeline.
- Most of the gain is **precision**: the U-Net marks far fewer false cracks (joints, stains, coarse aggregate). Recall is the same for both methods.

<p align="center"><img src="images/per_image_f1.png" width="380"></p>

![Example test images](images/examples.png)

## Method

**Data.** [DeepCrack](https://github.com/yhlleo/DeepCrack) has 537 RGB images (544 × 384 px) with
hand-drawn binary masks: 300 training images and 237 test images. Cracks cover about 3% of the
pixels. 40 training images (seed 0) are held out as **validation**, which is used to choose the
epoch and the decision threshold. The test images are used only once, in `evaluate.py`.

**U-Net** (`model.py`, `train.py`)
- **Architecture:** 4 levels, 16 base channels, BatchNorm, 1.94 M parameters.
- **Loss:** weighted BCE (pos_weight 5, because cracks are rare) plus soft Dice.
- **Augmentation:** 256 px crops that prefer crack regions, flips, 90° rotations, brightness/contrast/colour jitter, blur, noise.
- **Optimisation:** AdamW with a OneCycle schedule. The best epoch is chosen by validation F1, and the threshold by validation F1 over 0.30–0.98. The chosen threshold is 0.88.

**Classical baseline** (`classical.py`, `tune_classical.py`)
- **Pipeline:** grey value → CLAHE → optional Gaussian smoothing → morphological black-hat (bright where the surface has dark structures narrower than the kernel) → threshold (Otsu or mean + k·std) → removal of small and non-elongated blobs.
- **Tuning:** grid search on 80 training images with the same F1 metric (1,400 combinations).
- **Grid widening:** the first grid had its best values on its edges, so it was extended (twice) until the optimum lay inside it. This raised the baseline's training F1 from 0.766 to 0.855.
- **Final parameters:** kernel 121 px, no smoothing, k = 1.5, min_area 50 px, min_elongation 4.

**Metric** (`metrics.py`). Crack masks are only a few pixels wide and hand annotation is not
pixel-exact, so the main metric allows a **2 px tolerance**, as is usual in crack-detection
papers. A predicted pixel counts as correct if a true crack pixel lies within 2 px, and the same
rule applies to recall. Pixels are pooled over all images. The strict pixel F1 and mean IoU are
also reported.

## Honest limitations

- **The U-Net is under-trained.** It was trained on a CPU with a 30-epoch schedule and stopped at epoch 19. The best epoch is 15, with validation F1 0.870. Longer GPU training (`train_colab.ipynb`: base 32, 80 epochs) should help.
- **It misses wide, low-contrast cracks on smooth concrete.** The model is conservative (high threshold, precision 0.96). On blurry, wide cracks it marks only the darkest core, while the black-hat filter captures the whole width. These are the three worst test images:

  ![Failure cases](images/failure_cases.png)
- **The classical baseline is strong.** Once properly tuned, it is only about 4 F1 points behind. Without the grid widening, it would have looked much weaker than it is.
- **One dataset, one domain.** DeepCrack photos are close-ups taken under similar conditions. Expect lower scores on other cameras, distances or surfaces, such as drone images, wet roads or shadows.
- **Small validation set.** 40 images means the choice of epoch and threshold is noisy. Validation F1 jumps between epochs, see `results/training_log.csv`.
- **No comparison with published DeepCrack numbers.** Papers use different tolerances and metrics (ODS/OIS), so the numbers are not directly comparable.

## How to run

```bash
git clone https://github.com/elmoussaouioumayma/deep-learning-crack-segmentation.git
cd deep-learning-crack-segmentation
pip install -r requirements.txt

python tune_classical.py        # downloads DeepCrack into data/, tunes the baseline (~12 min on 4 CPU cores)
python train.py --epochs 30     # trains the U-Net -> results/unet.pt (~3 min/epoch on CPU)
cp results/unet.pt weights/unet.pt
python evaluate.py              # test-set comparison -> results/test_metrics.json, images/*.png

python predict.py my_photo.jpg  # any image size -> my_photo_crack_mask.png, my_photo_crack_overlay.png
python app.py                   # local Gradio demo
```

GPU training: open `train_colab.ipynb` in Google Colab (T4 GPU). `weights/unet.pt` is the trained
model used for the results above. `choose_threshold.py` re-selects its threshold on the validation
images.

| File | Purpose |
|---|---|
| `data.py` | download, loading, augmentation |
| `model.py` | U-Net and loss |
| `metrics.py` | precision/recall/F1 with tolerance, IoU |
| `classical.py`, `tune_classical.py` | classical baseline and its grid search |
| `train.py`, `choose_threshold.py` | training, threshold selection on validation |
| `evaluate.py` | test-set comparison and figures |
| `predict.py`, `app.py` | inference on any image, Gradio demo (Hugging Face Spaces) |

## Dataset licence and citation

The DeepCrack images are **not** part of this repository. `data.py` downloads them from the
authors' repository, and their use is restricted to **non-commercial research and education**.
The example figures show a few test images for illustration only. If you use the data, please
cite:

> Y. Liu, J. Yao, X. Lu, R. Xie, L. Li. *DeepCrack: A deep hierarchical feature learning
> architecture for crack segmentation.* Neurocomputing 338 (2019) 139–153.
> doi:[10.1016/j.neucom.2019.01.036](https://doi.org/10.1016/j.neucom.2019.01.036)

```bibtex
@article{liu2019deepcrack,
  title   = {DeepCrack: A deep hierarchical feature learning architecture for crack segmentation},
  author  = {Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal = {Neurocomputing},
  volume  = {338},
  pages   = {139--153},
  year    = {2019},
  doi     = {10.1016/j.neucom.2019.01.036}
}
```

The U-Net follows Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image
Segmentation*, MICCAI 2015.

**Code licence:** MIT (code only), © 2026 Oumayma El Moussaoui.
