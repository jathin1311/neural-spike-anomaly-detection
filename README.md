# Anomaly detection in neural spike recordings

Which recordings in a primate visual-cortex dataset look unlike the rest, and do different unsupervised methods agree on them? Final project for MATH 5466 (Mathematics of Machine Learning and Data Analysis II), University of Minnesota, May 2025.

By Jathin Mallampati.

## Contents

| Path | What it is |
|---|---|
| `report/MATH5466_final_report.pdf` | The final report |
| `notebooks/anomaly_detection.ipynb` | Data loading, autoencoder and optimizer comparison, reconstruction-error scoring, spectral embedding, and one-class SVM |

## Data

The [UW Neural Data Challenge 2019](https://www.kaggle.com/c/uwndc19/data) on Kaggle: responses of 18 neurons in visual area V4 to 551 images, stored in NetCDF as an array of shape (551, 17, 18, 800). The data isn't stored here; download it from Kaggle after accepting the competition rules. The notebook expects the zip file at `/content/drive/MyDrive/uwndc19/uwndc19.zip` in Google Drive.

## Methods

1. **Autoencoder:** each sample is flattened to 244,800 values, compressed to a 32-number code, and reconstructed. SGD, SGD with momentum, and Adam are compared, and samples whose reconstruction error exceeds the mean plus three standard deviations are flagged.
2. **Spectral embedding:** Laplacian eigenmaps on a 10-nearest-neighbor graph of the samples.
3. **One-class SVM:** an RBF-kernel one-class SVM on the autoencoder's codes, shown in two dimensions with PCA.

## Running the notebook

The notebook was written for Google Colab with Google Drive mounted. It uses PyTorch, netCDF4, scikit-learn, and matplotlib.

## History and review notes

The first commit (tag [`original-submission`](https://github.com/jathin1311/neural-spike-anomaly-detection/tree/original-submission)) contains the report and notebook as submitted, except that the report's page header, left over from the LaTeX template, was removed. Later commits are my fixes and additions.

Reviewing the project in September 2026, I found these issues:

1. **The flagged sample number refers to a shuffled order.** Reconstruction errors are computed by looping over a `DataLoader` with `shuffle=True`, so a position in the error list is not a sample number. That is why the notebook flags sample 242 and the report sample 419, and why the plots of "that sample" show a different one. In a reproduction with a known anomaly at index 100, the same code reported 105, 75, and 166 on three passes. The error histogram doesn't depend on order, and it does show one sample with clearly higher error; it just isn't the one plotted. Scoring with an unshuffled loader fixes this.
2. **The figures don't support the claim that the methods agree.** In the spectral embedding, the highlighted sample sits inside the main cluster, and in the one-class SVM plot, the reconstruction-error outlier sits inside the SVM's boundary.
3. **The autoencoder barely trained.** Reconstruction error moved from 0.2506 to 0.2489 over 10 epochs, and an untrained copy of the model already scores about 0.251, so the three-standard-deviation rule separates samples by differences of about 0.0002.
4. **The figures come from different models.** The error histogram and the reconstruction plot use the first model (Adam, learning rate 1e-5); the one-class SVM plot uses the model retrained in the optimizer comparison, where Adam ran at 1e-4, not the 1e-5 the report lists.
5. **The 551 samples are images, not trials.** The challenge has 531 training and 20 evaluation images, recorded from 18 neurons.
6. **Missing values are replaced with zeros,** which treats unrecorded data as silence. The long zero stretches in both plotted samples suggest many unrecorded repeats.
