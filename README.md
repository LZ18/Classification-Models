📂 Dataset

The dataset used is the SVHN (Street View House Numbers) dataset.

🔗 Official Stanford Link (Format 2):
https://ufldl.stanford.edu/housenumbers/

SVHN contains real-world images of house numbers extracted from Google Street View.
Unlike MNIST, digits appear with complex backgrounds, varying lighting, and multiple digits per image, making the classification task more challenging.

✨ Features of This Program
🧮 Implemented Machine Learning Models

Gaussian Naive Bayes

Multinomial Logistic Regression

Principal Component Analysis (PCA) + Logistic Regression

🏎️ Performance Comparison

The program measures:

Accuracy

Precision/Recall

Training time

Prediction time

Impact of PCA on speed and accuracy

🖼️ Dataset Processing

The SVHN dataset does not include standard “rows and columns.”
Instead, it uses pixel images, which require flattening:

3D image array → 1D feature vector


This allows models like Logistic Regression and Naive Bayes to work effectively.

📊 PCA Dimensionality Reduction

Reduces high-dimensional pixel inputs → smaller feature space

Improves training speed

Evaluates accuracy tradeoffs

🛠️ Technologies & Libraries Used
Python

Core language for all training and experimentation

NumPy / Pandas

For numerical operations and managing pixel matrices

Scikit-learn

Used for:

Logistic Regression

Naive Bayes

PCA

Train/test splitting

Accuracy scoring

Matplotlib

For plotting accuracy graphs (optional)

📁 Project Structure
Assignment3/
 ├── main.py                 # Main script to run all models
 ├── utils/
 │     └── loader.py         # Helper functions for loading SVHN
 ├── models/
 │     ├── naive_bayes.py
 │     ├── logistic_regression.py
 │     └── pca_logistic.py
 ├── results/
 │     ├── accuracy_report.txt
 │     └── graphs/
 └── README.md
