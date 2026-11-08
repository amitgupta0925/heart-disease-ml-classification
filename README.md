# Machine Learning Assignment 2 – UCI Heart Disease

**Programme:** M.Tech. Artificial Intelligence & Machine Learning  
**Course:** Machine Learning  
**Dataset:** Combined UCI Heart Disease (Cleveland, Hungary, Switzerland, VA Long Beach)  
**Problem type:** Binary classification

> This project is an educational machine-learning demonstration, not a medical
> diagnostic system. It must not be used to make clinical decisions.

## 1. Problem statement

The objective is to build and compare five supervised classification algorithms
that predict whether a patient record indicates the presence or absence of heart
disease. The original UCI diagnosis field (`num`) ranges from 0 to 4. For this
binary classification exercise, `num = 0` is mapped to class 0 (absence), while
`num = 1, 2, 3, or 4` is mapped to class 1 (presence).

## 2. Dataset description

The dataset is sourced from the UCI Machine Learning Repository's Heart Disease
collection. The combined version merges four processed source files:

| Source | Original records |
|---|---:|
| Cleveland | 303 |
| Hungary | 294 |
| Switzerland | 123 |
| VA Long Beach | 200 |
| **Total** | **920** |

There are 13 predictor variables and one target variable. The 13 predictors are:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`,
`oldpeak`, `slope`, `ca`, and `thal`.

The data also retains a `source` column in the combined master CSV for provenance.
It is not used as a model input. The original `num` diagnosis field is transformed
to `target` as described above. Missing predictor values are imputed inside the
training pipeline. Zero values in `trestbps` and `chol` are treated as missing
because they are not physiologically plausible measurements.

**Dataset provenance note:** UCI's catalog page displays 303 instances for its
processed Cleveland dataset and lists the four source databases/files separately.
This project explicitly combines the four processed files to obtain the
approximately 920-record version. Do not describe the Cleveland-only 303-row
file as the 920-row dataset.

**UCI source and citation**

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
*Heart Disease*. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X

Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease

## 3. Project structure

```text
Heart_Disease_Assignment_2/
├── app.py
├── download_data.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv                 # created by train_models.py
├── data/
│   ├── raw/                      # four downloaded UCI files
│   └── processed/
│       └── heart_disease_combined.csv
├── models/                       # five saved model pipelines
└── outputs/
    ├── model_comparison.csv
    ├── classification_reports.txt
    ├── dataset_summary.json
    └── dataset_summary.txt
```

## 4. Models and evaluation metrics

The following five classifiers are trained on the same stratified 80:20
train/test split (random state 42):

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The preprocessing pipeline imputes missing numeric values with the training-set
median, scales numeric variables, imputes categorical values with the
training-set most frequent value, and one-hot encodes categorical variables.
The preprocessing steps are fitted only on the training data to avoid data
leakage. Each complete fitted pipeline is saved using Joblib.

### Evaluation metrics

- **Accuracy:** proportion of all test records classified correctly.
- **AUC:** area under the ROC curve; measures ranking/discrimination across
  classification thresholds.
- **Precision:** among records predicted positive, the proportion actually
  positive.
- **Recall:** among actual positive records, the proportion identified.
- **F1 score:** harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient; ranges from -1 to +1, with +1
  indicating perfect prediction, 0 roughly random prediction, and -1 total
  disagreement.

### Model comparison table

Run `python train_models.py` to generate actual values in
`outputs/model_comparison.csv`. Copy those measured values into the table below
and into the PDF submission. Do not insert guessed or borrowed scores.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Run code | Run code | Run code | Run code | Run code | Run code |
| Decision Tree | Run code | Run code | Run code | Run code | Run code | Run code |
| KNN | Run code | Run code | Run code | Run code | Run code | Run code |
| Gaussian Naive Bayes | Run code | Run code | Run code | Run code | Run code | Run code |
| Support Vector Machine | Run code | Run code | Run code | Run code | Run code | Run code |

### Observation template

Use the actual comparison table and classification reports to complete these
observations after execution:

- **Logistic Regression:** discuss its measured AUC, precision/recall balance,
  and whether its linear decision boundary appears adequate.
- **Decision Tree:** discuss its test performance and whether its scores suggest
  possible overfitting compared with the other models.
- **KNN:** discuss the effect of distance-based classification after scaling
  and its measured precision/recall.
- **Gaussian Naive Bayes:** discuss its measured results under the conditional
  independence assumption.
- **SVM:** discuss its measured AUC/F1 and performance with the RBF kernel.
- **Overall winner:** identify the model using the evaluation objective you
  justify (for example, F1 or AUC), not accuracy alone. Consider the class
  distribution and the consequences of false negatives and false positives.

## 5. GitHub repository link

**Add your repository URL here after creating and pushing the project:**

`[PASTE YOUR GITHUB REPOSITORY URL]`

## 6. Streamlit application link

**Add your deployed application URL here after deployment:**

`[PASTE YOUR STREAMLIT COMMUNITY CLOUD URL]`

The app provides:
- CSV upload for held-out test data only
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report

The app loads the saved model pipeline and evaluates the uploaded labelled test
CSV. It does not retrain the model on uploaded test data.

## 7. How to run locally or in BITS Virtual Lab

Use Python 3.10 or 3.11.

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Download and combine the four UCI source files
python download_data.py

# 3. Train all five models and create the test CSV
python train_models.py

# 4. Start the Streamlit application
python -m streamlit run app.py
```

If automatic downloads are blocked, download these four files from the UCI
Heart Disease files page and place them in `data/raw/` with the exact filenames:

- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

UCI files page:
https://archive.ics.uci.edu/dataset/45/heart+disease

## 8. Streamlit Community Cloud deployment

1. Push the complete project to a GitHub repository.
2. Confirm that `app.py`, `requirements.txt`, `README.md`, the `models/` folder,
   and `test_data.csv` are committed.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Create an app and select the repository and `main` branch.
5. Set the main file path to `app.py`.
6. Deploy and test the public app with `test_data.csv`.
7. Paste the live URL in Section 6 above.

The model files are generated locally by `train_models.py`; commit the five
`.joblib` files to the repository so the deployed app can load them.

## 9. BITS Virtual Lab evidence and submission checklist

The assignment asks for execution in BITS Virtual Lab and one screenshot as
proof. Run the data preparation and training scripts in BITS Virtual Lab, then
capture your own screenshot showing successful execution. Do not use a
fabricated or unrelated screenshot.

Before submitting the single PDF, verify:

- [ ] GitHub repository link works and contains complete source code.
- [ ] `requirements.txt` is included.
- [ ] `README.md` is included in the repository and its content is included in
      the submitted PDF.
- [ ] `test_data.csv` is included.
- [ ] All five saved model files are included.
- [ ] The metrics table contains actual outputs from your run.
- [ ] Observations and overall winner are based on your actual results.
- [ ] Streamlit app is deployed and opens correctly.
- [ ] App supports CSV upload, model selection, all metrics, and confusion
      matrix or classification report.
- [ ] BITS Virtual Lab execution screenshot is included.
- [ ] PDF contains, in the assignment's requested order: GitHub link, live
      Streamlit link, BITS Lab screenshot, and README content.

## 10. Academic integrity

Run the code yourself in BITS Virtual Lab, understand each preprocessing and
modelling step, and make your own observations from your results. Keep your own
Git commit history and customize the report in your own words. The generated
metrics depend on the exact source files, software versions, and run settings.
