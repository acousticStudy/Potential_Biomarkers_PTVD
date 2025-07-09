README: SHAP-Based Identification of Potential Acoustic Biomarkers in Patients with Post-Thyroidectomy Voice Disorder
==================================================================

Overview
--------
This repository contains MATLAB and Python scripts developed for machine learning analysis of acoustic voice features, primarily focused on preoperative vs. postoperative voice classification.

The pipeline includes preprocessing, feature selection (via LASSO), model training (SVM and Boosting algorithms), performance evaluation, and SHAP-based explainability analysis.

Environment
-----------
- MATLAB R2025a
- Python 3.11

Contents
-------

1. SVMModelsCode.m  
   - Main MATLAB script to train SVM classifiers (Cubic, Quadratic, RBF).  
   - Performs stratified data splitting (80% train, 20% test), z-score normalization, LASSO feature selection, 3-fold cross-validation, test evaluation, and SHAP analysis.  
   - Performance metrics: Accuracy, Precision, Recall, Specificity, F1-score, ROC AUC.  
   - SHAP values computed using MATLAB's `shapley()` function.  
   - Tested on MATLAB R2025a. Older versions may also work if `shapley()` function is available.

2. BoostingModelsCode.m  
   - Trains GentleBoost and LogitBoost classifiers.  
   - Similar structure to the SVMModelsCode.  
   - Learning rate, number of learners, and other hyperparameters can be adjusted.  

3. smartCI.m  
   - Computes confidence intervals for any performance metric vector.  
   - Chooses between parametric (normal-based) or bootstrap methods based on Shapiro-Wilk test results.  
   - Returns mean, lower/upper CI bounds, method used, and normality p-value.  

4. swtest.m  
   - Custom implementation of the Shapiro-Wilk and Shapiro-Francia tests.  
   - Determines if the metric distribution is normal.  
   - Used internally by `smartCI.m`.  
   - swtest.m function is authored by Ahmed Ben Saïda (IHEC Sousse - Tunisia) and is publicly available with full credit. No modifications were made to the original implementation.

5. generateCITable.m  
   - Aggregates performance metrics across runs and applies `smartCI` to produce formatted confidence intervals.  
   - Designed to be used after 100-iteration model runs.  
   - Outputs tables compatible with export to Excel or LaTeX.  

6. shapboxscatter_diagonal.py  
   - Python script to visualize SHAP generalizability.  
   - Reads SHAP values from Excel sheets ("Train_SHAP", "Test_SHAP", "SHAP_Correlation").  
   - Plots scatter graphs comparing train vs. test SHAP means.  
   - Calculates Spearman correlation and exports all figures and coordinate logs to a dedicated folder.  

Dependencies
------------
Python script requires the following libraries:
- pandas
- numpy
- matplotlib
- seaborn

Install with:
pip install pandas numpy matplotlib seaborn

MATLAB scripts require following:
- Statistics and Machine Learning Toolbox

Usage Notes
-----------
- All scripts are modular and can be run independently.  
- `shapley()` is used in MATLAB for SHAP value computation.  
- Z-score normalization and feature selection are applied to the training set only (to avoid data leakage).  
- SHAP scatter plots are intended to assess model generalizability and feature stability.  

Quick Start (MATLAB)
--------------------
1. Open MATLAB R2025a.
2. Run SVMModelsCode.m or BoostingModelsCode.m.
3. Ensure all helper functions (smartCI, swtest, generateCITable) are in the same directory or MATLAB path.

Quick Start (Python)
--------------------
1. Run shapboxscatter_diagonal.py after exporting SHAP values to Excel.
2. Figures and SHAP coordinates will be saved to "SHAP_Figures_diagonal" folder.

Reproducibility
---------------
- All randomization is controlled using `random_state=2`.  
- Iterations: 100 runs per model with validation and test metrics recorded.  
- SHAP values are computed for both train and test sets to assess consistency.  

File Integrity (SHA-256 Hashes)
-------------------------------
The following SHA-256 hash codes were computed for critical files to ensure reproducibility and integrity. 
Any change in these files will result in a different hash value.

- "BoostingModelsCode.m": "1ad22fcf6dec0bee9b8cb7aa9252d37a611a6402920fe0c44a390681a689e012",
- "Dataset.xlsx": "a7792b797c58e8bf289880c3fc80cd3cf9406176ab3e9ab260becbc4062ba2b1",
- "generateCITable.m": "dd6840a3d394493f5494e7f3262cc048f561e0c5e9ff08128e4531c78f5b328f",
- "shapboxscatter_diagonal.py": "32ab7af1fa38a2808ede2b34522712a08ea34149162a1a16be9ca6fd5a4203e2",
- "smartCI.m": "b0c1305a4049ff3375b93a8b31e63ab7608ba15f195ec3c2c0e880b51a3b2c4c",
- "SVMModelsCode.m": "781d247eee9ebc783b7ff7351b815916d83058ead01abc8ce159554eb461e6b9",
- "swtest.m": "0f3b818e616e4a825dc922fe750a1caf606f894deaf034ef0a1e9ddddbbedcf4"

Citation
--------
If you use any part of this codebase in your research, please cite the related publication.

Note: This repository is shared for peer-review purposes only. No license has been granted at this stage due to anonymous submission, and inclusion of third-party code (e.g., swtest.m). Please do not reuse or redistribute without permission. A proper open-source license will be added after peer review is complete.

Contact
-------
This repository is currently shared as part of an anonymous peer-review process. Contact information will be provided after acceptance.

