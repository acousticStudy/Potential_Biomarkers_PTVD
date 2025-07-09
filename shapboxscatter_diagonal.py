import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_shap_figures(excel_path, model_name, output_folder="SHAP_Figures_diagonal"):
    os.makedirs(output_folder, exist_ok=True)

    # Read the Excel sheets
    shap_corr = pd.read_excel(excel_path, sheet_name="SHAP_Correlation")
    shap_train = pd.read_excel(excel_path, sheet_name="Train_SHAP")
    shap_test = pd.read_excel(excel_path, sheet_name="Test_SHAP")

    # === 1. Histogram Plot ===
    rho_vals = shap_corr["SpearmanRho"].dropna()
    plt.figure(figsize=(10, 6))
    sns.histplot(rho_vals, bins=20, kde=True, color='skyblue', edgecolor='black')
    plt.title(f"{model_name} - SHAP Correlation Histogram", fontsize=16)
    plt.xlabel("Spearman's Rho", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    hist_path = os.path.join(output_folder, f"{model_name}_SHAP_Correlation_Histogram.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()

    # === 2. Scatter Plot: Train vs Test SHAP values ===
    train_means = shap_train.mean(skipna=True)
    test_means = shap_test.mean(skipna=True)
    common_features = train_means.index.intersection(test_means.index)

    # Create DataFrame of coordinates
    df_coords = pd.DataFrame({
        "Feature": common_features,
        "Train_SHAP": train_means[common_features].values,
        "Test_SHAP": test_means[common_features].values
    })
    
    # Save SHAP coordinates
    scatter_coordinates_path = os.path.join(output_folder, f"{model_name}_SHAP_Coordinates.xlsx")
    df_coords.to_excel(scatter_coordinates_path, index=False)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df_coords["Train_SHAP"], df_coords["Test_SHAP"], color='navy', alpha=0.7)
    for i, row in df_coords.iterrows():
        plt.text(row["Train_SHAP"], row["Test_SHAP"], row["Feature"], fontsize=8, alpha=0.8)
    min_val = min(df_coords["Train_SHAP"].min(), df_coords["Test_SHAP"].min())
    max_val = max(df_coords["Train_SHAP"].max(), df_coords["Test_SHAP"].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', linewidth=1.5)
    plt.xlabel("Mean SHAP Value (Train)", fontsize=14)
    plt.ylabel("Mean SHAP Value (Test)", fontsize=14)
    plt.title(f"{model_name} - SHAP Scatter Plot (Train vs Test)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    scatter_path = os.path.join(output_folder, f"{model_name}_SHAP_Scatter_TrainVsTest.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # === 3. Save Captions ===
    caption_text = f"""{model_name} - SHAP Correlation Histogram
This histogram illustrates the distribution of Spearman's correlation coefficients between training and test SHAP values across all iterations. A peak near +1 indicates strong consistency.

{model_name} - SHAP Scatter Plot (Train vs Test)
This scatter plot compares the average SHAP values of stable features in the training and test sets. Points close to the diagonal line indicate high SHAP agreement between sets.
"""

    caption_path = os.path.join(output_folder, f"{model_name}_SHAP_Captions.txt")
    with open(caption_path, "w") as f:
        f.write(caption_text)

    return hist_path, scatter_path, caption_path, scatter_coordinates_path

excel_path = "SHAP_Results_Stable.xlsx"
model_name = "Quadratic SVM C=0.1"

plot_shap_figures(excel_path, model_name)
