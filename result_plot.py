import numpy as np
import matplotlib.pyplot as plt

# Data for sample sizes
sample_sizes = [1000, 10000, 100000, 150000]

# Numeric data
numeric_metrics = {
    "rating": {
        "MSE": [2.2278, 1.7258, 1.3032, 1.2975],
        "RMSE": [1.4926, 1.3137, 1.1416, 1.1391]
    },
    "review_count": {
        "MSE": [117670.3942, 85030.0967, 82919.1288, 83267.9016],
        "RMSE": [343.0312, 291.5992, 287.9568, 288.5618]
    },
    "weight": {
        "MSE": [10.1897, 8.5494, 8.4618, 8.1443],
        "RMSE": [3.1921, 2.9239, 2.9089, 2.8538]
    },
    "price": {
        "MSE": [117688.4276, 84165.9621, 82971.9398, 83211.6397],
        "RMSE": [343.0575, 290.1137, 288.0485, 288.4643]
    }
}

# Category data
category_metrics = {
    "brand": [66.97, 67.99, 33.21, 66.73],
    "size": [37.56, 67.99, 66.99, 66.88],
    "color": [41.63, 40.45, 19.84, 40.04],
    "material": [28.05, 50.14, 50.05, 24.93],
    "category": [18.10, 40.40, 39.70, 20.11]
}

# Baselines
numeric_baseline = 0
category_baselines = {
    "brand": 67.00,
    "size": 67.00,
    "color": 66.00,
    "material": 67.00,
    "category": 80.00
}

# Plot numeric data
plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
for metric, values in numeric_metrics.items():
    plt.plot(sample_sizes, values["MSE"], label=f"{metric} MSE")
    plt.plot(sample_sizes, values["RMSE"], linestyle="--", label=f"{metric} RMSE")
plt.axhline(y=numeric_baseline, color="red", linestyle="--", label="Numeric Baseline")
plt.title("Numeric Data Metrics")
plt.xlabel("Sample Size")
plt.ylabel("Metrics")
plt.legend()
plt.grid()

# Plot categorical data
plt.subplot(2, 1, 2)
for metric, values in category_metrics.items():
    plt.plot(sample_sizes, values, label=f"{metric} Accuracy")
    plt.axhline(y=category_baselines[metric], linestyle="--", label=f"{metric} Baseline")
plt.title("Categorical Data Metrics")
plt.xlabel("Sample Size")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()

# Save and show the plot
plt.tight_layout()
plt.savefig("metrics_vs_sample_size.png")
plt.show()

# Calculate overall metrics
numeric_mse = []
numeric_rmse = []
for metric, values in numeric_metrics.items():
    numeric_mse.extend(values["MSE"])
    numeric_rmse.extend(values["RMSE"])

overall_mse = np.mean(numeric_mse)
overall_rmse = np.mean(numeric_rmse)

category_accuracy = []
for metric, values in category_metrics.items():
    category_accuracy.extend(values)

overall_accuracy = np.mean(category_accuracy)

print("Overall Numeric MSE:", overall_mse)
print("Overall Numeric RMSE:", overall_rmse)
print("Overall Categorical Accuracy:", overall_accuracy)
