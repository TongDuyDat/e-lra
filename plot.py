import pandas as pd
import matplotlib.pyplot as plt

# Reading the CSV file
df = pd.read_csv(
    "D:/NCKH/NCKH2025/LGPS/logs/logs_20250605_214639/logs_20250605_214639/metrics.csv"
)

# Limit to first 200 epochs
df_200 = df.iloc[:200].copy()

# Define training, validation, and loss metrics
train_metrics = [
    "train_mean_iou",
    "train_precision",
    "train_recall",
    "train_f2",
    "train_accuracy",
    "train_dice",
]
val_metrics = ["mean_iou", "precision", "recall", "accuracy", "dice", "f2"]
train_loss = ["train_g_loss", "train_d_loss", "val_loss"]

# Metrics where lower is better (losses)
loss_metrics = ["train_g_loss", "train_d_loss", "val_loss"]

# Colors for different metrics
colors = ["blue", "green", "red", "orange", "purple", "brown", "cyan", "magenta"]


# Function to find peak epoch and value (min for loss, max for others)
def get_peak_info(metric, df):
    if metric in loss_metrics:
        peak_idx = df[metric].idxmin()
        peak_value = df[metric].min()
    else:
        peak_idx = df[metric].idxmax()
        peak_value = df[metric].max()
    peak_epoch = df.loc[peak_idx, "epoch"]
    return peak_epoch, peak_value


# Create a figure with 3 subplots (2 rows, 2 columns, last cell empty)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # Flatten to easily index subplots
fig.suptitle(
    "Model Performance Metrics Over 200 Epochs", fontsize=16, fontweight="bold", y=1.05
)
# List to collect legend handles and labels for shared metrics legend
metric_handles = []
metric_labels = []

# Plot 1: Loss Metrics with Minimum Points (No Annotations)
for i, metric in enumerate(train_loss):
    axes[0].plot(df_200["epoch"], df_200[metric], label=metric, color=colors[-i])
    peak_idx = df_200[metric].idxmin()
    peak_value = df_200[metric].min()
    peak_epoch = df_200.loc[peak_idx, "epoch"]
    offset = (20, 20) if i % 2 == 0 else (30, -20)
    axes[0].annotate(
        f"{peak_value:.6f}\nE:{int(peak_epoch)}",
        xy=(peak_epoch, peak_value),
        xytext=offset,
        textcoords="offset points",
        color=colors[-i],
        fontsize=7,
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.2", edgecolor=colors[-i], facecolor="white", alpha=0.8
        ),
        arrowprops=dict(arrowstyle="->", color=colors[-i]),
    )
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Value")
axes[0].set_title("Loss")
axes[0].grid(True)
axes[0].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title="Loss")

# Plot 2: Training Metrics with Peak Values
offsets = [(20, 20), (60, -30), (10, -40), (-50, 25), (-20, -30), (10, -40), (50, -60)]
for i, metric in enumerate(train_metrics):
    (line,) = axes[1].plot(
        df_200["epoch"], df_200[metric], label=metric, color=colors[i]
    )
    metric_handles.append(line)
    metric_labels.append(metric.replace("train_", ""))
    peak_epoch, peak_value = get_peak_info(metric, df_200)
    axes[1].scatter(
        peak_epoch,
        peak_value,
        color=colors[i],
        s=20,
        marker="o",
    )
    offset = offsets[i % len(offsets)]
    offset = (-7, offset[1]) if i == 2 else offset
    axes[1].annotate(
        f"{peak_value:.6f}\nE:{int(peak_epoch)}",
        xy=(peak_epoch, peak_value),
        xytext=offset,
        textcoords="offset points",
        color=colors[i],
        fontsize=7,
        ha="left" if offset[0] >= 0 else "right",  # Căn chỉnh văn bản dựa trên offset
        bbox=dict(
            boxstyle="round,pad=0.2", edgecolor=colors[i], facecolor="white", alpha=0.8
        ),
        arrowprops=dict(arrowstyle="->", color=colors[i]),
    )
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Value")
axes[1].set_title("Training Metrics")
axes[1].grid(True)

# Plot 3: Validation Metrics with Peak Values
for i, metric in enumerate(val_metrics):
    (line,) = axes[2].plot(
        df_200["epoch"], df_200[metric], label=metric, color=colors[i]
    )
    if metric not in metric_labels:
        metric_handles.append(line)
        metric_labels.append(metric)
    peak_epoch, peak_value = get_peak_info(metric, df_200)
    axes[2].scatter(
        peak_epoch,
        peak_value,
        color=colors[i],
        s=20,
        marker="o",
    )
    offset = offsets[i % len(offsets)] if i % 3 == 0 else (10, 10)
    axes[2].annotate(
        f"{peak_value:.6f}\nE:{int(peak_epoch)}",
        xy=(peak_epoch, peak_value),
        xytext=offset,
        textcoords="offset points",
        color=colors[i],
        fontsize=7,
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.2", edgecolor=colors[i], facecolor="white", alpha=0.8
        ),
        arrowprops=dict(arrowstyle="->", color=colors[i]),
    )
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Value")
axes[2].set_title("Validation Metrics")
axes[2].grid(True)

# Add a single shared legend for Plots 2 and 3 (Metrics)
fig.legend(
    metric_handles,
    metric_labels,
    fontsize=8,
    bbox_to_anchor=(0.98, 0.5),
    loc="center right",
    title="Metrics",
)
axes[3].text(
    0.5,
    0.5,
    "Experiment Details:\n"
    "Dataset: Kvasir-SEG and CVC-ClinicDB\n"
    "Training Duration: 200 Epochs\n"
    "Model: LGPS\n"
    "Date: 2025-06-05",
    fontsize=20,
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.5", edgecolor="black", facecolor="lightgray", alpha=0.8
    ),
)
axes[3].set_axis_off()  # Keep axes off but allow text to be displayed
# Hide the last (empty) subplot
axes[3].axis("off")
# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])
# Show the combined figure
plt.show()

# Print peak details for each metric
print("Peak values for Training Metrics:")
for metric in train_metrics:
    peak_epoch, peak_value = get_peak_info(metric, df_200)
    print(f"{metric}: {peak_value:.6f} at epoch {int(peak_epoch)}")

print("\nPeak values for Validation Metrics:")
for metric in val_metrics:
    peak_epoch, peak_value = get_peak_info(metric, df_200)
    print(f"{metric}: {peak_value:.6f} at epoch {int(peak_epoch)}")

print("\nPeak values for Loss Metrics:")
for metric in train_loss:
    peak_epoch, peak_value = get_peak_info(metric, df_200)
    print(f"{metric}: {peak_value:.6f} at epoch {int(peak_epoch)}")
