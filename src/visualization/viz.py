import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix as sk_confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


class Visualization:
    """
    Class for visualizing results and analyses in a consistent format.
    """

    def plot_roc_curve(self, y_test, y_prob):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#2c3e50")
        plt.plot([0, 1], [0, 1], linestyle='--', color="#b60000")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = sk_confusion_matrix(y_test, y_pred)

        labels = np.array([
            ["TN", "FP"],
            ["FN", "TP"]
        ])

        annotated = np.empty_like(cm).astype(str)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotated[i, j] = f"{labels[i, j]}\n{cm[i, j]}"

        cmap = LinearSegmentedColormap.from_list(
            "custom",
            ["#dbdbdb", "#2c3e50", "#e74c3c"]
        )

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=annotated, fmt="", cmap=cmap, cbar=False)

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    def plot_all(self, results):
        self.plot_roc_curve(results['y_test'], results['y_prob'])
        self.plot_confusion_matrix(results['y_test'], results['y_pred'])