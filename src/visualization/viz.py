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
        
    def plot_pd_vs_interest(self, data):
        plt.figure(figsize=(7, 5))
        plt.scatter(data['predicted_pd'], data['interest_rate'])
        plt.xlabel("Predicted PD")
        plt.ylabel("Interest Rate")
        plt.title("PD vs Interest Rate")
        plt.show()
        
    def plot_expected_loss_distribution(self, data):
        plt.figure(figsize=(7, 5))
        plt.hist(data['expected_loss'], bins=50)
        plt.xlabel("Expected Loss")
        plt.title("Distribution of Expected Loss")
        plt.show()
        
    def plot_pd_by_bucket(self, data):
        data.groupby('risk_bucket')['predicted_pd'].mean().plot(kind='bar')

        plt.title("Average PD by Risk Bucket")
        plt.xlabel("Risk Bucket")
        plt.ylabel("Average PD")
        plt.show()
        
    def plot_expected_loss_by_bucket(self, data):
        data.groupby('risk_bucket')['expected_loss'].mean().plot(kind='bar')

        plt.title("Expected Loss by Risk Bucket")
        plt.show()

    def plot_all(self, results, data):
        self.plot_roc_curve(results['y_test'], results['y_prob'])
        self.plot_confusion_matrix(results['y_test'], results['y_pred'])
        self.plot_pd_vs_interest(data)
        self.plot_expected_loss_distribution(data)
        self.plot_pd_by_bucket(data)
        self.plot_expected_loss_by_bucket(data)