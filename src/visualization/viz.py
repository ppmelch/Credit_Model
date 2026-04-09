import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, roc_curve, confusion_matrix as sk_confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_theme(style="whitegrid", palette="Greys_r")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100


class Visualization:
    """
    Visualization utilities for evaluating and analyzing credit risk models.

    Provides methods for plotting model performance metrics, distributions,
    and relationships between key variables such as PD, expected loss, and interest rate.
    """

    def plot_roc_curve(self, y_test, y_prob, name="Test Set"):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.

        Parameters
        ----------
        y_test : array-like
            True binary labels.

        y_prob : array-like
            Predicted probabilities for the positive class.

        name : str, optional
            Label for the dataset (e.g., "Train Set", "Test Set").
        """
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#2c3e50")
        plt.plot([0, 1], [0, 1], linestyle='--', color="#b60000")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Plot a confusion matrix with labeled cells.

        Parameters
        ----------
        y_test : array-like
            True labels.

        y_pred : array-like
            Predicted class labels.
        """
        cm = sk_confusion_matrix(y_test, y_pred)

        labels = np.array([["TN", "FP"], ["FN", "TP"]])
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

    def plot_scatter(self, data, hue, x, y):
        """
        Plot a scatter plot for two variables with optional hue grouping.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing variables.

        hue : str
            Column name used for color grouping.

        x : str
            Column name for x-axis.

        y : str
            Column name for y-axis.
        """
        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=["#2c3e50", "#e74c3c"],
            marker="^",
            s=70,
            alpha=0.8
        )

        plt.xlabel(x.replace('_', ' ').title())
        plt.ylabel(y.replace('_', ' ').title())
        plt.title(
            f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}")
        plt.legend(title=hue)
        plt.grid(alpha=0.2)
        plt.show()

    def plot_bar(self, data, x, y):
        """
        Plot a bar chart of average values grouped by a categorical variable.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing variables.

        x : str
            Grouping variable.

        y : str
            Numerical variable to average.
        """
        data.groupby(x)[y].mean().plot(kind='bar')

        plt.title(
            f"Average {y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}")
        plt.xlabel(x.replace('_', ' ').title())
        plt.ylabel(f"Average {y.replace('_', ' ').title()}")
        plt.xticks(rotation=0)
        plt.grid(alpha=0.2)
        plt.show()

    def plot_distribution(self, x, dataset_name, var_name):
        """
        Plot the distribution (KDE) of a numerical variable.

        Parameters
        ----------
        x : array-like
            Data values.

        dataset_name : str
            Dataset label.

        var_name : str
            Variable name for labeling.
        """
        plt.figure(figsize=(10, 5))

        sns.kdeplot(
            x=x,
            fill=True,
            alpha=0.3,
            linewidth=2
        )

        plt.title(f"{dataset_name} {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Density")
        plt.grid(alpha=0.2)
        plt.show()

    def plot_boxplot(self, data, x, y, hue=None, order=None):
        """
        Plot a boxplot for a numerical variable grouped by categories.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing variables.

        x : str
            Categorical variable.

        y : str
            Numerical variable.

        hue : str, optional
            Additional grouping variable.

        order : list, optional
            Order of categories for x-axis.
        """
        plt.figure(figsize=(10, 5))

        ax = sns.boxplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            order=order,
            palette=["#4b4b4b", "#e74c3c"],
            showfliers=False,
            linewidth=1.2
        )

        ax.set_title(
            f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}")
        ax.set_xlabel(x.replace('_', ' ').title())
        ax.set_ylabel(y.replace('_', ' ').title())

        plt.grid(axis='y', alpha=0.2)
        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_all(self, results, data):
        """
        Generate a full set of visualizations for model evaluation and analysis.

        Parameters
        ----------
        results : dict
            Dictionary containing model outputs (predictions, probabilities, labels).

        data : pd.DataFrame
            Dataset with engineered features and risk metrics.
        """

        # == ROC Curves ==
        self.plot_roc_curve(results['y_test'],
                            results['y_prob'], name="Test Set")
        self.plot_roc_curve(
            results['y_train'],
            results['y_train_prob'], name="Train Set")

        # == Confusion Matrix ==
        self.plot_confusion_matrix(results['y_test'], results['y_pred'])

        # == Scatter Plots ==
        self.plot_scatter(data, hue='loan_status',
                          x='predicted_pd', y='interest_rate_model')

        # == Distributions ==
        self.plot_distribution(
            data['loan_amount'], dataset_name="Distribution of", var_name="Loan Amount")
        self.plot_distribution(
            data['expected_loss'], dataset_name="Distribution of", var_name="Expected Loss")
        self.plot_distribution(
            data['interest_rate_model'], dataset_name="Distribution of", var_name="Interest Rate")
        self.plot_distribution(
            results['y_prob'], dataset_name="Test Set - Distribution of", var_name="PD")
        self.plot_distribution(
            results['y_pred'], dataset_name="Train Set - Distribution of", var_name="PD")

        # == Bar Plots ==
        self.plot_bar(data, x='risk_bucket', y='predicted_pd')
        self.plot_bar(data, x='risk_bucket', y='expected_loss')

        # == Box Plots ==
        self.plot_boxplot(data, x='risk_bucket', y='predicted_pd')
        self.plot_boxplot(data, x='risk_bucket', y='expected_loss')
        self.plot_boxplot(data, x='risk_bucket', y='interest_rate_model')
