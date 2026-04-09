import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix as sk_confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

sns.set_theme(style="whitegrid", palette="Greys_r")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

class Visualization:
    """
    Class for visualizing results and analyses in a consistent format.
    """

    def plot_roc_curve(self, y_test, y_prob , name="Test Set"):
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
        

    def plot_scatter(self, data , hue,  x , y):
        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            palette=["#2c3e50", "#e74c3c"],
            hue = hue,
            marker="^",
            s=70,
            alpha=0.8
        )

        plt.xlabel("Predicted PD")
        plt.ylabel("Interest Rate")
        plt.title("PD vs Interest Rate")
        plt.legend(title=hue)
        plt.grid(alpha=0.2)
        plt.show()

        
    def plot_bar(self, data , x , y):
        data.groupby(x)[y].mean().plot(kind='bar')

        plt.title(f"Average {y.replace('_', ' ').title()} by Risk Bucket")
        plt.xlabel("Risk Bucket")
        plt.ylabel(f"Average {y.replace('_', ' ').title()}")
        plt.xticks(rotation=0)
        plt.grid(alpha = 0.2)
        plt.show()
        
        
    def plot_variable_distribution_by_status(self, values, loan_status, var_name="Variable", dataset_name="Train"):
        """
        Visualizes the distribution of a numerical variable segmented by loan status.

        Parameters
        ----------
        values : array-like
            Numerical variable (e.g., income, interest_rate, loan_amount).

        loan_status : array-like
            Binary or categorical target (e.g., 0 = No Default, 1 = Default).

        var_name : str
            Name of the variable (for titles).

        dataset_name : str
            Name of dataset (Train/Test).
        """

        data = pd.DataFrame({
            var_name: values,
            "Loan_Status": loan_status
        })

        status_map = {
            0: "Non-Default",
            1: "Default"
        }

        # -------- GENERAL DISTRIBUTION --------
        plt.figure(figsize=(10, 5))

        sns.kdeplot(
            x=data[var_name],
            fill=True,
            alpha=0.2,
            linewidth=2
        )

        plt.title(f"{dataset_name} Distribution of {var_name}")
        plt.xlabel(var_name)
        plt.ylabel("Density")
        plt.grid(alpha=0.2)
        plt.show()

        # -------- DISTRIBUTION PER STATUS (JUNTAS) --------
        plt.figure(figsize=(10, 5))

        for status_id, status_name in status_map.items():

            sns.kdeplot(
                x=data.loc[data["Loan_Status"] == status_id, var_name],
                fill=True,
                alpha=0.3,
                linewidth=2,
                label=status_name
            )

        plt.title(f"{dataset_name} {var_name} Distribution by Loan Status")
        plt.xlabel(var_name)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.2)

        plt.show()

        # -------- INDIVIDUAL DISTRIBUTIONS --------
        for status_id, status_name in status_map.items():

            plt.figure(figsize=(10, 5))

            sns.kdeplot(
                x=data.loc[data["Loan_Status"] == status_id, var_name],
                fill=True,
                alpha=0.3,
                linewidth=2
            )

            plt.title(f"{dataset_name} {var_name} - {status_name}")
            plt.xlabel(var_name)
            plt.ylabel("Density")
            plt.grid(alpha=0.2)
            plt.show()
        
    def plot_distribution(self , x , dataset_name, var_name):
        
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
        import matplotlib.pyplot as plt
        import seaborn as sns

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

        # Mejor formato de labels
        ax.set_title(f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}", fontsize=12)
        ax.set_xlabel(x.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(y.replace('_', ' ').title(), fontsize=10)

        # Grid más sutil
        plt.grid(axis='y', alpha=0.2)

        # Quitar borde superior/derecho (look más profesional)
        sns.despine()
        plt.tight_layout()
        plt.show()



    def plot_all(self, results, data):
        self.plot_roc_curve(results['y_test'], results['y_prob'] , name="Test Set")
        self.plot_roc_curve(results['y_train'], results['y_pred'] , name="Train Set")
        self.plot_confusion_matrix(results['y_test'], results['y_pred'])
        self.plot_scatter(data ,hue = 'loan_status', x='predicted_pd', y='interest_rate')
        self.plot_distribution(data['loan_amount'], dataset_name="Distribution of", var_name="Loan Amount")
        self.plot_distribution(data['expected_loss'], dataset_name="Distribution of", var_name="Expected Loss")
        self.plot_distribution(data['interest_rate'], dataset_name="Distribution of", var_name="Interest Rate")
        self.plot_distribution(results['y_prob'], dataset_name="Test Set - Distribution of", var_name="PD")
        self.plot_distribution(results['y_pred'], dataset_name="Train Set - Distribution of", var_name="PD")
        self.plot_bar(data, x='risk_bucket', y='predicted_pd')
        self.plot_bar(data, x='risk_bucket', y='expected_loss')
        self.plot_variable_distribution_by_status(data['predicted_pd'], data['decision'], var_name="Predicted PD", dataset_name="Distribution of")
        self.plot_boxplot(data, x='risk_bucket', y='predicted_pd')
        self.plot_boxplot(data, x='risk_bucket', y='expected_loss')
        self.plot_boxplot(data, x='risk_bucket', y='interest_rate')
        