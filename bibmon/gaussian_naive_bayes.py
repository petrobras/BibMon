import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from _probabilistic_genereic_model import ProbabilisticGenericModel


class GNB(ProbabilisticGenericModel):
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None

    def train_core(self):
        """
        Treina o modelo Gaussian Naive Bayes estimando a média, variância
        e probabilidades a priori para cada classe.
        """
        X = self.X_train.values
        y = self.Y_train.values.flatten()

        # Identificar classes únicas
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        # Inicializar parâmetros
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)

        # Calcular médias, variâncias e priors para cada classe
        for idx, cls in enumerate(self.classes):
            X_c = X[y == cls]
            self.means[idx, :] = np.mean(X_c, axis=0)
            # Adicionar pequeno valor às variâncias para evitar divisão por zero
            self.variances[idx, :] = np.var(X_c, axis=0) + 1e-9
            self.class_priors[idx] = X_c.shape[0] / X.shape[0]

    def predict_proba(self, X):
        """
        Calcula as probabilidades para cada classe para as amostras de teste fornecidas.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        # Calcular log-probabilidades para cada classe
        for idx, cls in enumerate(self.classes):
            mean = self.means[idx]
            var = self.variances[idx]
            prior = np.log(self.class_priors[idx])

            # Calcular log-verossimilhança
            log_likelihood = -0.5 * np.sum(np.log(2. * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_probs[:, idx] = prior + log_likelihood

        # Normalizar log-probabilidades usando o truque log-sum-exp
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)

        return probs

    def predict(self, X):
        """
        Prediz a classe para cada amostra baseada na probabilidade máxima.
        """
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes[class_indices]

    def plot_confusion_matrix(self, X_test, Y_test):
        """
        Plota a matriz de confusão para os dados de teste fornecidos.
        """
        y_pred = self.predict(X_test)
        cm = confusion_matrix(Y_test, y_pred, labels=self.classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.show()

    def plot_roc_curve(self, X_test, Y_test):
        """
        Plota a curva ROC para os dados de teste fornecidos.
        """
        if len(self.classes) != 2:
            print("A curva ROC é aplicável apenas para problemas de classificação binária.")
            return

        y_score = self.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, y_score, pos_label=self.classes[1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='Curva ROC (área = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self, X_test, Y_test):
        """
        Plota a curva de Precisão-Recall para os dados de teste fornecidos.
        """
        if len(self.classes) != 2:
            print("A curva de Precisão-Recall é aplicável apenas para problemas de classificação binária.")
            return

        y_score = self.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(Y_test, y_score, pos_label=self.classes[1])
        average_precision = average_precision_score(Y_test, y_score, pos_label=self.classes[1])

        plt.figure()
        plt.step(recall, precision, where='post', color='b', alpha=0.2,
                 label='Precisão-Recall (AP = %0.2f)' % average_precision)
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precisão')
        plt.title('Curva de Precisão-Recall')
        plt.legend(loc="upper right")
        plt.show()
