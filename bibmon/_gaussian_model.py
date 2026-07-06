import numpy as np

from _probabilistic_genereic_model import ProbabilisticGenericModel

class GaussianModel(ProbabilisticGenericModel):
    def __init__(self):
        self.mean = None
        self.std = None

    def train_core(self):
        """
        Ajusta uma distribuição normal aos dados de treinamento.
        """
        self.mean = self.X_train.mean()
        self.std = self.X_train.std()

    def predict_proba(self, X):
        """
        Calcula a probabilidade de cada observação em X sob a distribuição ajustada.

        Parâmetros
        ----------
        X: numpy.array
            Dados de entrada para previsão.

        Retorna
        -------
        probabilities: numpy.array
            Probabilidades calculadas.
        """
        from scipy.stats import norm
        probabilities = norm.pdf(X, loc=self.mean, scale=self.std)
        return probabilities

    def pre_train(self, X_train, Y_train=None, *args, **kwargs):
        """
        Pré-processamento específico para o modelo Gaussiano.
        """
        self.X_train = pd.DataFrame(X_train)
        # Possíveis etapas de pré-processamento

    def pre_test(self, X_test, Y_test=None, *args, **kwargs):
        """
        Pré-processamento específico para o modelo Gaussiano.
        """
        self.X_test = pd.DataFrame(X_test)
        # Possíveis etapas de pré-processamento

    def plot_histogram_with_pdf(self, feature_name=None, bins=30):
        """
        Plota o histograma dos dados de treinamento com a PDF ajustada sobreposta.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        import seaborn as sns

        if feature_name is None:
            # Se nenhuma feature for especificada, usar a primeira
            feature_name = self.X_train.columns[0]

        data = self.X_train[feature_name]

        plt.figure(figsize=(10,6))
        sns.histplot(data, bins=bins, kde=False, stat='density', label='Dados')

        x_axis = np.linspace(data.min(), data.max(), 100)
        pdf = norm.pdf(x_axis, self.mean[feature_name], self.std[feature_name])
        plt.plot(x_axis, pdf, color='red', label='PDF Ajustada')
        plt.title(f'Histograma e PDF Ajustada para {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Densidade')
        plt.legend()
        plt.show()

    def plot_qq(self, feature_name=None):
        """
        Plota o gráfico Q-Q para verificar a normalidade dos dados.
        """
        import scipy.stats as stats
        import matplotlib.pyplot as plt

        if feature_name is None:
            feature_name = self.X_train.columns[0]

        data = self.X_train[feature_name]

        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"Gráfico Q-Q para {feature_name}")
        plt.show()