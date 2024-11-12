# -*- coding: utf-8 -*-
"""
Created on Tue  8 08:14:01 2024

@author: leovo
"""


import matplotlib.pyplot as plt

from ._generic_model import GenericModel

###############################################################################

class sklearnManifold(GenericModel):
    """
    Interface for sklearn manifold learning models.
            
    Parameters
    ----------
    manifold_model: any manifold model that uses the sklearn interface. 
        For example:
            * sklearn.manifold.MDS,
            * sklearn.manifold.Isomap,
            * sklearn.manifold.TSNE,
            * sklearn.manifold.LocallyLinearEmbedding,
            * etc....
    """

    ###########################################################################

    def __init__(self, manifold_model):
        self.has_Y = False  # Manifold models generally don't use a target variable
        self.manifold_model = manifold_model

        self.name = self.manifold_model.__class__.__name__

    ###########################################################################

    def train_core(self):
        """
        Fits the manifold model using the training data.
        """
        # Manifold models often apply dimensionality reduction directly to the input data
        self.transformed_data = self.manifold_model.fit_transform(self.X_train.values)

    ###########################################################################

    def map_from_X(self, X):
        """
        Applies the transformation to a new dataset. Note that some manifold
        models, like TSNE, may not have a direct `transform` method.
        """
        if hasattr(self.manifold_model, 'transform'):
            return self.manifold_model.transform(X)
        else:
            raise NotImplementedError("This manifold model does not support transformation on new data.")

    ###########################################################################

    def set_hyperparameters(self, params_dict):
        """
        Sets the hyperparameters for the manifold model.
        """
        for key, value in params_dict.items():
            setattr(self.manifold_model, key, value)

    ###########################################################################

    def plot_embedding(self):
        """
        Plots the 2D or 3D embedding resulting from the manifold model.
        """
        if self.transformed_data.shape[1] == 2:
            plt.scatter(self.transformed_data[:, 0], self.transformed_data[:, 1], s=50, cmap='viridis')
            plt.title(f"{self.name} 2D Embedding")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        elif self.transformed_data.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.transformed_data[:, 0], self.transformed_data[:, 1], self.transformed_data[:, 2], s=50, cmap='viridis')
            ax.set_title(f"{self.name} 3D Embedding")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
        else:
            print("Embedding dimensionality is not 2D or 3D; custom plotting is required.")
        
        plt.show()