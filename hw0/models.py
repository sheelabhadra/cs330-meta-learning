"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=32,
        layer_sizes=[96, 64],
        sparse=False,
        embedding_sharing=True,
    ):

        super().__init__()

        self.embedding_dim = embedding_dim

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        def mlp(
            input_dim,
            output_dim,
            hidden_layers,
            activation=nn.ReLU,
            output_activation=nn.Identity,
        ):
            sizes = [input_dim] + hidden_layers + [output_dim]
            layers = []
            for i in range(len(sizes) - 1):
                act = activation if i < len(sizes) - 2 else output_activation
                layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
            return nn.Sequential(*layers)

        self.num_users = num_users
        self.num_items = num_items
        # U matrix
        self.user_embedding = ScaledEmbedding(
            self.num_users,
            self.embedding_dim,
        )
        self.user_bias = ZeroEmbedding(self.num_users, 1)
        # Q matrix
        self.item_embedding = ScaledEmbedding(
            self.num_items,
            self.embedding_dim,
        )
        self.item_bias = ZeroEmbedding(self.num_items, 1)

        self.embedding_sharing = embedding_sharing

        # Rating network
        self.rating_net = mlp(3 * self.embedding_dim, 1, layer_sizes)

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of
            shape (batch,). This corresponds to p_ij in the
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape
            (batch,). This corresponds to r_ij in the
            assignment.
        """

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        u = self.user_embedding(user_ids)
        q = self.item_embedding(item_ids)
        a = self.user_bias(user_ids)
        b = self.item_bias(item_ids)

        # p_ij
        predictions = torch.matmul(u, q.T)[:, 0].reshape(-1, 1) + a + b
        # r_ij
        if self.embedding_sharing:
            # r_ij
            x = torch.cat([u, q, torch.mul(u, q)], dim=-1)
        else:
            x1 = u.detach()
            x2 = q.detach()
            x = torch.cat([x1, x2, torch.mul(x1, x2)], dim=-1)
        score = self.rating_net(x)
        # ********************************************************
        # ********************************************************
        # ********************************************************
        return predictions, score
