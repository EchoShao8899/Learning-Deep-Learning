import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.conv.nets import ConvLayers, DeconvLayers
from models.fc.nets import MLP
from models.utils import modules
from models.utils import loss_functions as lf
from models.utils.rotateHelper import rot_img
import pdb


class mainModel(nn.Module):
    """
    feature_extractor(CNN) -> classifier (MLP)
                  -> generator (VAE)
    """

    def __init__(self, image_size, image_channels, classes,
                 lamda_pl=1., lamda_rcl=1., lamda_vl=1.,  # loss weight in joint training
                 neg_item_per_batch=128,
                 recon_loss="BCE"):
        super(mainModel, self).__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes

        # for encoder
        self.convE = ConvLayers(image_channels)
        self.flatten = modules.Flatten()
        self.fcE = nn.Linear(self.convE.out_feature_dim, 1024)
        self.z_dim = 20
        self.fcE_mean = nn.Linear(1024, self.z_dim)
        self.fcE_logvar = nn.Linear(1024, self.z_dim)

        # for decoder
        self.fromZ = nn.Linear(self.z_dim, 1024)
        self.convD = DeconvLayers(image_channels)
        self.fcD = nn.Linear(1024, self.convD.in_feature_dim)
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels)

        self.classifier = MLP(self.convE.out_feature_dim, classes)

        self.optimizer = None  # needs to be set before training starts

        self.lambda_pl = lamda_pl
        self.lambda_rcl = lamda_rcl
        self.lambda_vl = lamda_vl

        self.neg_item_per_batch = neg_item_per_batch

        self.device = None  # needs to be set before using the model

        self.recon_loss = recon_loss

    # --------- FROWARD FUNCTIONS ---------#
    def encode(self, x):
        """
        pass input through feed-forward connections to get [image_features]
        and [z_mean], [z_logvar] and [hE] for variational autoEncoder
        """
        # Forward-pass through conv-layers
        hidden_x = self.convE(x)
        # [batch_size] x [128] x [image_size] x [image_size] tensor
        image_features = self.flatten(hidden_x)
        # [batch_size] x [128 * image_size * image_size] tensor

        # Forward-pass through fc-layers
        hE = F.elu(self.fcE(image_features))

        # Get parameters for reparametrization
        z_mean = self.fcE_mean(hE)
        z_logvar = self.fcE_logvar(hE)

        return z_mean, z_logvar, hE, hidden_x

    def classify(self, x):
        """
        For input [x] (image or extracted "internal“ image features),
        return predicted scores (<2D tensor> [batch_size] * [classes])
        """
        result = self.classifier(x)
        return result

    def reparameterize(self, mu, logvar):
        """
        Perform "reparametrization trick" to make these stochastic variables differentiable.
        """
        # sigma = 0.5 * exp(log(sigma^2)) = 0.5 * exp(log(var))
        std = 0.5 * torch.exp(logvar).cpu()
        # N(mu, std^2) = N(0,1) * std + mu
        z = torch.randn(std.size()).cpu() * std + mu.cpu()
        return z.to(self.device)

    def decode(self, z):
        """
        Decode latent variable activations.
                INPUT:  - [z]            <2D-tensor>; latent variables to be decoded
                OUTPUT: - [image_recon]  <4D-tensor>
        """
        hD = F.elu(self.fromZ(z))
        image_features = F.elu(self.fcD(hD))
        image_recon = self.convD(image_features.view(-1, 128, 7, 7))

        return image_recon

    def forward(self, x, generator_work=True, classifier_work=True):
        """
        Forward function to propagate [x] through the encoder, reparametrization and decoder.

                Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]
                Output - [image_recon] if generator_work=True,
                         [prediction] if classifier_work=True

        """
        if generator_work and classifier_work:
            mu, logvar, hE, hidden_x = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            prediction = self.classifier(hidden_x)
            return x_recon, prediction
        if generator_work and not classifier_work:
            mu, logvar, hE, hidden_x = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon
        if not generator_work and classifier_work:
            _, _, _, hidden_x = self.encode(x)
            prediction = self.classifier(hidden_x)
            return prediction

    # ------------------LOSS FUNCTIONS--------------------------#
    def calculate_recon_loss(self, x, x_recon, average=False):
        """
        Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]
        """

        batch_size = x.size(0)
        if self.recon_loss == "MSE":
            reconL = -lf.log_Normal_standard(x=x, mean=x_recon, average=average, dim=-1)
        elif self.recon_loss == "BCE":
            reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1),
                                            reduction='none')
            reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)

        return reconL

    def calculate_variat_loss(self, mu, logvar):
        """
        Calculate variational loss for each element in the batch.

        INPUT:  - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]
        """
        # ---- see Appendix B from: Kingma & Welling (2014) Auto-Encoding Variational Bayes, ICLR ----#
        variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return variatL

    def loss_function(self, x, y, x_recon, y_hat, mu, logvar):
        """
        Calculate and return various losses that could be used for training and/or evaluating the model.

                INPUT:  - [x]           <4D-tensor> original image
                        - [y]           <1D-tensor> with target-classes
                        - [x_recon]     <4D-tensor> reconstructed image in same shape as [x]
                        - [y_hat]       <2D-tensor> with predicted "logits" for each class
                        - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                        - [logvar]       <2D-tensor> with estimated log(SD^2) of [z]

                OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                        - [variatL]      variational (KL-divergence) loss "indicating how close distribution [z] is to prior"
                        - [predL]        prediction loss indicating how well targets [y] are predicted
        """
        # reconstruction loss

        # rot 90 degrees
        x_rotate = torch.rot90(x, 1, [2, 3])

        # randomly rotate (just for a TEST)
        # x_rotate = rot_img(x, 2 * torch.rand(1)[0] * np.pi)
        # x_rotate = x_rotate.to(self.device)

        x_recon_resize = x_recon.contiguous().view(-1, self.image_channels * self.image_size * self.image_size)
        x_rotate_resize = x_rotate.contiguous().view(-1, self.image_channels * self.image_size * self.image_size)
        reconL = self.calculate_recon_loss(x=x_rotate_resize, average=True, x_recon=x_recon_resize)
        reconL = lf.weighted_average(reconL, weights=None, dim=0)

        # pdb.set_trace()
        # reconL = F.binary_cross_entropy(x_recon_resize, x_rotate_resize) # , size_average=False)

        # variational loss (KL-divergence)
        variatL = self.calculate_variat_loss(mu=mu, logvar=logvar)
        variatL = lf.weighted_average(variatL, weights=None, dim=0)
        variatL /= (self.image_channels * self.image_size ** 2)

        # prediction loss
        predL = F.cross_entropy(y_hat, y, reduction='none')
        predL = lf.weighted_average(predL, weights=None, dim=0)

        return reconL, variatL, predL

    # ------------------TRAINING FUNCTIONS----------------------#
    def train_a_batch(self, x, y, epoch_num, batch_num, joint_train=True):
        """
        Train model for one batch ([x], [y])
        if joint_train=True, train the autoEncoder and the classifier together
        else, train the classifier alone with input data and data generated from the autoEncoder
        """
        # Set model to training-mode
        if joint_train:
            self.train()
            for p in self.convE.parameters():
                p.requires_grad = True
            for p in self.fcE.parameters():
                p.requires_grad = True
            for p in self.fcE_mean.parameters():
                p.requires_grad = True
            for p in self.fcE_logvar.parameters():
                p.requires_grad = True
            for p in self.fromZ.parameters():
                p.requires_grad = True
            for p in self.fcD.parameters():
                p.requires_grad = True
            for p in self.convD.parameters():
                p.requires_grad = True
        else:
            self.eval()
            self.classifier.train()  # only update the classifier part, freeze the parameters of the other part
            for p in self.convE.parameters():
                p.requires_grad = False
            for p in self.fcE.parameters():
                p.requires_grad = False
            for p in self.fcE_mean.parameters():
                p.requires_grad = False
            for p in self.fcE_logvar.parameters():
                p.requires_grad = False
            for p in self.fromZ.parameters():
                p.requires_grad = False
            for p in self.fcD.parameters():
                p.requires_grad = False
            for p in self.convD.parameters():
                p.requires_grad = False

        # Reset optimizer
        self.optimizer.zero_grad()

        # Run the model
        if joint_train:
            mu, logvar, hE, hidden_x = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            prediction = self.classifier(hidden_x)
            reconL, variatL, predL = self.loss_function(x, y, x_recon, prediction, mu, logvar)
            # pdb.set_trace()
            # print(reconL.shape)
            # print(variatL.shape)
            # print(predL.shape)
            loss = self.lambda_rcl * reconL + self.lambda_vl * variatL + self.lambda_pl * predL

            loss.backward()
            self.optimizer.step()

            if batch_num % 100 == 0:
                print(
                    'Epoch {}, Batch index {}, loss = {:.6f}, reconL = {:.6f}, variatL = {:.6f}, predL = {:.6f}'.format(
                        epoch_num, batch_num, loss.item(), reconL, variatL, predL))

        else:
            _, _, _, hidden_x = self.encode(x)
            prediction = self.classifier(hidden_x)
            loss_input = F.cross_entropy(prediction, y, reduction='none')
            loss_input = lf.weighted_average(loss_input, weights=None, dim=0)


            # random generate image as negative data
            scalar = 1
            random_z = torch.randn(self.neg_item_per_batch, self.z_dim).to(self.device) * scalar
            x_generate = self.decode(random_z)
            y_generate = torch.ones(self.neg_item_per_batch).to(self.device) * 5  # view negative data as class 5
            y_generate = y_generate.long()
            _, _, _, x_generate_feature = self.encode(x_generate)
            prediction_generate = self.classify(x_generate_feature)
            # print(x_generate.shape)
            # print(y_generate.shape)
            # print(prediction_generate.shape)
            loss_generate = F.cross_entropy(prediction_generate, y_generate, reduction='none')
            loss_generate = lf.weighted_average(loss_generate, weights=None, dim=0)

            '''
            # use rotated data as negative data
            x_rotate = torch.rot90(x, 1, [2, 3])
            y_generate = torch.ones(x.shape[0]).to(self.device) * 5  # view negative data as class 5
            y_generate = y_generate.long()
            _, _, _, x_rotate_feature = self.encode(x_rotate)
            prediction_neg = self.classify(x_rotate_feature)
            loss_generate = F.cross_entropy(prediction_neg, y_generate, reduction='none')
            loss_generate = lf.weighted_average(loss_generate, weights=None, dim=0)
            '''

            '''
            # data -> encode -> decode -> negative data
            x_recon = self.forward(x, generator_work=True, classifier_work=False)
            y_generate = torch.ones(x.shape[0]).to(self.device) * 5  # view negative data as class 5
            y_generate = y_generate.long()
            _, _, _, x_recon_feature = self.encode(x_recon)
            prediction_neg = self.classify(x_recon_feature)
            loss_generate = F.cross_entropy(prediction_neg, y_generate, reduction='none')
            loss_generate = lf.weighted_average(loss_generate, weights=None, dim=0)
            '''

            loss = 5 * loss_generate + loss_input  # 5 is a hyper parameter (indicate the weights of the loss)

            loss.backward()
            self.optimizer.step()

            if batch_num % 100 == 0:
                print('Epoch {}, Batch index {}, loss = {:.6f}, loss_neg = {:.6f}, loss_input = {:.6f}'.format(
                    epoch_num, batch_num, loss.item(), loss_generate.item(), loss_input.item()))

        return loss.item()
