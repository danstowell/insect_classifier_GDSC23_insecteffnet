from glob import glob

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support
from torch_audiomentations import ApplyImpulseResponse

class TrainModule(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer_name, optimizer_hparams, cfg):
        """
        Lighting training module.
        Contains the main logic of the training procedure and automatically coordinates the training loop,
        backward calls and optimizer updates.
        The functions training_steps() and validation_steps() can be customized for tracking of any metric to be
        visualized in Tensorboard or saved as a .csv.


        Parameters
        ----------
        model: Name of the model/CNN to run. Used for creating the model (see function below)
        loss_fn: Pytorch loss function, e.g. nn.CrossEntropyLoss()
        optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
        optimizer_hparams: Hyperparameters for the optimizer, as dictionary.
                           This includes learning rate, weight decay, etc.
        cfg: SimpleNameSpace containing all configurations
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters(ignore=["model", 'loss_fn'])
        # Create model
        self.model = model
        # Create loss module
        self.loss_fn = loss_fn
        self.use_leaf = cfg.use_leaf
        # Data augmentation function to apply a fixed set of impulse responses
        # ir_paths has to be adapted when compute environment is changed
        # Should be a list of .wav files, e.g. from https://www.openair.hosted.york.ac.uk/
        self.impulse = ApplyImpulseResponse(p=cfg.impulse_prob, p_mode="per_example",
                                            sample_rate=cfg.sample_rate, mode="per_example",
                                            compensate_for_propagation_delay=True,
                                            ir_paths=glob("../data/irs/*/*/mono/*"))

    def forward(self, wave):
        # Forward function that is run when visualizing the graph
        return self.model(wave)

    def configure_optimizers(self):
        #params_decay, params_no_decay = (self.parameters(), [])  # This default option skips the weightdecay question, using Lightning's convenience method for the parameters
        params_decay, params_no_decay = get_wd_params(self.model)
        # Choose Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW([{'params': params_decay}, {'params': params_no_decay, 'weight_decay': 0}], **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(  [{'params': params_decay}, {'params': params_no_decay, 'weight_decay': 0}], **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1.0e-9,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader and a dictionary containing 'wave' and 'labels'
        wave, labels = batch['wave'], batch['labels']
        batch_size = wave.shape[0]
        model = self.model

        # Batch has to be reshaped for the impulse response function
        wave = self.impulse(wave.view(batch_size, 1, -1)).view(batch_size, -1)
        preds, labels = model(wave, labels)
        loss = self.loss_fn(preds, labels)

        # Calculate Metrics
        with torch.no_grad():
            acc = (preds.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
            precision, recall, f1, _ = precision_recall_fscore_support(labels.argmax(dim=-1).detach().cpu().numpy(),
                                                                       preds.argmax(dim=-1).detach().cpu().numpy(),
                                                                       average="macro",
                                                                       zero_division=1)

            # Logs the accuracy per epoch to tensorboard (weighted average over batches)
            self.log("train_acc", acc, on_step=False, on_epoch=True)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_precision", precision, on_step=True, on_epoch=True)
            self.log("train_recall", recall, on_step=False, on_epoch=True)
            self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

            if self.use_leaf:
                self.log("leaf_alpha", model.wav2timefreq.compression.alpha.mean(), on_step=False, on_epoch=True, prog_bar=False)
                self.log("leaf_delta", model.wav2timefreq.compression.delta.mean(), on_step=False, on_epoch=True, prog_bar=False)
                self.log("leaf_root",  model.wav2timefreq.compression.root.mean(), on_step=False, on_epoch=True, prog_bar=False)
                self.log("leaf_emaweights", model.wav2timefreq.compression.ema._weights.mean(), on_step=False, on_epoch=True, prog_bar=False)
                self.log("leaf_poolingweights", model.wav2timefreq.pooling.weights.mean(), on_step=False, on_epoch=True, prog_bar=False)
                self.log("leaf_poolingbias", model.wav2timefreq.pooling._bias.mean(), on_step=False, on_epoch=True, prog_bar=False)

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        wave, labels = batch['wave'], batch['labels']
        preds = self.model(wave).argmax(dim=-1)
        acc = (labels.argmax(dim=-1) == preds).float().mean()
        precision, recall, f1, _ = precision_recall_fscore_support(labels.argmax(dim=-1).detach().cpu().numpy(),
                                                                   preds.detach().cpu().numpy(), average="macro",
                                                                   zero_division=1)

        # By default, logs it per epoch (weighted average over batches)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True)

@torch.no_grad()
def get_wd_params(model):
    """Divide the learnable parameters into those which should have weight decay, and those which should not.
    See https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348 for discussion.
    We use this here specifically to exclude LEAF parameters from weight decay.
    """
    wd_params = list()
    no_wd_params = list()
    for p in model.named_parameters():
        if p[0].startswith("wav2timefreq"):
            no_wd_params.append(p[1])
        else:
            wd_params.append(p[1])
    # Only weights of specific layers should undergo weight decay.
    print(f"Parameter groups: {len(wd_params)} will use weight decay, {len(no_wd_params)} will not.")
    #assert len(wd_params) + len(no_wd_params) == len(all_params), "Sanity check failed."
    return wd_params, no_wd_params
