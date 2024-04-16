
import torchmetrics
import pytorch_lightning as pl
from .utils.transformers import *
from tqdm.auto import tqdm

from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassAccuracy

class TrainingModule(pl.LightningModule):
    def __init__(self, config, model, num_classes, normalisation):
        super().__init__()
        self.config = config
        self.model = model
        self.normalisation = normalisation
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_classwise_acc = ClasswiseWrapper(MulticlassAccuracy(num_classes=10, average=None))
        self.train_criterion = torch.nn.CrossEntropyLoss()
        self.val_criterion = torch.nn.CrossEntropyLoss()

        self.learning_rate = config.opt.params.lr
        self.decay_factor = config.opt.params.weight_decay

    def configure_optimizers(self):
        return init_optims_from_config(self.config, self.model)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(self.normalisation(x))
        loss = self.train_criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(self.normalisation(x))
        loss = self.val_criterion(y_hat, y)
        self.val_acc(y_hat, y)

        # log loss
        self.log('val_loss', loss, on_epoch=True, on_step=False, logger=True, sync_dist=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, on_step=False, logger=True, sync_dist=True, prog_bar=True)
        return loss

    ################################# Added for DL #########################################
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(self.normalisation(x))
        loss = self.val_criterion(y_hat, y)
        self.test_acc.update(y_hat, y)
        self.test_classwise_acc.update(y_hat,y)


    def on_validation_epoch_end(self):
        print()

    def on_test_end(self):


        log = {
            'test_acc': self.test_acc.compute(),
            'test_class_wise': self.test_classwise_acc.compute()
        }

        self.test_acc.reset()
        return log

    #################################  Until here  #########################################

    def manual_test(self, test_loader=None, name='nunya'):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """

        print(f'name: {name}')

        accelerator = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

        self.model.to(accelerator)
        device = next(self.model.parameters()).device
        self.test_acc.to(device)
        self.test_classwise_acc.to(device)

        test_loader_wrapped = tqdm(enumerate(test_loader), desc=f'Testing... {name}', total=len(test_loader))

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in test_loader_wrapped:
                data, target = data.to(device), target.to(device)

                data = self.normalisation(data)
                output = self.model(data)

                self.test_acc.update(output, target)

                ################################# Added for DL #########################################
                if name == 'clean':
                    self.test_classwise_acc.update(output, target)

                #################################  Until here  #########################################

                test_loader_wrapped.set_postfix(test_acc=f'{self.test_acc.compute().item():.3f}')

        log = {
            'val_acc': self.test_acc.compute(),
        }

        ################################# Added for DL #########################################
        if name == 'clean':
            log['classwise_acc'] = self.test_classwise_acc.compute()
            self.test_classwise_acc.reset()
        #################################  Until here  #########################################

        self.test_acc.reset()


        return log

def init_optims_from_config(config, model):
    params = [p for p in model.parameters() if p.requires_grad]

    if hasattr(torch.optim, config.opt.type):
        opt = getattr(torch.optim, config.opt.type)(
            params,
            **config.opt.params
        )
    else:
        raise NotImplementedError(f'Unknown optimizer: {config.opt.type}')

    lr_scheduler = []
    for s, p, i in zip(config.lr_scheduler.type, config.lr_scheduler.params, config.lr_scheduler.interval):
        if hasattr(torch.optim.lr_scheduler, s):
            lr_scheduler.append(
                {
                    'scheduler': getattr(torch.optim.lr_scheduler, s)(opt, **p),
                    'interval': i,
                    'frequency': 1
                }
            )
        else:
            print("Error")
    return [opt], lr_scheduler