import torch
from tqdm import tqdm
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.utils import Container
from datetime import datetime
from optuna.exceptions import TrialPruned
from os import mkdir
from optuna.trial import Trial as OptunaTrial


# TODO: Remove model and optimizer from class inputs
class Engine():
    """
    Engine class contains all functions needed to train a model.
    This one is made specifivally for a VAE (KL divergence in loss function) training
    on an airfoil dataset (or any dataset where two lists of values must match really).
    Note that it uses the custom Container class frequently, because I like it.
    """
    def __init__(self, optimizer: optim.Optimizer = None) -> object:
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer

    @staticmethod
    def defaultLossFn(x_, x):
        """Default loss function for (airfoil) y coordinates.
        calculates the relative error: mean [(y-y')/y]^2
        This should help with scaling (?)

        All y coordinates are offset by +1 to avoid division by 0

        Args:
            x_ (torch.tensor): reconstructed input
            x (torch.tensor): original input

        Returns:
            torch.tensor: reconstruction loss
        """
        # return torch.mean(torch.abs((torch.add(x, 1) - torch.add(x_, 1)) / torch.add(x, 1)))
        return torch.mean(((torch.add(x, 1) - torch.add(x_, 1)) / torch.add(x, 1)).pow(2) * 100)

    def trainModel(self,
                   model: nn.Module,
                   dataContainer: Container,
                #    HyperParam: Container,
                   loss_fn: nn.functional = defaultLossFn,
                   save: bool = True,
                   earlyStopIter: int = float('nan'),
                   trial: OptunaTrial = None
                   ):
        """Main model  training loop. Works for both a single training, or in an Optuna hyperparameter study.
        If the training is part of an Optuna study, the current trial should be passed

        Args:
            dataContainer (Container): Container with train, test, and validation dataLoaders.
            # // HyperParam (Container): Container with all hyper parameters.
            loss_fn (torch.nn.Module, optional): The loss function to be used.
                └── Defaults to defaultLossFn.
            save (bool, optional): determines whether the model is saved whenever a new best case is found.
                └── Defaults to True.
            earlyStopIter (int, optional): Max nr of iterations without improvement before the training loop is stopped.
                └── Defaults to 50.
            trial (optuna.trial.Trial): current trial, shows if optuna is used
                └── Defaults to None.

        Raises:
            ValueError: When mode is different from "Train" or "Eval"

        Returns:
            Losses (Container): Container object with all the losses from the operation of each epoch
            bestCase (Container): Container object with best model and parameters found
        """
        model.to(self.DEVICE)

        Losses = Container(
            train=Container(
                avgLoss=[],
                avgRecon=[],
                avgKLdiv=[]
            ),
            test=Container(
                avgLoss=[],
                avgRecon=[],
                avgKLdiv=[]
            ),
            bestLoss=float('nan')
        )
        bestCase = Container(loss=float('nan'))
        earlyStopCounter = 0

        """
        Set up the save location
        """
        if save:
            if trial:                                                       # if this training is part of an optuna study
                bestCase.trial = trial
                saveDir = "./optunaOutput"                                  # change save location to optuna folder to save checkpoints
                if trial.number != 0:                                       # If there have been trials before (otherwise the trainLosses file won't exist yet, or be the one from a previous run)
                    Losses = torch.load(f"{saveDir}/trainLosses.ctnr")      # to append training losses to those of previous optuna trials for overview of the entire study
            else:
                start_time = datetime.today().strftime('%Y_%m_%d_%Hh%M')
                saveDir = f"./savedModels/{start_time}_{'_'.join(str(i) for i in model.hyperParam.H_DIM)}_z{model.hyperParam.Z_DIM}_b{model.hyperParam.BETA:.1e}"  # single model training can have net structure in the name for easier identification
                mkdir(saveDir)                                              # make the directory for saving
            self.saveDir = saveDir

        for epoch in tqdm(range(model.hyperParam.EPOCHS), unit="epoch", total=model.hyperParam.EPOCHS, leave=False, colour='green', ncols=100):
            avgLoss_train, avgRecon_train, avgKLdiv_train = self.batch_oper("Train", model, dataContainer.trainLoader, loss_fn=loss_fn)
            avgLoss_test, avgRecon_test, avgKLdiv_test = self.batch_oper("Eval", model, dataContainer.testLoader, loss_fn=loss_fn)

            # * save everything to container object
            Losses.train.avgLoss.append(avgLoss_train)
            Losses.train.avgRecon.append(avgRecon_train)
            Losses.train.avgKLdiv.append(avgKLdiv_train)
            Losses.test.avgLoss.append(avgLoss_test)
            Losses.test.avgRecon.append(avgRecon_test)
            Losses.test.avgKLdiv.append(avgKLdiv_test)

            if trial:                               # when using optuna, see whether the trial is worth continuing
                trial.report(avgLoss_test, epoch)   # report the results
                if trial.should_prune():            # if the trial is not worth continuing
                    print(f"trial {trial.number} pruned with loss = {avgLoss_test:.3e}")
                    raise TrialPruned()             # quit trial early

            if avgLoss_test < bestCase.loss or np.isnan(bestCase.loss):     # replace if new result is better than last best, or if last best is nan
                Losses.bestLoss = avgLoss_test
                bestCase.loss = avgLoss_test
                bestCase.reconloss = avgRecon_test
                bestCase.KLdiv = avgKLdiv_test
                bestCase.epoch = epoch
                bestCase.params = model.state_dict()
                bestCase.model = model

                # tqdm.write(f"Epoch {epoch}: \ttrainLoss = {avgLoss_train:.3e} \ttestLoss = {avgLoss_test:.3e} * New best - iter since last best: {earlyStopCounter}")
                earlyStopCounter = 0

                if save:
                    if trial and trial.number != 0:
                        lastBest = torch.load(f"{saveDir}/bestCase.ctnr")
                        if bestCase.loss < lastBest.loss:   # check if current trial best is better than previous trials
                            torch.save(model.state_dict(), f"{saveDir}/ParamDict.pth")
                            torch.save(model, f"{saveDir}/model.pth")
                    else:
                        torch.save(model.state_dict(), f"{saveDir}/ParamDict.pth")
                        torch.save(bestCase, f"{saveDir}/bestCase.ctnr")

            else:
                earlyStopCounter += 1
                # tqdm.write(f"Epoch {epoch}: \ttrainLoss = {avgLoss_train:.3e} \ttestLoss = {avgLoss_test:.3e}")

            if earlyStopCounter == earlyStopIter:
                tqdm.write(f"Early stop after {earlyStopCounter} iterations without improvement.")
                break

            if avgLoss_test != avgLoss_test:  # NaN detection
                break

        """
        this saves the best case again after the trianing is completed this step
        is probably redundent, but I feel safer having it for debugging later
        on. For now I also don't know whether it is best to save using the lines
        below, or the intermediate saved above.
        """
        if save:
            if trial and trial.number != 0:
                lastBest = torch.load("optunaOutput/bestCase.ctnr")
                if bestCase.loss < lastBest.loss:   # check if current trial best is better than previous trials
                    torch.save(bestCase, f"{saveDir}/bestCase.ctnr")
                    torch.save(Losses, f"{saveDir}/trainLosses.ctnr")
            else:
                torch.save(bestCase, f"{saveDir}/bestCase2.ctnr")
                torch.save(Losses, f"{saveDir}/trainLosses.ctnr")
        return Losses, bestCase

    def batch_oper(self,
                   MODE: str,
                   model: nn.Module,
                   dataLoader: DataLoader,
                   loss_fn: torch.nn.Module = defaultLossFn
                   ):
        """Perform an operation using batches, either a training operation or evaluation / testing.

        Args:
            MODE (str): defines the type of operation: "Train" or "Eval"
            dataLoader (DataLoader): DataLoader with the dataset for the respective mode
            beta (float): KL-Divergence coefficient
            loss_fn (torch.nn.Module, optional): The loss function to be used.
                └── Defaults to defaultLossFn.

        Raises:
            ValueError: When mode is different from "Train" or "Eval"

        Returns:
            # // Losses (Container): Container object with all the losses from the operation
            avgLoss (float): batch average total loss
            avgRecon (float): batch average reconstruction loss
            avgKLdiv (float): batch average KL-divergence multiplied with factor Beta
        """

        if MODE == "Train":
            model.train()              # model in training mode
            torch.set_grad_enabled(True)    # enable gradient computation for training
        elif MODE == "Eval":
            model.eval()               # model in evaluation mode
            torch.set_grad_enabled(False)   # no need to compute gradients during testing, save some time and effort
        else:
            raise ValueError("Mode should be 'Train' or 'Eval'.")

        avgLoss = 0
        avgRecon = 0
        avgKLdiv = 0

        # for batch, (x, _) in tqdm(enumerate(dataLoader), total=len(dataLoader), leave=False, unit="batch", ncols=500,):
        for batch, (x, _) in enumerate(dataLoader):
            x = x.to(self.DEVICE)                                                               # data to device
            x_, z, mean, logvar = model(x)
            reconLoss = loss_fn(x_, x)                                          # compute reconstruct loss
            KLdiv = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())   # compute KL divergence loss
            loss = reconLoss + KLdiv * model.hyperParam.BETA                    # compute total loss

            if MODE == "Train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avgLoss += loss.detach().numpy()                 # average loss of all batches
            avgRecon += reconLoss.detach().numpy()           # average reconstruction loss of all batches
            avgKLdiv += KLdiv.detach().numpy()               # average KL divergence of all batches

        avgLoss /= len(dataLoader)
        avgRecon /= len(dataLoader)
        avgKLdiv /= len(dataLoader)
        return avgLoss, avgRecon, avgKLdiv  # * beta

    def plot_losses(self, Losses, saveDir=None):
        """Plot the evolution of the average losses

        Args:
            Losses (Container): Container with all the training and test losses
            save (bool, optional): Option to save the plot.
                └── Defaults to False
        """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        fig.suptitle("Average training and test losses")
        l1,=ax1.semilogy(range(len(Losses.test.avgLoss)), Losses.test.avgLoss, 'b-', label="Test loss")
        l2,=ax1.semilogy(range(len(Losses.train.avgLoss)), Losses.train.avgLoss, 'b--', label="Train loss")
        l3,=ax2.semilogy(range(len(Losses.test.avgKLdiv)), Losses.test.avgKLdiv, 'g-', label="Test KLdivergence")
        l4,=ax2.semilogy(range(len(Losses.train.avgKLdiv)), Losses.train.avgKLdiv, 'g--', label="Train KLdivergence")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color="b")
        ax2.set_ylabel(r"$\beta\cdot$KLdivergence", color="g")
        fig.legend([l1, l2, l3, l4],["Avg. batch test loss", "Avg. batch train loss", "Avg. test KLdivergence", "Avg. train KLdivergence"], loc="lower center", ncol=2)
        if saveDir:
            plt.savefig(f"{saveDir}/TestLossPlot.png")
        plt.show(block=False)
        return fig, [ax1, ax2]


if __name__ == "__main__":
    from src.data.utils import Container
    from src.networks.betaVarAutoEncoderAltForm import betaVarAutoEncoder
    from src.data.airfoilDataset import Naca4Dataset

    dataset = Naca4Dataset()
    HyperParam = Container(
        H_DIM=[128, 64, 32],
        Z_DIM=4,
        BATCH_SIZE=64,
        LEARN_RATE=3e-4,
        EPOCHS=200,
        BETA=1e-6,
    )
    model = betaVarAutoEncoder(
        input_dim=dataset.input_dim,
        h_dim=HyperParam.H_DIM,
        z_dim=HyperParam.Z_DIM
    )

    # print(str(summary(model, input_size=[HyperParam.BATCH_SIZE, dataset.input_dim])))

    optimizer = optim.Adam(model.parameters(), lr=HyperParam.LEARN_RATE)
    engine = Engine(model, optimizer)
    losses = engine.trainModel(dataset, HyperParam, save=False)
