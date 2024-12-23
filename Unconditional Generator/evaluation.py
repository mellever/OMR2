"""
Defines the evaluation environment
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2
from datetime import datetime
from rsigw1 import *
from sigw1 import *
from utils import *


class Evaluation:
    def __init__(self, training, x_train, x_test, scaler, generator_id, discriminator_id, activation_id, data_type, device=DEVICE):
        self.training = training
        self.x_train = x_train
        self.x_test = x_test
        self.scaler = scaler
        self.best_generator = self.training.generator
        self.generator_id = generator_id
        self.discriminator_id = discriminator_id
        self.activation_id = activation_id
        self.num_epochs = self.training.num_grad_steps
        self.learning_rate = self.training.learning_rate
        self.activation = get_activation(ACTIVATION_ID)
        self.data_type = data_type
        self.n_lags = self.training.n_lags
        self.batch_size = self.training.batch_size
        self.device = device
        self.x_fake = self.best_generator(batch_size=self.batch_size, n_lags=self.n_lags).to(self.device)
        self.t = np.linspace(0, TIME, N_LAGS)



        
        if self.discriminator_id == "RSigW1":
            self.expected_rsig_x_train = compute_rsig_td(self.x_train, self.training.A1, self.training.A2,
                                                     self.training.xi1,
                                                     self.training.xi2, RESERVOIR_DIM_METRIC, self.activation).mean(0).to(self.device)
            self.expected_rsig_x_test = compute_rsig_td(self.x_test, self.training.A1, self.training.A2, self.training.xi1,
                                                    self.training.xi2, RESERVOIR_DIM_METRIC, self.activation).mean(0).to(self.device)
            self.expected_rsig_x_fake = compute_rsig_td(self.x_fake, self.training.A1, self.training.A2, self.training.xi1,
                                                    self.training.xi2, RESERVOIR_DIM_METRIC, self.activation).mean(0).to(self.device)

            self.train_error = l2_dist(self.expected_rsig_x_train, self.expected_rsig_x_fake)
            self.test_error = l2_dist(self.expected_rsig_x_test, self.expected_rsig_x_fake)

        if self.discriminator_id == "SigW1":
            self.expected_sig_x_train = compute_exp_sig(self.x_train, TRUNCATION_DEPTH, NORMALISE_SIG).to(self.device)
            self.expected_sig_x_test = compute_exp_sig(self.x_test, TRUNCATION_DEPTH, NORMALISE_SIG).to(self.device)
            self.expected_sig_x_fake = compute_exp_sig(self.x_fake, TRUNCATION_DEPTH, NORMALISE_SIG).to(self.device)

            self.train_error = l2_dist(self.expected_sig_x_train, self.expected_sig_x_fake)
            self.test_error = l2_dist(self.expected_sig_x_test, self.expected_sig_x_fake)

        self.corr_train_error = cov_diff(self.x_train, self.x_fake)
        self.corr_test_error = cov_diff(self.x_test, self.x_fake)

        self.acf_train_error = acf_diff(self.x_train, self.x_fake, lag=self.n_lags // 2)
        self.acf_test_error = acf_diff(self.x_test, self.x_fake, lag=self.n_lags // 2)

        self.x_train_scale_inverse = self.scaler.inverse(x_train)
        self.x_test_scale_inverse = self.scaler.inverse(x_test)
        self.x_fake_scale_inverse = self.scaler.inverse(self.x_fake)

    def print_summary(self):
        try:
          os.makedirs("evaluation/{}".format(self.data_type))
          file = open("evaluation/{}/{}-{}-{}.txt".format(self.data_type, self.generator_id, self.discriminator_id,
                                                          datetime.now().strftime("%d%m%Y-%H%M%S")), 'w+')
          file.write("-------------------------------\nTRAINING SUMMARY\n------------------------------\n")
          file.write("Generator: {}\nDiscriminator: {}\nActivation: {}\nGradient steps: {}\nLearning rate:"
                    " {}\nReservoir dimension: {}\nData dimension: {}\n".format(self.generator_id,
                                                                                self.discriminator_id,
                                                                                self.activation_id, self.num_epochs,
                                                                                self.learning_rate,
                                                                                RESERVOIR_DIM_METRIC,
                                                                                DATA_DIM))
          file.write("-------------------------------\n")
          file.write("Drift BM: {}\nStd BM: {}\nDrift GBM: {}\nStd GBM: {}\nPhi AR: {}\nStd AR: {}\n".format(
              DRIFT_BM, STD_BM, DRIFT_GBM, STD_GBM, PHI, STD_AR))
          file.write("-------------------------------\n")
          file.write("{}:\n".format(self.discriminator_id))
          file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.train_error, self.test_error))
          file.write("-------------------------------\n")
          file.write("Correlation Metric:\n")
          file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.corr_train_error,
                                                                          self.corr_test_error))
          file.write("-------------------------------\n")
          file.write("Autocorrelation Metric:\n")
          file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.acf_train_error,
                                                                          self.acf_test_error))
          if self.data_type == "BM":
              file.write("-------------------------------\n")
              file.write("Results of Normality Test:\n")
              for i in range(1, self.n_lags):
                  file.write("Normaltest #{}: {}\n".format(i, p_val_normaltest(self.x_fake, i) > 0.05))

          file.write("-------------------------------\n")
          sys.stdout = file
          print(self.training.train_losses_history)
          file.write("-------------------------------\n")
          sys.stdout = file
          print(self.training.val_losses_history)

        except FileExistsError:
          file = open("evaluation/{}/{}-{}-{}.txt".format(self.data_type, self.generator_id, self.discriminator_id,
                                                          datetime.now().strftime("%d%m%Y-%H%M%S")), 'w+')
          file.write("-------------------------------\nTRAINING SUMMARY\n------------------------------\n")
          file.write("Generator: {}\nDiscriminator: {}\nActivation: {}\nGradient steps: {}\nLearning rate:"
                    " {}\nReservoir dimension: {}\nData dimension: {}\n".format(self.generator_id,
                                                                                self.discriminator_id,
                                                                                self.activation_id, self.num_epochs,
                                                                                self.learning_rate,
                                                                                RESERVOIR_DIM_METRIC,
                                                                                DATA_DIM))
          file.write("-------------------------------\n")
          file.write("Drift BM: {}\nStd BM: {}\nDrift GBM: {}\nStd GBM: {}\nPhi AR: {}\nStd AR: {}\n".format(
              DRIFT_BM, STD_BM, DRIFT_GBM, STD_GBM, PHI, STD_AR))
          file.write("-------------------------------\n")
          file.write("{}:\n".format(self.discriminator_id))
          file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.train_error, self.test_error))
          file.write("-------------------------------\n")
          file.write("Correlation Metric:\n")
          file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.corr_train_error,
                                                                          self.corr_test_error))
          file.write("-------------------------------\n")
          file.write("Autocorrelation Metric:\n")
          file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.acf_train_error,
                                                                          self.acf_test_error))
          if self.data_type == "BM":
              file.write("-------------------------------\n")
              file.write("Results of Normality Test:\n")
              for i in range(1, self.n_lags):
                  file.write("Normaltest #{}: {}\n".format(i, p_val_normaltest(self.x_fake, i) > 0.05))

          file.write("-------------------------------\n")
          sys.stdout = file
          print(self.training.train_losses_history)
          file.write("-------------------------------\n")
          sys.stdout = file
          print(self.training.val_losses_history)

    def save_best_generator(self):
      try:
        os.makedirs("best_generators/{}".format(self.data_type))
        torch.save(self.best_generator.state_dict(), "best_generators/{}/{}-{}-{}-{}-{}.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
      except FileExistsError:
        torch.save(self.best_generator.state_dict(), "best_generators/{}/{}-{}-{}-{}-{}.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))

    def save_paths(self):
      try:
        os.makedirs("best_generators/{}".format(self.data_type))
        torch.save(self.x_fake_scale_inverse.detach(), "best_generators/{}/{}-{}-{}-{}-{}-fake.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        torch.save(self.x_train_scale_inverse.detach(), "best_generators/{}/{}-{}-{}-{}-{}-train.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        torch.save(self.x_test_scale_inverse.detach(), "best_generators/{}/{}-{}-{}-{}-{}-test.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
      except FileExistsError:
        torch.save(self.x_fake_scale_inverse.detach(), "best_generators/{}/{}-{}-{}-{}-{}-fake.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        torch.save(self.x_train_scale_inverse.detach(), "best_generators/{}/{}-{}-{}-{}-{}-train.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        torch.save(self.x_test_scale_inverse.detach(), "best_generators/{}/{}-{}-{}-{}-{}-test.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))

    def plot_paths(self, num_paths=50):
        #Restore printing
        sys.stdout = sys.__stdout__ 

        fig = plt.figure()
        sns.set_theme()
        for i in range(num_paths):
            if i==0:
              plt.plot(self.t, to_numpy(self.x_train_scale_inverse)[i], color="tab:blue", linewidth=0.7, label="Training")
              plt.plot(self.t, to_numpy(self.x_fake_scale_inverse)[i], color="tab:orange", linewidth=0.7, label="Generated")
            else:
              plt.plot(self.t, to_numpy(self.x_train_scale_inverse)[i], color="tab:blue", linewidth=0.7)
              plt.plot(self.t, to_numpy(self.x_fake_scale_inverse)[i], color="tab:orange", linewidth=0.7)

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("X")
        fig.savefig(FIG_PATH+"paths.pdf")

    def plot_mean(self):
      #Convert data to numpy
      train_data = to_numpy(self.x_train_scale_inverse) #Shape  = (0.2*SAMPLES, N_LAGS, DATA_DIM)
      fake_data = to_numpy(self.x_fake_scale_inverse) #Shape  = (BATCH_SIZE, N_LAGS, DATA_DIM)
      
      #Compute mean
      mean_train = np.mean(train_data, axis=0)[:,0]
      mean_fake = np.mean(fake_data, axis=0)[:,0]

      #Compute 95% confidence intervals
      ci_train = 1.96 * np.std(train_data, axis=0)[:,0] / np.sqrt(len(train_data))
      ci_fake = 1.96 * np.std(fake_data, axis=0)[:,0] / np.sqrt(len(fake_data))

      #Plot the mean and the 95% confidence intervals
      fig = plt.figure()
      plt.plot(self.t, mean_train, label="Training", marker='o', color="tab:blue")
      plt.fill_between(self.t, mean_train - ci_train, mean_train + ci_train, color='blue', alpha=0.2)
      plt.plot(self.t, mean_fake, label="Generated", marker='o', color="tab:orange")
      plt.fill_between(self.t, mean_fake - ci_fake, mean_fake + ci_fake, color='orange', alpha=0.2)
      plt.ylabel("Mean")
      plt.xlabel("Time")
      plt.legend()
      fig.savefig(FIG_PATH+"mean.pdf")


    def plot_var(self):
      #Convert data to numpy
      train_data = to_numpy(self.x_train_scale_inverse) #Shape  = (0.2*SAMPLES, N_LAGS, DATA_DIM)
      fake_data = to_numpy(self.x_fake_scale_inverse) #Shape  = (BATCH_SIZE, N_LAGS, DATA_DIM)

      #Compute the variance
      var_train = np.var(train_data, axis=0)[:,0]
      var_fake = np.var(fake_data, axis=0)[:,0]

      #Determine lengths
      n_train = len(train_data)
      n_fake = len(fake_data)

      # Compute the 95% confidence interval for the variance (using chi-squared distribution)
      ci_train_lower = ((n_train - 1) * var_train) / chi2.ppf(0.975, df=n_train - 1)
      ci_train_upper = ((n_train - 1) * var_train) / chi2.ppf(0.025, df=n_train - 1)
      
      ci_fake_lower = ((n_fake - 1) * var_fake) / chi2.ppf(0.975, df=n_fake - 1)
      ci_fake_upper = ((n_fake - 1) * var_fake) / chi2.ppf(0.025, df=n_fake - 1)

      fig = plt.figure()
      plt.plot(self.t, var_train, label="Training", marker='o', color="tab:blue")
      plt.fill_between(self.t, ci_train_lower, ci_train_upper, color='blue', alpha=0.2)
      plt.plot(self.t, var_fake, label="Generated", marker='o', color="tab:orange")
      plt.fill_between(self.t, ci_fake_lower, ci_fake_upper, color='orange', alpha=0.2)
      plt.ylabel("Variance")
      plt.xlabel("Time")
      plt.legend()
      fig.savefig(FIG_PATH+"var.pdf")
      plt.show()
    

