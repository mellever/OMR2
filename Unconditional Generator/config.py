"""
Global variables for training configuration
"""

# GPU Setup if applicable
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Path for saving figures
FIG_PATH = "/home/melle/OneDrive/Master/Year2/OrientationMathematicalResearch/Project2/experiments/figures/"

# Timeseries
N_LAGS = 6 #Number of timesteps
DATA_DIM = 1 #Dimension of the timeseries
TIME = 0.5 #Final time (T in experiments)

# Brownian motion
SAMPLES_BM = 10000
DRIFT_BM = 0.1
STD_BM = 0.2 

# Geometric Brownian motion
SAMPLES_GBM = 100000 #Number of samples 
DRIFT_GBM = 1 #Mu
STD_GBM = 0.5 #Sigma
INIT_GBM = 100.0 #X0

# AR process
SAMPLES_AR = 50000
PHI = -0.1
STD_AR = 1.0

# Hyperparameters training
LEARNING_RATE = 1e-2 #CHATGPT: The learning rate determines the size of the steps taken during the optimization process to update the model's parameters (like weights). It controls how quickly or slowly a model learns from the data during training.
GRADIENT_STEPS = 2500 #Number of epochs
BATCH_SIZE = 1500 #CHATGPT: The batch size determines the number of training samples processed before the model's internal parameters (weights) are updated.
N = 40 #Dimension of random signature

# R-SIG-W1
RESERVOIR_DIM_METRIC = N #Not sure what this one is, I believe just for evaluation at the end. 

# SIG-W1 --> Not relevant for now
TRUNCATION_DEPTH = 4
NORMALISE_SIG = True

# NeuralSDE
INPUT_DIM_NSDE = 32 #This is lowercase m in the paper I believe
BROWNIAN_DIM = 1 #No mention in the paper I believe
RESERVOIR_DIM_GEN = N #Dimension of reservoir, in our case just N
ACTIVATION_ID = "Sigmoid" #Not sure what the impact of this on the results is

# LSTM --> Not relevant for now
INPUT_DIM_LSTM = 5
HIDDEN_DIM_LSTM = 64
NUM_LAYERS_LSTM = 2

# Random matrices and biases

B1, B2 = (torch.randn(RESERVOIR_DIM_GEN, RESERVOIR_DIM_GEN, device = DEVICE),
                        torch.randn(BROWNIAN_DIM, RESERVOIR_DIM_GEN, RESERVOIR_DIM_GEN, device = DEVICE))

lambda1, lambda2 = (torch.randn(RESERVOIR_DIM_GEN, 1, device = DEVICE),
                             torch.randn(BROWNIAN_DIM, RESERVOIR_DIM_GEN, 1, device = DEVICE))


# Data
DATA_ID = "GBM"

# Generator
GENERATOR_ID = "NeuralSDE"

# Discriminator
DISCRIMINATOR_ID = "RSigW1"

# Switch Trainable Variance Parameter
TRAINABLE_VARIANCE = True

# Switch Same Random Matrices for Generator and RSig-W1
SAME_MATRICES = False

# Switch for time (in)homogeneous readouts
TIME_HOMOGENEOUS_READOUT = True
