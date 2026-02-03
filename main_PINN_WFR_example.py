# A baseline example of wind field reconstruction using lidar measurement and physics-informed neural network
# Feng Guo @ Shanghai Jiao Tong University  3 Feb. 2026
# This method is presented in the Paper:
# On Wind Directions Estimated by Nacelle Lidar UnderDifferent Reconstruction Methods, Wind Energy
# https://onlinelibrary.wiley.com/doi/epdf/10.1002/we.70098
# Please cite this paper if you find this code helpful for your research.

# Note: This code is intended for demonstration purposes only.
# It has been tested only for the non-yawed case and does not account for sequential lidar scanning.

import math
import sys
import json
import os

import numpy as np
import tensorflow_probability as tfp
import scipy.io
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from logger import Logger
from pyDOE import lhs
import matplotlib.pyplot as plt
import math
from NS2D_InformedNN import NS2D_InformedNN_LOS
import tensorflow as tf


def plot_lidar_Trajectory(lidar,index,it):
    plt.figure(index)
    plt.scatter(lidar["Mea_x"][:,:,it], lidar["Mea_y"][:,:,it], c="g", alpha=0.5)

    for i in range(lidar["Azimuth"].shape[0]):
             plt.plot(np.array([lidar["x_lidar"], lidar["Mea_x"][i,-1,it]]), np.array([lidar["y_lidar"], lidar["Mea_y"][i,-1,it]]), c="r", alpha=0.5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')

def plot_Turb_Grid(Turbulence,index):
    plt.figure(index)
    plt.scatter(Turbulence["X_mesh"], Turbulence["Y_mesh"], c="b", alpha=0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')


def error():
    u_pred, _ = pinn.predict(X_star)
    return 0

def plot_windfield(X_star, u_star, ub, lb,index):
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='rainbow')
    plt.colorbar()
    #plt.clim(0.0, 0.3)
    plt.xlim(lb[0], ub[0])
    plt.ylim(lb[1], ub[1])



# ---------------------------------------Main code start here

np.random.seed(36)              # random seed initialization
tf.random.set_seed(36)          # random seed initialization


# Load Data, the index sequence of U_LES and V_LES is xyt
# The full data set can be ontained from:
# https://data.dtu.dk/articles/dataset/EllipSys3D_large_eddy_simulation_data_of_single_wind_turbine_wakes_in_neutral_atmospheric_conditions/10259936
# Only the 60s data in the waked condition is analyzed in this example. Ambient TI=12.8%
Turb1 = scipy.io.loadmat('DTU_LES_data_wake_60s_part1.mat')
Turb2 = scipy.io.loadmat('DTU_LES_data_wake_60s_part2.mat')


# Generate mesh
Turb = {};
Turb["X_mesh"] , Turb["Y_mesh"] = np.meshgrid(Turb1["x_LES"], Turb1["y_LES"], indexing='ij')
Turb["X_vec"] = Turb["X_mesh"].flatten()
Turb["Y_vec"] = Turb["Y_mesh"].flatten()
Turb["x_LES"] = Turb1["x_LES"]
Turb["y_LES"] = Turb1["y_LES"]
Turb["U_LES"] = np.concatenate((Turb1["U_LES1"], Turb2["U_LES2"]),axis=2)
Turb["V_LES"] = np.concatenate((Turb1["V_LES1"], Turb2["V_LES2"]),axis=2)
Turb["t"]     = np.concatenate((Turb1["t1"], Turb2["t2"]),axis=1)


# Define a lidar
lidar            = {}
lidar["x_lidar"]     = 1000   # lidar or turbine location x [m]
lidar["y_lidar"]     = 0     # lidar or turbine location y [m]
lidar["nRangeGate"]  = 15    # number of range gate        [-]
lidar["minRangeGate"]= 100    # the nearest range gate      [m]
lidar["maxRangeGate"]= 750   # the farest range gate       [m], should not be too large to aviod exceeding the wind field
lidar["RangeGate"]   = np.linspace(lidar["minRangeGate"], lidar["maxRangeGate"], lidar["nRangeGate"], dtype=float)   # range gate vector
lidar["nAzimuth"]    = 15     # number of azimuth angle     [-]
lidar["minAzimuth"]  = -18   # the minimal azimuth angle in lidar coordinate system [deg]
lidar["maxAzimuth"]  = 18    # the maximal azimuth angle in lidar coordinate system [deg]
lidar["Yaw"]         = 180   # the lidar yaw angle in inertial coordinate system [deg] 180 degree aligns with the initial wind direction
lidar["Azimuth"]     = np.linspace(lidar["minAzimuth"]+lidar["Yaw"], lidar["maxAzimuth"]+lidar["Yaw"], lidar["nAzimuth"], dtype=float) # vector of azimuth angles [deg]
lidar["ProbeVolume"] = np.array([-15, 0, 15])
lidar["ProbeWeights"]= np.array([0.25, 0.5, 0.25])


# ---------------------------Simulate lidar measurement

lidar["Mea_x"]             = np.zeros((lidar["Azimuth"].shape[0],lidar["RangeGate"].shape[0],Turb["t"].shape[1]))
lidar["Mea_y"]             = np.zeros((lidar["Azimuth"].shape[0],lidar["RangeGate"].shape[0],Turb["t"].shape[1]))
lidar["unit_x"]            = np.zeros((lidar["Azimuth"].shape[0],lidar["RangeGate"].shape[0],Turb["t"].shape[1]))
lidar["unit_y"]            = np.zeros((lidar["Azimuth"].shape[0],lidar["RangeGate"].shape[0],Turb["t"].shape[1]))
lidar["Mea_x_Probe"]       = np.zeros((lidar["Azimuth"].shape[0],lidar["RangeGate"].shape[0],lidar["ProbeVolume"].shape[0],Turb["t"].shape[1]))
lidar["Mea_y_Probe"]       = np.zeros((lidar["Azimuth"].shape[0],lidar["RangeGate"].shape[0],lidar["ProbeVolume"].shape[0],Turb["t"].shape[1]))

lidar["LOS"]             = np.zeros((lidar["Azimuth"].shape[0],lidar["RangeGate"].shape[0],Turb["t"].shape[1]))
u_temp                   = np.zeros((lidar["ProbeVolume"].shape[0]))
v_temp                   = np.zeros((lidar["ProbeVolume"].shape[0]))
LOS_temp                 = np.zeros((lidar["ProbeVolume"].shape[0]))


it_analyze = Turb["t"].shape[1]
#loop over range gate, beam drection and time to get all the LOS measurements, this may be improved be better efficiency
print('Simulating lidar LOS measurement...')
for it in range(it_analyze):
    U_interpolator = NearestNDInterpolator(list(zip(Turb["X_vec"], Turb["Y_vec"])),Turb["U_LES"][:, :, it].flatten()[:, None])
    V_interpolator = NearestNDInterpolator(list(zip(Turb["X_vec"], Turb["Y_vec"])),Turb["V_LES"][:, :, it].flatten()[:, None])
    for ibeam in range(lidar["Azimuth"].shape[0]):
        for igate in range(lidar["RangeGate"].shape[0]):
            lidar["Mea_x"][ibeam, igate, it] = (lidar["RangeGate"][igate]) * np.cos((lidar["Azimuth"][ibeam]) / 180 * np.pi) + \
                                           lidar["x_lidar"]
            lidar["Mea_y"][ibeam, igate, it] = (lidar["RangeGate"][igate]) * np.sin((lidar["Azimuth"][ibeam]) / 180 * np.pi) + \
                                           lidar["y_lidar"]

            lidar["unit_x"][ibeam, igate, it]  = (lidar["Mea_x"][ibeam, igate, it]-lidar["x_lidar"])/np.sqrt(np.power(lidar["Mea_x"][ibeam, igate, it]-lidar["x_lidar"], 2)+np.power(lidar["Mea_y"][ibeam, igate, it]-lidar["y_lidar"],2))  #lidar unit vector
            lidar["unit_y"][ibeam, igate, it]  = (lidar["Mea_y"][ibeam, igate, it]-lidar["y_lidar"])/np.sqrt(np.power(lidar["Mea_x"][ibeam, igate, it]-lidar["x_lidar"], 2)+np.power(lidar["Mea_y"][ibeam, igate, it]-lidar["y_lidar"],2))  #lidar unit vector

            for iProbe in range(lidar["ProbeVolume"].shape[0]):
                lidar["Mea_x_Probe"][ibeam, igate, iProbe, it] = (lidar["RangeGate"][igate] + lidar["ProbeVolume"][
                    iProbe]) * np.cos((lidar["Azimuth"][ibeam]) / 180 * np.pi) + lidar["x_lidar"]
                lidar["Mea_y_Probe"][ibeam, igate, iProbe, it] = (lidar["RangeGate"][igate] + lidar["ProbeVolume"][
                    iProbe]) * np.sin((lidar["Azimuth"][ibeam]) / 180 * np.pi) + lidar["y_lidar"]

                u_temp[iProbe]    = U_interpolator(lidar["Mea_x_Probe"][ibeam,igate,iProbe,it], lidar["Mea_y_Probe"][ibeam,igate,iProbe,it])
                v_temp[iProbe]    = V_interpolator(lidar["Mea_x_Probe"][ibeam,igate,iProbe,it], lidar["Mea_y_Probe"][ibeam,igate,iProbe,it])
                LOS_temp[iProbe]  = u_temp[iProbe]*lidar["unit_x"][ibeam, igate, it]+v_temp[iProbe]*lidar["unit_y"][ibeam, igate, it]
            TempLOS               = np.multiply(LOS_temp,lidar["ProbeWeights"])
            lidar["LOS"][ibeam, igate, it] = np.sum(TempLOS)



# show the lidar trajectory at a certain time
it  = 0
plot_lidar_Trajectory(lidar, 1, it)
plot_Turb_Grid(Turb, 1)
plt.title('lidar trajectory and turbulence field grid')
plt.show()

## Configurations for the Neural Network
NNConfig = {}

# DeepNN topology (3-sized input [x y t], 8 hidden layer of 32-width, 2-sized output [u,v]
NNConfig["layers"]    = [3, 32, 32, 32, 32, 32, 32, 32, 32, 2]

# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
NNConfig["SGD_epochs"] = 2000   # number of epochs
NNConfig["SGD_lr"]     = 0.05   # learning rate
NNConfig["SGD_b1"]     = 0.9    # Adam optimizer parameter https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
NNConfig["SGD_eps"]    = 1e-7   # A small constant for numerical stability in Adam optimizer parameter https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam

NNConfig["LBGFS_epochs"] = 5000   # number of epochs 2000
NNConfig["LBGFS_lr"]     = 0.3    # learning rate 0.3 good for 100s data 0.5 0.3
NNConfig["LBGFS_ncorr"]  = 50     #50
NNConfig["LBGFS_logfrequency"] = 10 # frequency of writting log message


logger    = Logger(NNConfig)

# We have to normalize the training data
LOS_Normalizefactor = np.max([np.abs(np.amax(lidar["LOS"])), np.abs(np.amin(lidar["LOS"]))])
xyt_train = np.zeros((lidar["RangeGate"].shape[0]*lidar["Azimuth"].shape[0]*(it_analyze),3))

# LOS_train; first dimension: LOS speed. second dimension: first unit vector element # third dimension: second unit vector element
LOS_train = np.zeros((lidar["RangeGate"].shape[0]*lidar["Azimuth"].shape[0]*(it_analyze),3))

# Prepare training data. We need to inject the projection unit vectors which should always be defined using the lidar local coordinate system
counter   = 0
for ibeam in range(lidar["Azimuth"].shape[0]):
    for igate in range(lidar["RangeGate"].shape[0]):
        for it in range(it_analyze):     #Turb["t"].shape[0]
            xyt_train[counter,:] = np.array([lidar["Mea_x"][ibeam][igate][it],lidar["Mea_y"][ibeam][igate][it],Turb["t"][0][it]]).T
            LOS_train[counter,:] = np.array([lidar["LOS"][ibeam, igate, it]/LOS_Normalizefactor,lidar["unit_x"][ibeam, igate, it],lidar["unit_y"][ibeam, igate, it]]).T
            counter = counter +1

ub  = np.array([xyt_train[:,0].max(axis=0), xyt_train[:,1].max(axis=0), xyt_train[:,2].max(axis=0)])
lb  = np.array([xyt_train[:,0].min(axis=0), xyt_train[:,1].min(axis=0), xyt_train[:,2].min(axis=0)])

N_f = 10000  # Number of points for initializing the NN
X_f = lb + (ub-lb)*lhs(3, N_f)

# Defining the error function for the logger and training
logger.set_error_fn(error)
pinn = NS2D_InformedNN_LOS(NNConfig, logger, X_f, ub, lb)

# train the NN
pinn.train(xyt_train, LOS_train)

#   Compare model predicted with the reference LES Data
snap = 10

xindex    =  np.where(np.logical_and(np.greater_equal(Turb['x_LES'][0,:],lb[0]), np.greater_equal(ub[0],Turb['x_LES'][0,:])))
yindex    =  np.where(np.logical_and(np.greater_equal(Turb['y_LES'][0,:],lb[1]), np.greater_equal(ub[1],Turb['y_LES'][0,:])))
x_pred_mesh, y_pred_mesh, t_pred_mesh = np.meshgrid(Turb["x_LES"][0,:], Turb["y_LES"][0,:], Turb["t"][0,snap], indexing='ij')

pred_in   = np.concatenate((x_pred_mesh.flatten()[:,None],y_pred_mesh.flatten()[:,None], t_pred_mesh.flatten()[:,None]), axis=1)

u_star    = Turb["U_LES"][:,:,snap].flatten()[:,None]
v_star    = Turb["V_LES"][:,:,snap].flatten()[:,None]

# Prediction
uv_pred   = pinn.predict(pred_in)
u_pred    = uv_pred[:, 0]*LOS_Normalizefactor  # multiply with the normalization factor to recover data
v_pred    = uv_pred[:, 1]*LOS_Normalizefactor
xy_pred   = np.concatenate((x_pred_mesh.flatten()[:,None],y_pred_mesh.flatten()[:,None]), axis=1)


# u component (axial wind) prediction is much better than the v component, as u is more known by the lidar than the v components
plot_windfield(xy_pred, u_pred, ub, lb, 3)
plt.title('u prediction')
plt.show()

plot_windfield(xy_pred, u_star, ub, lb, 4)
plt.title('u LES exact')
plt.show()

plot_windfield(xy_pred, v_pred, ub, lb, 5)
plt.title('v prediction')
plt.show()
plot_windfield(xy_pred, v_star, ub, lb, 6)
plt.title('v LES exact')
plt.show()

success = True