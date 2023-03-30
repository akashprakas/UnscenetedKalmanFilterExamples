from math import tan, sin, cos, sqrt, atan2
from sympy import Symbol, symbols, Matrix, sin, cos
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Saver
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from sympy import init_printing
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.random import randn
pd.set_option('display.max_columns', 50)


init_printing(use_latex=True)


numstates = 5  # States
dt = 1.0/50.0  # Sample Rate of the Measurements is 50Hz
dtGPS = 1.0/10.0  # Sample Rate of GPS is 10Hz


def move(x, yawrateCurrent):

    if np.abs(yawrateCurrent) < 0.0001:  # Driving straight
        x[0] = x[0] + x[3]*dt * np.cos(x[2])
        x[1] = x[1] + x[3]*dt * np.sin(x[2])
        x[2] = x[2]
        x[3] = x[3]
        x[4] = 0.0000001  # avoid numerical issues in Jacobians

    else:  # otherwise
        x[0] = x[0] + (x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2]))
        x[1] = x[1] + (x[3]/x[4]) * (-np.cos(x[4]*dt+x[2]) + np.cos(x[2]))
        x[2] = (x[2] + x[4]*dt + np.pi) % (2.0*np.pi) - np.pi
        x[3] = x[3]
        x[4] = x[4]

    return x


def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def residual_x(a, b):
    y = a - b
    # we have the angle at the second position so we need to normalize it to [-pi to pi]
    y[2] = normalize_angle(y[2])
    return y


def Hx(x):
    hx = np.array([float(x[0]), float(x[1]), float(x[3]), float(x[4])])
    return hx


# we need to take the mean of the angle which is not easy
def state_mean(sigmas, Wm):
    z = np.zeros(5)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(sigmas[:, 3], Wm))
    x[4] = np.sum(np.dot(sigmas[:, 4], Wm))

    return x


P = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
sGPS = 0.5*8.8*dt**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sCourse = 0.1*dt  # assume 0.1rad/s as maximum turn rate for the vehicle
sVelocity = 8.8*dt  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sYaw = 1.0*dt  # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle

Q = np.diag([sGPS**2, sGPS**2, sCourse**2, sVelocity**2, sYaw**2])

datafile = '2014-03-26-000-Data.csv'
df = pd.read_csv(datafile)


altitude = df['altitude'].values
latitude = df['latitude'].values
longitude = df['longitude'].values
course = df['course'].values
speed = df['speed'].values
yawrate = df['yawrate'].values
# A course of 0° means the Car is traveling north bound
# and 90° means it is traveling east bound.
# In the Calculation following, East is Zero and North is 90°
# We need an offset.
course = (-course+90.0)


varGPS = 6.0  # Standard Deviation of GPS Measurement
varspeed = 1.0  # Variance of the speed measurement
varyaw = 0.1  # Variance of the yawrate measurement
R = np.matrix([[varGPS**2, 0.0, 0.0, 0.0],
               [0.0, varGPS**2, 0.0, 0.0],
               [0.0, 0.0, varspeed**2, 0.0],
               [0.0, 0.0, 0.0, varyaw**2]])


RadiusEarth = 6378388.0  # m
# this line tells us about the meters per degree
arc = 2.0*np.pi*(RadiusEarth+altitude)/360.0  # m/°

dx = arc * np.cos(latitude*np.pi/180.0) * \
    np.hstack((0.0, np.diff(longitude)))  # in m
dy = arc * np.hstack((0.0, np.diff(latitude)))  # in m

mx = np.cumsum(dx)
my = np.cumsum(dy)

ds = np.sqrt(dx**2+dy**2)

GPS = (ds != 0.0).astype('bool')  # GPS Trigger for Kalman Filter


measurements = np.vstack((mx, my, speed/3.6, yawrate/180.0*np.pi))
# Lenth of the measurement
m = measurements.shape[1]


# In other words, the value in km/h divide by 3.6 to get a value in m/s.
x = np.array([mx[0], my[0], course[0]/180.0*np.pi,
              speed[0]/3.6+0.001, yawrate[0]/180.0*np.pi]).T

x_initial = x
# beta = 2 for gaussians, kappa = 3 - n where n is x dim
# points = MerweScaledSigmaPoints(n=5, alpha=.001, beta=2, kappa=-2,
#                                 subtract=residual_x)


points = JulierSigmaPoints(n=5, kappa=-2)

ukf = UKF(dim_x=5, dim_z=4, fx=move, hx=Hx, dt=dt, points=points, x_mean_fn=state_mean,
          residual_x=residual_x)


ukf.x = x_initial
ukf.P = P
ukf.R = R


for filterstep in range(500):
    print(filterstep)

    yawRateCurrent = yawrate[filterstep]

    # while taking measurements there we will get GPS only in some cycles
    # as of now I am just using the predicted states as measurements when there is not gps
    # the predict is using move behind the scenes and we should pass all the arguments move requries
    # x will be passed automatically  and the remaining we need to supply.
    ukf.P = (ukf.P + ukf.P.transpose())/2
    ukf.predict(yawRateCurrent)

    if GPS[filterstep]:
        z = measurements[:, filterstep].reshape(-1,)
    else:
        z = measurements[:, filterstep].reshape(-1,)
        z[0] = ukf.x[0]
        z[1] = ukf.x[1]

    ukf.update(z)
