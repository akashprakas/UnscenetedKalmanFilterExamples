from numpy import array, sqrt
from filterpy.kalman import ExtendedKalmanFilter as EKF
from sympy import Symbol, symbols, Matrix, sin, cos
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
x = np.array([[mx[0], my[0], course[0]/180.0*np.pi,
              speed[0]/3.6+0.001, yawrate[0]/180.0*np.pi]]).T


def normalize(x):
    temp = x % (2.0*np.pi)  # force in range [0, 2 pi)
    if temp > np.pi:  # move to [-pi, pi)
        temp -= 2*np.pi
    return temp


class RoboEKF(EKF):
    def __init__(self, dt, Q, x_initial):
        EKF.__init__(self, 5, 4)
        self.dt = dt
        self.x = x_initial

        vs, psis, dpsis, dts, xs, ys, lats, lons = symbols(
            'v \psi \dot\psi T x y lat lon')
        self.F = Matrix([[xs+(vs/dpsis)*(sin(psis+dpsis*dts)-sin(psis))],
                         [ys+(vs/dpsis)*(-cos(psis+dpsis*dts)+cos(psis))],
                         [psis+dpsis*dts],
                         [vs],
                         [dpsis]])
        self.F_straight = Matrix([[xs + vs*dts*(cos(psis))],
                                  [ys + vs*dts*(sin(psis))],
                                  [psis],
                                  [vs],
                                  [0.0000001]])
        self.state = Matrix([xs, ys, psis, vs, dpsis])
        self.F_J = self.F.jacobian(self.state)
        self.F_j_straight = self.F_straight.jacobian(self.state)
        self.subs = {vs: 0, psis: 0, dpsis: 0, dts: 0, xs: 0, ys: 0}

        self.x_s, self.y_s, self.time = xs, ys, dts
        self.v_s, self._psis, self._dpsis = vs, psis, dpsis
        self.Q = Q

        # initialize subs with the initial values
        self.subs[self.x_s] = self.x[0, 0]
        self.subs[self.y_s] = self.x[1, 0]
        self.subs[self._psis] = self.x[2, 0]
        self.subs[self.v_s] = self.x[3, 0]
        self.subs[self._dpsis] = self.x[4, 0]
        self.subs[self.time] = self.dt

    def predict(self, yawrate):
        # we should put it at first because we need to evaluate he move from the previous step with the the self.subs
        self.subs[self.x_s] = self.x[0, 0]
        self.subs[self.y_s] = self.x[1, 0]
        self.subs[self._psis] = self.x[2, 0]
        self.subs[self.v_s] = self.x[3, 0]
        self.subs[self._dpsis] = self.x[4, 0]

        self.x = self.move(yawrate)
        # print(self.x)
        # if we want to calculate the jacobian at the predicted state we should calculate the jacobian at this point.
        # to do that we should re-substitute s with self.subs

        if np.abs(yawrate) < 0.0001:  # Driving straight
            print("driving straight")
            F = np.array(self.F_j_straight.evalf(subs=self.subs)).astype(float)
            self.calculatedJacobian = F

        else:
            F = np.array(self.F_J.evalf(subs=self.subs)).astype(float)
            self.calculatedJacobian = F

        self.P = F@self.P@F.T + self.Q

    def move(self, yawrate):
        if np.abs(yawrate) < 0.0001:  # Driving straight
            print("driving straight")
            x_t = np.array(self.F_straight.evalf(subs=self.subs)).astype(float)
            x_t[2, 0] = normalize(x_t[2, 0])
        else:
            x_t = np.array(self.F.evalf(subs=self.subs)).astype(float)
            x_t[2, 0] = normalize(x_t[2, 0])
        # print(x_t)
        return x_t


def H_of(x, filterStep):
    if GPS[filterStep]:  # with 10Hz, every 5th step
        print("GPS update")
        JH = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0]])
    else:  # every other step
        JH = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0]])

    return JH


# first argument is always the state eventhough it might be not used
# this is because this is being called internally in the update routine and
# self.x is the first argument
def Hx(x):
    hx = np.array([[float(x[0])],
                   [float(x[1])],
                   [float(x[3])],
                   [float(x[4])]])
    return hx


# z = measurements[:, 0].reshape(-1, 1)
# demo = RoboEKF(dt=0.02, Q=Q, x_initial=x)
# filterstep = 0
# yawRequired = yawrate[0]
# demo.P = P
# demo.R = R
# demo.predict(yawRequired)


# # argument mathcing
# # args will be the input of HJacobian
# # hx_args will be the input of Hx, if there are no arguments then leave it free
# demo.update(z, HJacobian=H_of, Hx=Hx, args=filterstep)


noOfMeasurements = measurements.shape[1]
ekf = RoboEKF(dt=0.02, Q=Q, x_initial=x)
s = Saver(ekf)
ekf.P = P
ekf.R = R

for filterstep in range(500):
    yawRateCurrent = yawrate[filterstep]
    # for prediction yaw rate is required since we
    # need to differentiatte between if we are driving straight or not
    ekf.predict(yawRateCurrent)

    z = measurements[:, filterstep].reshape(-1, 1)

    # filter step is need as an argumenet for jacoabian calculation
    # so we can see if GPS measurement was available or not
    # and return the H_jacob accordingly
    ekf.update(z, HJacobian=H_of, Hx=Hx, args=filterstep)
    print(f"FilterStep = {filterstep} \n", ekf.x)
    print()
    s.save()


print("end of the file")
