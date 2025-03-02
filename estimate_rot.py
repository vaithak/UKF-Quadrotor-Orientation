import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from matplotlib import pyplot as plt
import ukf_helper as ukf


def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('data/imu/imuRaw'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:].T
    accel = accel.astype(float)
    gyro = imu['vals'][3:6,:].T
    gyro = gyro.astype(float)
    T = np.shape(imu['ts'])[1]

    # Output from calibration
    gyro_biases = np.array([373.65, 375.15, 369.75])
    gyro_sensitivity = np.array([201.56, 204.86, 203.21])
    accel_biases = np.array([510.3, 500.23, 502.51])
    accel_sensitivity = np.array([34.23, 33.9, 33.76])

    # Initialize variables
    print("Accel shape: ", accel.shape)
    print("Gyro shape: ", gyro.shape)

    # Convert raw sensor data to physical units
    for i in range(T):
        accel[i] = (3300 * (accel[i] - accel_biases)) / (1023 * accel_sensitivity)
        gyro[i] = (3300 * (gyro[i] - gyro_biases)) / (1023 * gyro_sensitivity)

        # Negate accelerometer x and y axes
        accel[i,0] = -accel[i,0]
        accel[i,1] = -accel[i,1]
        accel[i,2] = accel[i,2]

        # Convert gyro data from from Wz, Wx, Wy to Wx, Wy, Wz
        gyro[i] = np.array([gyro[i,1], gyro[i,2], gyro[i,0]])

    # Assign Noise covariances and initialze state covariance
    meas_noise_cov = np.eye(6) * 6.0
    process_noise_cov = np.eye(6) * 0.1
    state_cov = np.eye(6) * 0.001

    # Initialize state
    state_mean = np.array([1, 0, 0, 0, 0, 0, 0]) # q = [1, 0, 0, 0], omega = [0, 0, 0]

    # Store results
    roll, pitch, yaw = [], [], []

    for i in range(T):
        if i == 0:
            dt = imu['ts'][0][i+1] - imu['ts'][0][i]
        else:
            dt = imu['ts'][0][i] - imu['ts'][0][i-1]

        # Sample sigma points
        sigma_points = ukf.sample_sigma_points(state_mean, state_cov, process_noise_cov, dt)

        # Apply dynamics update to each sigma point
        sigma_points_Y = np.zeros_like(sigma_points)
        for j in range(len(sigma_points)):
            sigma_points_Y[j] = ukf.process_update(sigma_points[j], dt)

        # Compute prior mean and covariance
        initial_mean_estimate = sigma_points_Y[0]
        mu_prior, Sigma_prior = ukf.mean_covariance_from_states(
            sigma_points_Y, initial_mean_estimate
        )

        # Compute new sigma points
        sigma_points_new = ukf.sample_sigma_points(mu_prior, Sigma_prior, 0, dt)

        # Compute predicted measurements
        predicted_measurements = np.zeros((len(sigma_points_new), 6))
        for j in range(len(sigma_points_new)):
            predicted_measurements[j] = ukf.measurement_model(sigma_points_new[j])

        # Compute predicted measurement mean and covariance
        mu_hat, Sigma_hat = ukf.mean_covariance_from_measurements(
            predicted_measurements, 
            meas_noise_cov
        )

        # Compute cross covariance
        cross_cov = ukf.cross_covariance(sigma_points_new, predicted_measurements, mu_prior, mu_hat)

        # Compute Kalman gain
        K = ukf.KalmanGain(cross_cov, Sigma_hat)

        # Update state mean and covariance
        observation = np.concatenate((accel[i], gyro[i]))
        state_mean = ukf.update_mean_state(mu_prior, K, observation, mu_hat)
        state_cov = ukf.update_covariance(Sigma_prior, K, cross_cov)

        # Store results
        q = Quaternion(scalar = state_mean[0], vec = state_mean[1:4])
        rpy = q.euler_angles()
        roll.append(rpy[0])
        pitch.append(rpy[1])
        yaw.append(rpy[2])

    print("State mean: ", state_mean)
    print("State covariance: ", state_cov)
    return roll, pitch, yaw


if __name__ == '__main__':
    roll, pitch, yaw = estimate_rot()
    plt.plot(roll, label='Roll')
    plt.plot(pitch, label='Pitch')
    plt.plot(yaw, label='Yaw')
    plt.legend()
    plt.show()