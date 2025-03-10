import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from matplotlib import pyplot as plt
from ukf_helper import *


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
        # Convert gyro data from from Wz, Wx, Wy to Wx, Wy, Wz
        gyro[i] = np.array([gyro[i,1], gyro[i,2], gyro[i,0]])

        accel[i] = (3300 * (accel[i] - accel_biases)) / (1023 * accel_sensitivity)
        gyro[i] = (3300 * (gyro[i] - gyro_biases)) / (1023 * gyro_sensitivity)

        # Negate accelerometer x and y axes
        accel[i,0] = -accel[i,0]
        accel[i,1] = -accel[i,1]
        accel[i,2] = accel[i,2]

    # Assign Noise covariances and initialze state covariance
    meas_noise_cov = np.eye(6) * 0.1
    process_noise_cov = np.eye(6) * 10
    state_cov = np.eye(6)

    # Initialize state
    state_mean = np.array([1, 0, 0, 0, 0, 0, 0]) # q = [1, 0, 0, 0], omega = [0, 0, 0]

    # Store results
    roll, pitch, yaw = [], [], []
    ox = []
    oy = []
    oz = []
    cov = []

    for i in range(T):
        if i == 0:
            dt = imu['ts'][0][i+1] - imu['ts'][0][i]
        else:
            dt = imu['ts'][0][i] - imu['ts'][0][i-1]

        # Sample sigma points
        sigma_points = sample_sigma_points(state_mean, state_cov, process_noise_cov, dt)

        # Apply dynamics update to each sigma point
        sigma_points_Y = np.zeros_like(sigma_points)
        for j in range(len(sigma_points)):
            sigma_points_Y[j] = process_update(sigma_points[j], dt)

        # Compute prior mean and covariance
        initial_mean_estimate = sigma_points_Y[0]
        mu_prior, Sigma_prior = mean_covariance_from_states(
            sigma_points_Y, initial_mean_estimate
        )

        # Compute new sigma points
        sigma_points_new = sample_sigma_points(mu_prior, Sigma_prior, 0, dt)

        # Compute predicted measurements
        predicted_measurements = np.zeros((len(sigma_points_new), 6))
        for j in range(len(sigma_points_new)):
            predicted_measurements[j] = measurement_model(sigma_points_new[j])

        # Compute predicted measurement mean and covariance
        mu_hat, Sigma_hat = mean_covariance_from_measurements(
            predicted_measurements, 
            meas_noise_cov
        )

        # Compute cross covariance
        cross_cov = cross_covariance(sigma_points_Y, predicted_measurements, mu_prior, mu_hat)

        # Compute Kalman gain
        K = KalmanGain(cross_cov, Sigma_hat)

        # Update state mean and covariance
        observation = np.concatenate((accel[i], gyro[i]))
        state_mean = update_mean_state(mu_prior, K, observation, mu_hat)
        state_cov = update_covariance(Sigma_prior, K, Sigma_hat)

        # Store results
        q = Quaternion(scalar = state_mean[0], vec = state_mean[1:4])
        rpy = q.euler_angles()
        roll.append(rpy[0])
        pitch.append(rpy[1])
        yaw.append(rpy[2])

        # Store omega
        ox.append(state_mean[4])
        oy.append(state_mean[5])
        oz.append(state_mean[6])

        # Store covariance
        cov.append(np.sqrt(np.diag(state_cov)))

    print("State mean: ", state_mean)
    print("State covariance: ", state_cov)

    # Load Vicon data
    vicon = io.loadmat('data/vicon/viconRot'+str(data_num)+'.mat')
    vicon_roll = []
    vicon_pitch = []
    vicon_yaw = []
    quat = Quaternion()
    for i in range(vicon['rots'].shape[-1]):
        rot = vicon['rots'][:,:,i].reshape(3,3)
        q = quat.from_rotm(rot)
        euler_angles = quat.euler_angles()
        vicon_roll.append(euler_angles[0])
        vicon_pitch.append(euler_angles[1])
        vicon_yaw.append(euler_angles[2])

    # roll, pitch, yaw are numpy arrays of length T
    plt.figure('roll')
    plt.plot(vicon_roll, label='vicon roll')
    plt.plot(roll, label='filtered roll')
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Roll vs Vicon Roll")
    
    
    plt.figure('pitch')
    plt.plot(vicon_pitch, label='vicon pitch')
    plt.plot(pitch, label='filtered pitch')
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Pitch vs Vicon Pitch")
    
    plt.figure('yaw')
    plt.plot(vicon_yaw, label='vicon yaw')
    plt.plot(yaw, label='filtered yaw')
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Yaw vs Vicon Yaw")
    #plt.show()
    
    plt.figure('ox')
    plt.plot(gyro[:,0], label='gyroX')
    plt.plot(ox, label='filtered omega x')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.legend()
    plt.title("Filtered Omega_x vs Gyro_x")
    
    plt.figure('oy')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.plot(gyro[:,1], label='gyroY')
    plt.plot(oy, label='filtered omega y')
    plt.legend()
    plt.title("Filtered Omega_y vs Gyro_y")
    
    plt.figure('oz')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.plot(gyro[:,2], label='gyroZ')
    plt.plot(oz, label='filtered omega z')
    plt.legend()
    plt.title("Filtered Omega_z vs Gyro_z")
    #plt.show()
    
    k = np.arange(1,T+1)
    cov = np.asarray(cov).T
    plt.figure('q_mc1')
    plt.plot(k, roll)
    #print(roll[:] - cov[0,:])
    plt.fill_between(k, roll[:] - cov[0,:], roll[:] + cov[0,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Roll with variance bounds")

    plt.figure('q_mc2')
    plt.plot(k, pitch)
    plt.fill_between(k, pitch - cov[1,:], pitch + cov[1,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Pitch with variance bounds")

    plt.figure('q_mc3')
    plt.plot(k, yaw)
    plt.fill_between(k, yaw - cov[2,:], yaw + cov[2,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Yaw with variance bounds")
    
    plt.figure('o_mc1')
    plt.plot(k, ox)
    plt.fill_between(k, ox - cov[3,:], ox + cov[3,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Omega_x with variance bounds")

    plt.figure('o_mc2')
    plt.plot(k, oy)
    plt.fill_between(k, oy - cov[4,:], oy + cov[4,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Omega_y with variance bounds")

    plt.figure('o_mc3')
    plt.plot(k, oz)
    plt.fill_between(k, oz - cov[5,:], oz + cov[5,:], alpha=0.5, color='red')
    plt.xlabel("Time steps")
    plt.ylabel("Values")
    plt.title("Filtered Omega_z with variance bounds")
    plt.show()

    return roll, pitch, yaw


if __name__ == '__main__':
    roll, pitch, yaw = estimate_rot(1)
    plt.plot(roll, label='Roll')
    plt.plot(pitch, label='Pitch')
    plt.plot(yaw, label='Yaw')
    plt.legend()
    plt.show()