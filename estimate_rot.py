import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from matplotlib import pyplot as plt

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

# def estimate_rot(data_num=1):
#     #load data
#     imu = io.loadmat('data/imu/imuRaw'+str(data_num)+'.mat')
#     vicon = io.loadmat('data/vicon/viconRot'+str(data_num)+'.mat')['rots']
#     accel = imu['vals'][0:3,:]
#     gyro = imu['vals'][3:6,:]
#     T = np.shape(imu['ts'])[1]

#     # Print data for debugging
#     print('accel: ', accel)
#     print('gyro: ', gyro)
#     print('T: ', T)
#     print('vicon: ', vicon)


#     # your code goes here
#     roll = np.zeros(T)
#     pitch = np.zeros(T)
#     yaw = np.zeros(T)

#     # roll, pitch, yaw are numpy arrays of length T
#     return roll, pitch, yaw

# def euler_angle_dt_to_omega(euler_angle_dt, roll, pitch, yaw):
#     # euler_angle_dt is 3d numpy array of shape (3, 1)
#     # roll, pitch, yaw are scalars
#     # Returns 3d numpy array of shape (3, 1)
#     # omega = [wx, wy, wz]
#     transformation_matrix = np.array([[np.cos(pitch), 0, -np.cos(roll)*np.sin(pitch)],
#                                       [0, 1, np.sin(roll)],
#                                       [np.sin(pitch), 0, np.cos(roll)*np.cos(pitch)]])
#     omega = transformation_matrix.dot(euler_angle_dt)
#     return omega


def omega_to_euler_angle_dt(omega, roll, pitch, yaw):
    transformation_matrix = np.array([[np.cos(pitch), 0, -np.cos(roll)*np.sin(pitch)],
                                      [0, 1, np.sin(roll)],
                                      [np.sin(pitch), 0, np.cos(roll)*np.cos(pitch)]])
    
    euler_angle_dt = np.linalg.inv(transformation_matrix).dot(omega)
    return euler_angle_dt

from scipy.signal import savgol_filter

def process_motion_data(times, orientations, window_length=15, polyorder=3):
    """
    Process 3D motion data to smooth pitch and calculate its mean derivative.
    
    Parameters:
    times (array-like): Timestamps for each measurement
    orientations (array-like): Nx3 array of (roll, pitch, yaw) measurements
    window_length (int): Window length for Savitzky-Golay filter (must be odd)
    polyorder (int): Polynomial order for the filter
    
    Returns:
    dict: Dictionary containing processed data and statistics
    """
    # Convert inputs to numpy arrays
    times = np.array(times)
    orientations = np.array(orientations)
    
    # Extract pitch data (second column)
    pitch = orientations[:, 1]
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Apply Savitzky-Golay filter to smooth pitch data
    # smoothed_pitch = savgol_filter(pitch, window_length, polyorder)
    smoothed_pitch = np.convolve(pitch, np.ones(3)/3, mode = 'valid')
    pitch = pitch[0:len(smoothed_pitch)]
    times = times[0:len(smoothed_pitch)]

    # Compute pitch derivative
    pitch_derivative = np.diff(smoothed_pitch) / np.diff(times)
    
    # Compute slope using linear regression best fit line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(times, smoothed_pitch)
    mean_derivative = slope
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, pitch, 'b-', label='Original Pitch', alpha=0.5)
    plt.plot(times, smoothed_pitch, 'r-', label='Smoothed Pitch')
    plt.legend()
    plt.title('Pitch Data: Original vs Smoothed')
    plt.xlabel('Time')
    plt.ylabel('Pitch (degrees)')
    
    plt.subplot(2, 1, 2)
    plt.plot(times[0:len(pitch_derivative)], pitch_derivative, 'g-', label='Pitch Derivative')
    plt.axhline(y=mean_derivative, color='r', linestyle='--', 
                label=f'Mean Derivative: {mean_derivative:.2f}')
    plt.legend()
    plt.title('Pitch Derivative')
    plt.xlabel('Time')
    plt.ylabel('Pitch Rate (degrees/s)')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'smoothed_pitch': smoothed_pitch,
        # 'pitch_derivative': pitch_derivative,
        'mean_derivative': mean_derivative
    }


def calibrate_sensor(data_num=1):
    #load data
    imu = io.loadmat('data/imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('data/vicon/viconRot'+str(data_num)+'.mat')
    vicon_ts = vicon['ts'][0]
    vicon_rots = vicon['rots'].transpose((2,0,1))
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    

    # To compute the bias and sensitivity, I will try to fit a linear model
    # using the vicon data and the IMU data.
    # alpha = sensitivity, beta = bias
    # We can create an equation of the form (in all 3 axes):
    # alpha * (1023 * body_frame_accn) / (3300 * 9.81) + beta = raw_imu_data
    # We can solve for alpha and beta using least squares method

    # Synchronize time stamps between vicon and IMU
    imu_ts = imu['ts'][0]
    vice_ts = vicon['ts'][0]

    # For each timestamp in IMU, find the closest timestamp in Vicon
    vicon_idx = 0
    vicon_rots_synced = []
    accel_synced = []
    gyro_synced = []
    time_diff_threshold = 0.01
    for i in range(len(imu_ts)):
        while vicon_idx < len(vice_ts) and vice_ts[vicon_idx] < imu_ts[i]:
            vicon_idx += 1
        if vicon_idx < len(vice_ts) and abs(vice_ts[vicon_idx] - imu_ts[i]) < time_diff_threshold:
            vicon_rots_synced.append(vicon_rots[vicon_idx])
            accel_synced.append(accel[:,i])
            gyro_synced.append(gyro[:,i])

    vicon_rots = np.array(vicon_rots_synced)
    accel = np.array(accel_synced).T
    gyro = np.array(gyro_synced).T

    print('vicon_rots.shape: ', vicon_rots.shape)
    print('accel.shape: ', accel.shape)
    print('gyro.shape: ', gyro.shape)

    vicon_roll = np.zeros(T)
    vicon_pitch = np.zeros(T)
    vicon_yaw = np.zeros(T)
    for i in range(len(vicon_rots)):
        vicon_rot = vicon_rots[i]#.reshape((3,3))
        q_vicon = Quaternion()
        q_vicon.from_rotm(vicon_rot)
        roll, pitch, yaw = q_vicon.euler_angles()
        vicon_roll[i] = roll
        vicon_pitch[i] = pitch
        vicon_yaw[i] = yaw

    # Plot vicon data
    plt.plot(vicon_roll, label='vicon_roll')
    plt.plot(vicon_pitch, label='vicon_pitch')
    plt.plot(vicon_yaw, label='vicon_yaw')
    plt.legend()
    plt.show()

    max_T = len(vicon_rots)
    imu_ax_X = np.zeros((max_T, 2))
    imu_ax_b = np.zeros(max_T)

    imu_ay_X = np.zeros((max_T, 2))
    imu_ay_b = np.zeros(max_T)

    imu_az_X = np.zeros((max_T, 2))
    imu_az_b = np.zeros(max_T)
    for i in range(max_T):
        # In world frame - accn = [0, 0, 9.81]
        # Convert to body frame
        accn_body_frame = np.dot(vicon_rots[i].reshape((3,3)), np.array([0, 0, -9.81]))

        # Convert body frame to IMU frame -> ax, and ay are reverse in IMU frame
        imu_ax = -accn_body_frame[0]
        imu_ay = -accn_body_frame[1]
        imu_az = -accn_body_frame[2]

        # This will act as a value for IMU's accelerometer
        imu_ax_X[i] = [(1023 * imu_ax) / (3300), 1]
        imu_ax_b[i] = accel[0][i]
    
        imu_ay_X[i] = [(1023 * imu_ay) / (3300), 1]
        imu_ay_b[i] = accel[1][i]
    
        imu_az_X[i] = [(1023 * imu_az) / (3300), 1]
        imu_az_b[i] = accel[2][i]
    
    # Least squares method
    fit_ax = np.linalg.lstsq(imu_ax_X, imu_ax_b, rcond=None)
    # print(f"fit_ax: {fit_ax}")
    fit_ay = np.linalg.lstsq(imu_ay_X, imu_ay_b, rcond=None)
    # print(f"fit_ay: {fit_ay}")
    fit_az = np.linalg.lstsq(imu_az_X, imu_az_b, rcond=None)
    # print(f"fit_az: {fit_az}")

    accel_bias = np.array([fit_ax[0][1], fit_ay[0][1], fit_az[0][1]])
    accel_sensitivity = np.array([fit_ax[0][0], fit_ay[0][0], fit_az[0][0]])

    gyro_wx_X = np.zeros((max_T-1, 2))
    gyro_wx_b = np.zeros(max_T-1)
    gyro_wy_X = np.zeros((max_T-1, 2))
    gyro_wy_b = np.zeros(max_T-1)
    gyro_wz_X = np.zeros((max_T-1, 2))
    gyro_wz_b = np.zeros(max_T-1)

    print('vicon_ts: ', vicon_ts)
    print(vicon_ts[0])
    print(vicon_ts[1])
    print('gyro_wx_X.shape: ', gyro_wx_X.shape)

    timesteps = []
    eulers = []
    for i in range(1890, 2090):
        # Take two rotation matrices at i, i+1
        Rt = vicon_rots[i]
        Rt_p1 = vicon_rots[i+1]
        # if (i < 10):
        #     print('Rt: ', Rt)
        #     print('Rt_p1: ', Rt_p1)

        # dt = time difference between the two readings
        dt = vicon_ts[i+1] - vicon_ts[i]

        # Extract the current euler angles
        q_curr = Quaternion()
        q_curr.from_rotm(Rt)
        curr_euler = q_curr.euler_angles()
        timesteps.append(vicon_ts[i])
        eulers.append(curr_euler)

        # Extract the next euler angles
        q_next = Quaternion()
        q_next.from_rotm(Rt_p1)
        next_euler = q_next.euler_angles()

        # Compute the angular velocity
        R_rel = Rt_p1.dot(Rt.transpose())
        # print("R_rel: ", R_rel)

        # # Convert to axis angle representation
        q_rel = Quaternion()
        q_rel.from_rotm(R_rel)
        omega = q_rel.axis_angle() / dt

        # Convert omega to euler angle derivative
        # euler_angle_dt = omega_to_euler_angle_dt(omega, curr_euler[0], curr_euler[1], curr_euler[2])
        # euler_angle_dt = omega
        euler_angle_dt = (next_euler - curr_euler) / dt
        # print("euler_angle_dt: ", euler_angle_dt)

        # Gyroscope data
        gyro_roll_rate = gyro[1][i]
        gyro_pitch_rate = gyro[2][i]
        gyro_yaw_rate = gyro[0][i]

        gyro_wx_X[i] = [1023 * euler_angle_dt[0] / (3300), 1]
        gyro_wx_b[i] = gyro_roll_rate

        gyro_wy_X[i] = [1023 * euler_angle_dt[1] / (3300), 1]
        gyro_wy_b[i] = gyro_pitch_rate

        gyro_wz_X[i] = [1023 * euler_angle_dt[2] / (3300), 1]
        gyro_wz_b[i] = gyro_yaw_rate

    print("gyro pitch mean: ", np.mean(gyro[2][1890:2090]))
    process_motion_data(timesteps, eulers)
    
    # # Least squares method
    fit_wx = np.linalg.lstsq(gyro_wx_X, gyro_wx_b, rcond=None)
    # print(f"fit_wx: {fit_wx}")
    fit_wy = np.linalg.lstsq(gyro_wy_X, gyro_wy_b, rcond=None)
    # print(f"fit_wy: {fit_wy}")
    fit_wz = np.linalg.lstsq(gyro_wz_X, gyro_wz_b, rcond=None)
    # print(f"fit_wz: {fit_wz}")

    gyro_bias = np.array([fit_wx[0][1], fit_wy[0][1], fit_wz[0][1]])
    gyro_sensitivity = np.array([fit_wx[0][0], fit_wy[0][0], fit_wz[0][0]])

    # Assume drone is at rest at the beginning in first 20 readings
    rest_T = 20
    bias = np.mean(gyro[:,0:rest_T], axis=1)
    print(gyro[:,0:rest_T])
    bias = bias[1], bias[2], bias[0]

    print("gyro bias using rest approach: ", bias)
    # print("gyro sensitivity using rest approach: ", sensitivity)
    
    # accel_bias, gyro_bias, accel_sensitivity, gyro_sensitivity are numpy arrays of length 3
    return accel_bias, gyro_bias, accel_sensitivity, gyro_sensitivity


if __name__ == '__main__':
    # for i in range(1,3):
        # roll, pitch, yaw = estimate_rot(i)
        # print('Data Set: ', i)
        # print('Roll: ', roll[-1])
        # print('Pitch: ', pitch[-1])
        # print('Yaw: ', yaw[-1])
        # print(' ')
    accel_bias, gyro_bias, accel_sensitivity, gyro_sensitivity = calibrate_sensor(2)
    print(' ')
    print('accel_bias: ', accel_bias)
    print('accel_sensitivity: ', accel_sensitivity)    
    print('gyro_bias: ', gyro_bias)
    print('gyro_sensitivity: ', gyro_sensitivity)