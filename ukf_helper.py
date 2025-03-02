import numpy as np
from quaternion import Quaternion


def sample_sigma_points(x, Sigma, R, dt):
    """
    x: mean, shape (7,) as (quat, omega)
    Sigma: covariance, shape (6,6)
    R: process noise, shape (6,6)
    dt: time step
    """
    n = Sigma.shape[0] # 6
    sigma_points = np.zeros((2*n+1, len(x))) # shape (13, 7)
    sigma_points[0] = x # mean is the first sigma point
    S = np.linalg.cholesky(Sigma + R*dt) # shape (6,6)
    L = np.sqrt(n)*S
    for i in range(n):
        q_x = Quaternion(scalar = x[0], vec = x[1:4])
    
        # Take the ith column of L
        W_1 = L[:,i]
        q_w = Quaternion()
        q_w.from_axis_angle(W_1[:3])
        sigma_points[i+1][:4] = (q_x*q_w).q
        sigma_points[i+1][4:] = x[4:] + W_1[3:]

        W_2 = -L[:,i]
        q_w = Quaternion()
        q_w.from_axis_angle(W_2[:3])
        sigma_points[i+n+1][:4] = (q_x*q_w).q
        sigma_points[i+n+1][4:] = x[4:] + W_2[3:]

    return sigma_points


def process_update(x_k, dt):
    """
    Given a state x_k at time k, and a time step dt, this function
    returns the predicted state x_kp1 at time k+1.
    Assumes a constant angular velocity model.
    """
    x_kp1 = np.zeros_like(x_k)
    q_k = Quaternion(scalar = x_k[0], vec = x_k[1:4])
    omega = x_k[4:]
    q_delta = Quaternion()
    q_delta.from_axis_angle(omega*dt)
    q_kp1 = q_k*q_delta
    x_kp1[:4] = q_kp1.q
    x_kp1[4:] = omega
    return x_kp1


def measurement_model_gyro(x_k):
    """
    Given a state x_k at time k, this function
    returns the predicted gyro measurement.
    """
    q_k = Quaternion(scalar = x_k[0], vec = x_k[1:4])
    omega = x_k[4:]
    omega_pure_quat = Quaternion(scalar = 0, vec = omega)
    gyro = q_k.inv() * omega_pure_quat * q_k
    return gyro.vec()


def measurement_model_accel(x_k):
    """
    Given a state x_k at time k, this function
    returns the predicted accelerometer measurement.
    """
    q_k = Quaternion(scalar = x_k[0], vec = x_k[1:4])
    gravity_pure_quat = Quaternion(scalar = 0, vec = [0, 0, -9.81]) # TODO: Check if this is correct
    accn = q_k.inv() * gravity_pure_quat * q_k
    return accn.vec()


def measurement_model(x_k):
    """
    Given a state x_k at time k, this function
    returns the predicted measurement.
    """
    return np.concatenate([measurement_model_gyro(x_k), measurement_model_accel(x_k)])


def quaternion_errors(qs, ref_q):
    """
    Given a list of quaternions qs and a reference quaternion ref_q,
    this function returns the errors between each quaternion and the reference quaternion.
    """
    errors = np.zeros((len(qs), 3))
    for i, q in enumerate(qs):
        q_error = q * ref_q.inv()
        errors[i] = q_error.axis_angle()
    return errors


def average_quaternions(quaternions, initial_estimate):
    """
    Given a list of quaternions, this function returns the average quaternion.
    It used a gradient descent approach to minimize the error.
    """
    q_mean = Quaternion(scalar = initial_estimate[0], vec = initial_estimate[1:4])

    # Gradient descent
    max_iter = 1000
    epsilon = 1e-5
    for i in range(max_iter):
        errors = quaternion_errors(quaternions, q_mean)
        error_mean = np.mean(errors, axis=0)
        if np.linalg.norm(error_mean) < epsilon:
            break
        error_q = Quaternion()
        error_q.from_axis_angle(error_mean)
        q_mean = error_q * q_mean
    return q_mean.q, errors


def mean_covariance_from_states(states, initial_estimate):
    """
    Given a list of states, this function returns the average state and the covariance matrix.
    """
    quat_avg, errors = average_quaternions([Quaternion(scalar = state[0], vec = state[1:4]) for state in states], 
                                            initial_estimate)
    omegas = np.array([state[4:] for state in states])
    omega_avg = np.mean(omegas, axis=0)
    omega_error = omegas - omega_avg
    state_mean = np.concatenate([quat_avg, omega_avg])

    Sigma = np.zeros((6,6))
    for i in range(len(states)):
        state_error = np.concatenate([errors[i], omega_error[i]])
        Sigma += np.outer(state_error, state_error)
    Sigma /= len(states)
    return state_mean, Sigma


def mean_covariance_from_measurements(measurements, Q):
    """
    Given a list of measurements, this function returns the average measurement and the covariance matrix.
    Q: measurement noise covariance
    """
    measurement_avg = np.mean(measurements, axis=0)
    measurement_error = measurements - measurement_avg
    Sigma = np.zeros((6,6))
    for i in range(len(measurements)):
        Sigma += np.outer(measurement_error[i], measurement_error[i])
    Sigma /= len(measurements)
    return measurement_avg, Sigma + Q


def cross_covariance(states, measurements, state_avg, measurement_avg):
    """
    Given a list of states, a list of measurements, the average state, and the average measurement,
    this function returns the cross covariance matrix.
    """
    q_avg = Quaternion(scalar = state_avg[0], vec = state_avg[1:4])
    qs = [Quaternion(scalar = state[0], vec = state[1:4]) for state in states]
    omega_avg = state_avg[4:]
    omegas = np.array([state[4:] for state in states])
    errors = quaternion_errors(qs, q_avg)
    omega_error = omegas - omega_avg
    state_errors = np.hstack([errors, omega_error])
    measurement_error = measurements - measurement_avg
    Sigma = np.zeros((6,6))
    for i in range(len(states)):
        Sigma += np.outer(state_errors[i], measurement_error[i])
    Sigma /= len(states)
    return Sigma


def KalmanGain(Sigma_xz, Sigma_zz):
    """
    Given the cross covariance matrix and the measurement covariance matrix,
    this function returns the Kalman gain.
    """
    return Sigma_xz @ np.linalg.inv(Sigma_zz)


def update_mean_state(x_k, K, z_k, z_pred):
    """
    Given the state x_k at time k, the Kalman gain, the measurement z_k at time k,
    and the predicted measurement z_pred, this function returns the updated state.
    """
    update_vector = K @ (z_k - z_pred)
    # Convert the first 3 elements of the update vector to a quaternion
    q_update = Quaternion()
    q_update.from_axis_angle(update_vector[:3])
    q_k = Quaternion(scalar = x_k[0], vec = x_k[1:4])
    q_k_new = q_k * q_update
    x_k_new = np.zeros_like(x_k)
    x_k_new[:4] = q_k_new.q
    x_k_new[4:] = x_k[4:] + update_vector[3:]
    return x_k_new


def update_covariance(Sigma, K, Sigma_zz):
    """
    Given the covariance matrix, the Kalman gain, and the measurement covariance matrix,
    this function returns the updated covariance matrix.
    """
    return Sigma - K @ Sigma_zz @ K.T