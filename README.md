# Orientation tracking of a Quadrotor using an Unscented Kalman Filter

This project was part of the course ESE650: Learning in Robotics at UPenn. This project involves implementing Quaternion-based orientation tracking using an Unscented Kalman filter. The algorithm was adopted from the paper : [Unscented-Kalman filter](https://ieeexplore.ieee.org/document/1257247). Quaternion representation of the orientation is much more efficient in terms of computation. The non-linear relationship between the estimated orientation and the measurements prevents the usage of the classical Kalman filter. This is where the Unscented Kalman filter comes into effect, allowing nonlinear process and measurement models, and is more accurate than Extended Kalman Filters.

# Results

The algorithm is tested on three Vicon datasets. Below are the results:

Dataset 1:
<p float="center">
  <img src="./Results/res1.png" alt="result 1" class="center">
</p>

Dataset 2:
<p float="center">
  <img src="./Results/res2.png" alt="result 2" class="center">
</p>

Dataset 3:
<p float="center">
  <img src="./Results/res3.png" alt="result 3" class="center">
</p>


