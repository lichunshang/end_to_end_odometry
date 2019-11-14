# end_to_end_odometry
End to End (Lidar) Odometry using CNN and LSTM
Inspired by the DeepVO and ESP-VO architecture, used CNN-LSTM to learn Lidar Odometry. 
This also included the attempt to add an EKF to fuse IMU data as part of the network 
for improving Odometry estimates. The EKF portion is not working, and it is difficult 
to debug in Tensorflow. The project is abandoned and https://github.com/lichunshang/deep_ekf_vio
contains the follow up work on this idea.
