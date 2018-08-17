clc;
close all;
clear;

disp('Working on it...')

run("ekf_model.m")
data = importdata('/home/cs4li/Dev/end_to_end_odometry/test/seq_06.dat');
data_size = size(data.data);
% range = 1:600;
range = 1:data_size(1);

dat_dt = data.data(range, 1);
dat_dx = data.data(range, 2);
dat_dy = data.data(range, 3);
dat_dz = data.data(range, 4);
dat_dyaw = data.data(range, 5);
dat_dpitch = data.data(range, 6);
dat_droll = data.data(range, 7);
dat_wx = data.data(range, 8);
dat_wy = data.data(range, 9);
dat_wz = data.data(range, 10);
dat_ax = data.data(range, 11);
dat_ay = data.data(range, 12);
dat_az = data.data(range, 13);
trajectory_gt_xyz = data.data(range,14:16);
trajectory_gt_eul = quat2eul(data.data(range, 17:20), 'ZYX');

dat_dt_size = size(dat_dt);
timesteps = dat_dt_size(1);

% initial states
x_prev = zeros(17, 1);
P_prev = eye(17) * 100;

x_prev(4) = dat_dx(1) / dat_dt(1);
% x_prev(5) = dat_dy(1) / dat_dt(1);
% x_prev(6) = dat_dz(1) / dat_dt(1);
P_prev(4,4) = 1e-3;
% P_prev(5,5) = 1e-3;
% P_prev(6,6) = 1e-3;

% x_prev(10) = 0.26980048;
% P_prev(10,10) = 1e-3;
% x_prev(11) = -0.01;
% P_prev(11,11) = 1e-3;


x_prev(7) = 0.009729;
P_prev(7,7) = 1e-4;
x_prev(8) = 0.04639;
P_prev(8,8) = 1e-4;


x_prev(16) = 0;
P_prev(16,16) = 1e-4;

% covariances
cov_imu = eye(6) * 1e-3;
cov_imu(1:3,1:3) = eye(3,3) * 1e-3;
% cov_imu(2,2) = 10;
% cov_imu(3,3) = 10;
cov_bias = eye(6) * 1e-3;
cov_bias(4:6,4:6) = eye(3,3) * 1e0;
cov_fc = eye(6) * 1e-1; % Rk

x_est_log = zeros(17, timesteps);
P_est_log = zeros(17, 17, timesteps);
x_pred_log = zeros(17, timesteps);
P_pred_log = zeros(17, 17, timesteps);
   
curr_tf = eye(4);
trajectory_xyz = zeros(3, timesteps);
trajectory_eul = zeros(3, timesteps);

for i = 1:timesteps
    dat_imu = [dat_wz(i); dat_wy(i); dat_wx(i); dat_ax(i); dat_ay(i); dat_az(i)];
    sub_in = [x_prev; dat_imu; dat_dt(i)];
       
    % Prediction
    Fk = F_func(sub_in);
    Juk = Ju_func(sub_in);
    Qk = Juk * cov_imu * Juk.' + Jb * cov_bias * Jb.';
    
    xk_pred = f_func(sub_in);
    Pk_pred = Fk * P_prev * Fk.' + Qk;
    
    % Update
    dat_fc = [dat_dx(i); dat_dy(i); dat_dz(i); dat_dyaw(i); dat_dpitch(i); dat_droll(i)];
    yk = dat_fc - H * xk_pred;
    Sk = H * Pk_pred * H.' + cov_fc;
    Kk = Pk_pred * H.' * inv(Sk);
    xk_est = xk_pred + Kk * yk;
    Pk_est = (eye(17, 17) - Kk * H) * Pk_pred;
        
    % Override state estimations
%     xk_est(15:17) = 0;
    
    % log results
    x_est_log(:,i) = xk_est;
    x_pred_log(:,i) = xk_pred;
    P_est_log(:,:,i) =  Pk_est;
    P_pred_log(:,:,i) =  Pk_pred;
    
    delta_tf = eye(4, 4);
    delta_tf(1:3, 1:3) = eul2rotm(xk_est(12:14)', 'ZYX');
%     delta_tf(1:3, 1:3) = eul2rotm(dat_fc(4:6)', 'ZYX');
    delta_tf(1:3, 4) = xk_est(1:3);
    curr_tf = curr_tf*delta_tf;
    trajectory_xyz(:,i) = curr_tf(1:3, 4);
    trajectory_eul(:,i) = rotm2eul(curr_tf(1:3,1:3),'ZYX');
%     
%     curr_tf(1:3, 4)
%     trajectory_eul(:,i)
    
    % Prep for the next time step
    x_prev = xk_est;
    P_prev = Pk_est;
    
    Pk_est
end

disp('all done!')
disp('Generating figures...')

figure_visible = 'off';

% ==============================================================
% Trajectory

figure('visible', figure_visible); hold on; % XY
plot(trajectory_xyz(1,:), trajectory_xyz(2,:), 'r')
plot(trajectory_gt_xyz(:,1), trajectory_gt_xyz(:,2), 'b')
title('Trajectory XY');
legend('Estimate', 'Ground Truth');
xlabel('x [m]');
ylabel('y [m]');
axis('equal');
grid;


figure('visible', figure_visible); hold on; % YZ
plot(trajectory_xyz(2,:), trajectory_xyz(3,:), 'r')
plot(trajectory_gt_xyz(:,2), trajectory_gt_xyz(:,3), 'b')
title('Trajectory YZ')
legend('Estimate', 'Ground Truth')
xlabel('y [m]')
ylabel('z [m]')
grid;

figure('visible', figure_visible); hold on; % XZ
plot(trajectory_xyz(1,:), trajectory_xyz(3,:), 'r')
plot(trajectory_gt_xyz(:,1), trajectory_gt_xyz(:,3), 'b')
title('Trajectory XZ')
legend('Estimate', 'Ground Truth')
xlabel('x [m]')
ylabel('z [m]')
grid;

% ==============================================================
% Individual

figure('visible', figure_visible); hold on; % X
plot(trajectory_xyz(1,:), 'r')
plot(trajectory_gt_xyz(:,1), 'b')
title('Global X')
legend('Estimate', 'Ground Truth')
xlabel('Frame # []')
ylabel('Dist [m]')
grid;

figure('visible', figure_visible); hold on; % Y
plot(trajectory_xyz(2,:), 'r')
plot(trajectory_gt_xyz(:,2), 'b')
title('Global Y')
legend('Estimate', 'Ground Truth')
xlabel('Frame # []')
ylabel('Dist [m]')
grid;

figure('visible', figure_visible); hold on; % Z
plot(trajectory_xyz(3,:), 'r')
plot(trajectory_gt_xyz(:,3), 'b')
title('Global Z')
legend('Estimate', 'Ground Truth')
xlabel('Frame # []')
ylabel('Dist [m]')
grid;

figure('visible', figure_visible); hold on; % Yaw
plot(unwrap(trajectory_eul(1,:)), 'r')
plot(unwrap(trajectory_gt_eul(:,1)), 'b')
title('Global Yaw')
legend('Estimate', 'Ground Truth')
xlabel('Frame # []')
ylabel('Angle [rad]')
grid;

figure('visible', figure_visible); hold on; % Pitch
plot(unwrap(trajectory_eul(2,:)), 'r')
plot(unwrap(trajectory_gt_eul(:,2)), 'b')
title('Global Pitch')
legend('Estimate', 'Ground Truth')
xlabel('Frame # []')
ylabel('Angle [rad]')
grid;

figure('visible', figure_visible); hold on; % Roll
plot(unwrap(trajectory_eul(3,:)), 'r')
plot(unwrap(trajectory_gt_eul(:,3)), 'b')
title('Global Roll')
legend('Estimate', 'Ground Truth')
xlabel('Frame # []')
ylabel('Angle [rad]')
grid;

% ==============================================================
% States
% 
figure('visible', figure_visible); hold on; 
plot(x_est_log(1,:), 'r'); plot(dat_dx, 'b');
title('State Delta X');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Dist [m]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(2,:), 'r'); plot(dat_dy, 'b');
title('State Delta Y');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Dist [m]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(3,:), 'r'); plot(dat_dz, 'b');
title('State Delta Z');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Dist [m]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(4,:), 'r'); plot(dat_dx./ dat_dt, 'b');
title('State Velocity X');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(5,:), 'r'); plot(dat_dy./ dat_dt, 'b');
title('State Velocity Y');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(6,:), 'r'); plot(dat_dz ./ dat_dt, 'b');
title('State Velocity Z');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(7,:), 'r'); 
title('State Gravity Pitch');xlabel('Frame # []');ylabel('Angle [rad]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(8,:), 'r'); 
title('State Gravity Roll');xlabel('Frame # []');ylabel('Angle [rad]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(9,:), 'r'); 
title('State Accel Bias X');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(10,:), 'r'); 
title('State Accel Bias Y');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(11,:), 'r'); 
title('State Accel Bias Z');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(12,:), 'r'); plot(dat_dyaw, 'b');
title('State Delta Yaw');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Angle [rad]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(13,:), 'r'); plot(dat_dpitch, 'b');
title('State Delta Pitch');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Angle [rad]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(14,:), 'r'); plot(dat_droll, 'b');
title('State Delta Roll');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Angle [rad]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(15,:), 'r'); 
title('State Gyro Bias Yaw');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(16,:), 'r'); 
title('State Gyro Bias Pitch');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_est_log(17,:), 'r'); 
title('State Gyro Bias Roll');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

% IMU values
figure('visible', figure_visible); hold on; 
plot(dat_wz, 'r'); 
title('Gyro Yaw Rate');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

figure('visible', figure_visible); hold on; 
plot(dat_wy, 'r'); 
title('Gyro Pitch Rate');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

figure('visible', figure_visible); hold on; 
plot(dat_wx, 'r'); 
title('Gyro Roll Rate');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

figure('visible', figure_visible); hold on; 
plot(dat_ax, 'r'); 
title('Accel X');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

figure('visible', figure_visible); hold on; 
plot(dat_ay, 'r'); 
title('Accel Y');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

figure('visible', figure_visible); hold on; 
plot(dat_az, 'r'); 
title('Accel Z');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

disp('saving figures...')

for i=1:32
    saveas(i, strcat('/home/cs4li/Dev/end_to_end_odometry/results/ekf_debug_matlab/', strcat(num2str(i), '.png')))
%     saveas(i, strcat('/home/cs4li/Dev/end_to_end_odometry/results/ekf_debug_matlab/', strcat(num2str(i), '.fig')))
end
