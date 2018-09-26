clc;
close all;
clear;

disp('Working on it...')
run('constants.m')
% data = importdata('/home/cs4li/Dev/end_to_end_odometry/test/data/artificial_straight.dat');
% data_init = importdata('/home/cs4li/Dev/end_to_end_odometry/test/data/artificial_straight_init.dat');
data = importdata('/home/cs4li/Dev/end_to_end_odometry/test/data/seq_00.dat');
data_init = importdata('/home/cs4li/Dev/end_to_end_odometry/test/data/seq_00_init.dat');
data_size = size(data.data);
range = 1:data_size(1);

dat_dt = data.data(range, 1);
dat_dt_size = size(dat_dt);
timesteps = dat_dt_size(1);


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
trajectory_gt_quat = data.data(range,17:20);
trajectory_gt_SO3 = quat2rotm(data.data(range,17:20));
trajectory_gt_SE3 = zeros(4, 4, timesteps);
trajectory_gt_SE3(4,4,:) = 1;
trajectory_gt_SE3(1:3,1:3,:) = trajectory_gt_SO3;
trajectory_gt_SE3(1:3,4,:) = trajectory_gt_xyz';
trajectory_gt_eul = quat2eul(data.data(range, 17:20), 'ZYX');
velocity_gt_xyz = data.data(range, 2:4).';

data_v0 = data_init.data(9:11);
data_euler0 = flip(data_init.data(4:6));

% initial states
x_nom_prev = zeros(16, 1);
x_nom_prev(1:3) = [0 0 0].';
x_nom_prev(4:6) = data_v0;
x_nom_prev(7:10) = rotm2quat(eul2rotm([data_euler0(1) 0 0], 'ZYX').' * eul2rotm(data_euler0, 'ZYX'));

x_prev = zeros(15, 1);
P_prev = eye(15) * 10;
P_prev(13:15, 13:15) = eye(3,3) * 1e-6; 
P_prev(10:12, 10:12) = eye(3,3) * 1e-3; 

% covariances
imu_covar = [1e-1, 1e-3 , 1e-3, 1e-6].'; % a w ba bw
cov_meas = eye(6,6) * 1e-2; % measurement covar

x_est_log = zeros(15, timesteps);
P_est_log = zeros(15, 15, timesteps);
P_est_reset_log = zeros(15, 15, timesteps);
x_pred_log = zeros(15, timesteps);
P_pred_log = zeros(15, 15, timesteps);
x_nom_pred_log = zeros(16, timesteps);
x_nom_est_log = zeros(16, timesteps);
   
trajectory_xyz = zeros(3, timesteps);
trajectory_eul = zeros(3, timesteps);

for i = 1:timesteps
    dat_imu = [dat_ax(i); dat_ay(i); dat_az(i); dat_wx(i); dat_wy(i); dat_wz(i)];
    
    % Updating Nominal State
    xk_nom_pred = f_nom_func(x_nom_prev, dat_imu, dat_dt(i));
    
    % Prediction Error State
    Fxk = Fx_func(x_nom_prev, dat_imu, dat_dt(i));
    Qik = Qi_func(imu_covar, dat_dt(i));
    
    xk_pred = Fxk * x_prev; % this is always zero
    Pk_pred = Fxk * P_prev * Fxk.' + Fi * Qik * Fi.';

    meas_p = [dat_dx(i); dat_dy(i); dat_dz(i)];
    meas_theta = log_map_nonsym(eul2rotm([dat_dyaw(i) dat_dpitch(i) dat_droll(i)], 'ZYX'));
    dat_meas = [meas_p; meas_theta];
    
    [meas_pred, H] = h_func(x_nom_prev, xk_nom_pred, dat_imu, dat_dt(i));
    
    % Update Error State
    yk = dat_meas - meas_pred;

    Sk = H * Pk_pred * H.' + cov_meas;
    Kk = Pk_pred * H.' / Sk;
    xk_est = xk_pred + Kk * yk; % xk_pred is always zero
    Pk_est = (eye(15, 15) - Kk * H) * Pk_pred;
%     Pk_est = (eye(15, 15) - Kk * H) * Pk_pred  * (eye(15, 15) - Kk * H).' + Kk * cov_meas * Kk.';
    
    % Propagate for reset
    Gk = G_func(xk_est);
    Pk_est_reset = Gk * Pk_est * Gk.';
    
    % Correct nominal states with estimate from EKF
    xk_nom_est = f_nom_corr_func(xk_nom_pred, xk_est);
%     xk_nom_est = xk_nom_pred;
   
    % log results
    x_est_log(:,i) = xk_est;
    x_nom_est_log(:,i) = xk_nom_est;
    x_pred_log(:,i) = xk_pred;
    x_nom_pred_log(:,i) = xk_nom_pred;
    P_est_log(:,:,i) =  Pk_est;
    P_pred_log(:,:,i) =  Pk_pred;
    P_est_reset_log(:,:,i) = Pk_est_reset;
    
    trajectory_xyz(:,i) = xk_nom_est(1:3);
    trajectory_eul(:,i) = quat2eul(xk_nom_est(7:10).','ZYX');
    velocity_gt_xyz(:,i) = trajectory_gt_SO3(:,:,i) * velocity_gt_xyz(:,i) / dat_dt(i);

    % Prep for the next time step
    x_prev = zeros(15, 1); % x_prev is always zero after reset
    P_prev = Pk_est;
    x_nom_prev = xk_nom_est;
    
    % disp
    i
    yk
    Pk_est
    xk_est
    xk_nom_est
end

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
% axis('equal');
grid;

figure('visible', figure_visible); hold on; % XZ
plot(trajectory_xyz(1,:), trajectory_xyz(3,:), 'r')
plot(trajectory_gt_xyz(:,1), trajectory_gt_xyz(:,3), 'b')
title('Trajectory XZ')
legend('Estimate', 'Ground Truth')
xlabel('x [m]')
ylabel('z [m]')
% axis('equal');
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
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(1,:), 'r'); plot(dat_dx, 'b');
% title('State Delta X');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Dist [m]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(2,:), 'r'); plot(dat_dy, 'b');
% title('State Delta Y');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Dist [m]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(3,:), 'r'); plot(dat_dz, 'b');
% title('State Delta Z');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Dist [m]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(4,:), 'r'); plot(velocity_gt_xyz(1,:), 'b');
title('State Velocity X');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(5,:), 'r'); plot(velocity_gt_xyz(2,:), 'b');
title('State Velocity Y');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(6,:), 'r'); plot(velocity_gt_xyz(3,:), 'b');
title('State Velocity Z');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(11,:), 'r'); 
title('State Accel Bias X');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(12,:), 'r'); 
title('State Accel Bias Y');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(13,:), 'r'); 
title('State Accel Bias Z');xlabel('Frame # []');ylabel('a [m/s^2]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(14,:), 'r'); 
title('State Gyro Bias X');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(15,:), 'r'); 
title('State Gyro Bias Y');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

figure('visible', figure_visible); hold on; 
plot(x_nom_est_log(16,:), 'r'); 
title('State Gyro Bias Z');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

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

figures_list = findobj('type','figure');
num_figures = length(figures_list);

for i=1:num_figures
    saveas(i, strcat('/home/cs4li/Dev/end_to_end_odometry/results/ekf_debug_matlab/', strcat(num2str(i), '.png')))
    saveas(i, strcat('/home/cs4li/Dev/end_to_end_odometry/results/ekf_debug_matlab/', strcat(num2str(i), '.fig')))
end

disp('All done')

% openfig('/home/cs4li/Dev/end_to_end_odometry/results/ekf_debug_matlab/1.fig','new','visible')