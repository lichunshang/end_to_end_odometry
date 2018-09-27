clc;
close all;
clear;

disp('Working on it...')

run("indirect_ekf_model.m")
data = importdata('/home/cs4li/Dev/end_to_end_odometry/test/seq_06.dat');
data_size = size(data.data);
range = 287:320;
% range = 1:data_size(1);

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
x_nom_prev = zeros(22, 1);
x_nom_prev(20:22) = g0; % initialize gravity to -g
x_nom_prev(7:10) = data.data(range(1), 17:20).'; % initialize the first pose to be identity
x_nom_prev(4) = dat_dx(1) / dat_dt(1);
x_prev = zeros(18, 1);
P_prev = eye(18) * 100;

% covariances
imu_covar = [1e-3, 1e-3 , 1e-3 , 1e-3].'; % a w ba bw
cov_fc = eye(6) * 1e-8; % measurement covar

x_est_log = zeros(18, timesteps);
P_est_log = zeros(18, 18, timesteps);
P_est_reset_log = zeros(18, 18, timesteps);
x_pred_log = zeros(18, timesteps);
P_pred_log = zeros(18, 18, timesteps);
x_nom_pred_log = zeros(22, timesteps);
x_nom_est_log = zeros(22, timesteps);
   
p_cumm =  data.data(range(1),14:16)';
trajectory_xyz = zeros(3, timesteps);
trajectory_eul = zeros(3, timesteps);

for i = 1:timesteps
    dat_imu = [dat_ax(i); dat_ay(i); dat_az(i); dat_wx(i); dat_wy(i); dat_wz(i)];
    dat_fc_axis_angle = ypr2axisangle([dat_dyaw(i) dat_dpitch(i) dat_droll(i)]);
    dat_fc = [dat_dx(i); 0; dat_dz(i); dat_fc_axis_angle]
    
    % Updating Nominal State
    xk_nom_pred = f_nom_func([x_nom_prev; dat_imu; dat_dt(i)]);
    
%     R_quat(x_nom_prev(7:10))*(dat_imu(1:3))
%     R_quat(x_nom_prev(7:10))*(dat_imu(1:3)) + g0
    
    % Prediction Error State
    Fxk = Fx_func([x_nom_prev; x_prev; dat_imu; dat_dt(i)]);
    Qik = Qi_func([imu_covar; dat_dt(i)]);
    
    xk_pred = Fxk * x_prev; % this is always zero
    Pk_pred = Fxk * P_prev * Fxk.' + Fi * Qik * Fi.';
    
    H = H_nom_func(xk_nom_pred) * H_es_func(xk_nom_pred);
%     
%     dat_fc
%     h_nom_func(xk_nom_pred)
    
    % Update Error State
    yk = dat_fc - h_nom_func(xk_nom_pred);
    Sk = H * Pk_pred * H.' + cov_fc;
    Kk = Pk_pred * H.' * inv(Sk);
    xk_est = xk_pred + Kk * yk; % xk_pred is always zero
    Pk_est = (eye(18, 18) - Kk * H) * Pk_pred;
    
    % Propagate for reset
    Gk = G_func(xk_est);
    Pk_est_reset = Gk * Pk_est * Gk.';
    xk_est
    % Correct nominal states with estimate from EKF
    xk_nom_est = f_nom_corr_func([xk_nom_pred; xk_est]);
%     xk_nom_est = xk_nom_pred;
    
    p_cumm = p_cumm + xk_nom_est(1:3);
    
    % log results
    x_est_log(:,i) = xk_est;
    x_nom_est_log(:,i) = xk_nom_est;
    x_pred_log(:,i) = xk_pred;
    x_nom_pred_log(:,i) = xk_nom_pred;
    P_est_log(:,:,i) =  Pk_est;
    P_pred_log(:,:,i) =  Pk_pred;
    P_est_reset_log(:,:,i) = Pk_est_reset;
    
%     delta_tf = eye(4, 4);
%     delta_tf(1:3, 1:3) = quat2rotm(xk_nom_est(7:10).');
%     delta_tf(1:3, 4) = xk_nom_est(1:3);
%     curr_tf = curr_tf*delta_tf;
%     trajectory_xyz(:,i) = curr_tf(1:3, 4);
%     trajectory_eul(:,i) = rotm2eul(curr_tf(1:3,1:3),'ZYX');
    trajectory_xyz(:,i) = p_cumm;
    trajectory_eul(:,i) = quat2eul(xk_nom_est(7:10).','ZYX');


    % Prep for the next time step
    x_prev = zeros(18, 1); % x_prev is always zero after reset
    P_prev = Pk_est_reset;
    x_nom_prev = xk_nom_est;
    
%     x_nom_prev = xk_nom_pred;
%     x_nom_est_log(:,i) = x_nom_prev;
end

% x_nom_est_eul_log = quat2eul(x_nom_est_log(7:10,:).', 'ZYX').';

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
axis('equal');
grid;

figure('visible', figure_visible); hold on; % XZ
plot(trajectory_xyz(1,:), trajectory_xyz(3,:), 'r')
plot(trajectory_gt_xyz(:,1), trajectory_gt_xyz(:,3), 'b')
title('Trajectory XZ')
legend('Estimate', 'Ground Truth')
xlabel('x [m]')
ylabel('z [m]')
axis('equal');
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
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(4,:), 'r'); plot(dat_dx./ dat_dt, 'b');
% title('State Velocity X');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(5,:), 'r'); plot(dat_dy./ dat_dt, 'b');
% title('State Velocity Y');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(6,:), 'r'); plot(dat_dz ./ dat_dt, 'b');
% title('State Velocity Z');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('V [m/s]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(7,:), 'r'); 
% title('State Gravity Pitch');xlabel('Frame # []');ylabel('Angle [rad]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(8,:), 'r'); 
% title('State Gravity Roll');xlabel('Frame # []');ylabel('Angle [rad]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(9,:), 'r'); 
% title('State Accel Bias X');xlabel('Frame # []');ylabel('a [m/s^2]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(10,:), 'r'); 
% title('State Accel Bias Y');xlabel('Frame # []');ylabel('a [m/s^2]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(11,:), 'r'); 
% title('State Accel Bias Z');xlabel('Frame # []');ylabel('a [m/s^2]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(12,:), 'r'); plot(dat_dyaw, 'b');
% title('State Delta Yaw');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Angle [rad]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(13,:), 'r'); plot(dat_dpitch, 'b');
% title('State Delta Pitch');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Angle [rad]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(14,:), 'r'); plot(dat_droll, 'b');
% title('State Delta Roll');legend('Estimate', 'Ground Truth');xlabel('Frame # []');ylabel('Angle [rad]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(15,:), 'r'); 
% title('State Gyro Bias Yaw');xlabel('Frame # []');ylabel('rate [rad/s]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(16,:), 'r'); 
% title('State Gyro Bias Pitch');xlabel('Frame # []');ylabel('rate [rad/s]');grid;
% 
% figure('visible', figure_visible); hold on; 
% plot(x_est_log(17,:), 'r'); 
% title('State Gyro Bias Roll');xlabel('Frame # []');ylabel('rate [rad/s]');grid;

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
    saveas(i, strcat('/home/cs4li/Dev/end_to_end_odometry/results/ekf_debug_matlab/', strcat(num2str(i), '.fig')))
end
