clc;
close all;
clear;

disp('Working on it...')

run("ekf_model.m")
data = importdata('/home/cs4li/Dev/end_to_end_odometry/test/seq_00.dat');

dat_dt = data.data(:, 1);
dat_dx = data.data(:, 2);
dat_dy = data.data(:, 3);
dat_dz = data.data(:, 4);
dat_dyaw = data.data(:, 5);
dat_dpitch = data.data(:, 6);
dat_droll = data.data(:, 7);
dat_wx = data.data(:, 8);
dat_wy = data.data(:, 9);
dat_wz = data.data(:, 10);
dat_ax = data.data(:, 11);
dat_ay = data.data(:, 12);
dat_az = data.data(:, 13);
trajectory_gt_xyz = data.data(:,14:16);
trajectory_gt_eul = quat2eul(data.data(:, 17:20));

data_size = size(data.data);
timesteps = data_size(1);

x_prev = zeros(17, 1);
x_prev(4) = dat_dx(1) / dat_dt(1);
P_prev = eye(17) * 10;
cov_imu = eye(6) * 1e-3;
cov_bias = eye(6) * 1e-3;
cov_fc = eye(6) * 1e-8; % Rk

x_est_log = zeros(17, timesteps);
P_est_log = zeros(17, 17, timesteps);
x_pred_log = zeros(17, timesteps);
P_pred_log = zeros(17, 17, timesteps);

% x_est_log(:,1) = x_prev;
% x_pred_log(:,1) = x_prev;
% P_est_log(:,:,1) =  P_prev;
% P_pred_log(:,:,1) =  P_prev;
   
curr_tf = eye(4);
trajectory_xyz = zeros(3, timesteps);
trajectory_eul = zeros(3, timesteps);

for i = 1:timesteps
    dat_imu = [dat_wz(i); dat_wy(i); dat_wx(i); dat_ax(i); dat_ay(i); dat_az(i)];
       
    % Prediction
    Fk = F_func([x_prev; dat_imu; dat_dt(i)]);
    Juk = Ju_func([x_prev; dat_imu; dat_dt(i)]);
    Qk = Juk * cov_imu * Juk.' + Jb * cov_bias * Jb.';
    
    xk_pred = f_func([x_prev; dat_imu; dat_dt(i)]);
    Pk_pred = Fk * P_prev * Fk.' + Qk;
    
    % Update
    dat_fc = [dat_dx(i); dat_dy(i); dat_dz(i); dat_dyaw(i); dat_dpitch(i); dat_droll(i)];
    yk = dat_fc - H * xk_pred;
    Sk = H * Pk_pred * H.' + cov_fc;
    Kk = Pk_pred * H.' * inv(Sk);
    xk_est = xk_pred + Kk * yk;
    Pk_est = (eye(17, 17) - Kk * H) * Pk_pred;
    
    % log results
    x_est_log(:,i) = xk_est;
    x_pred_log(:,i) = xk_pred;
    P_est_log(:,:,i) =  Pk_est;
    P_pred_log(:,:,i) =  Pk_pred;
    
    delta_tf = eye(4, 4);
    delta_tf(1:3, 1:3) = eul2rotm(xk_est(12:14)');
    delta_tf(1:3, 4) = xk_est(1:3);
    curr_tf = curr_tf * delta_tf;
    trajectory_xyz(:,i) = curr_tf(1:3, 4);
    trajectory_eul(:,i) = rotm2eul(curr_tf(1:3,1:3));
    
    % Prep for the next time step
    x_prev = xk_est;
    P_prev = Pk_est;
    
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

disp('saving figures...')

for i=1:25
    saveas(i, strcat('/home/cs4li/Dev/end_to_end_odometry/results/ekf_debug_matlab/', strcat(num2str(i), '.png')))
    saveas(i, strcat('/home/cs4li/Dev/end_to_end_odometry/results/ekf_debug_matlab/', strcat(num2str(i), '.fig')))
end
