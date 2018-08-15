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

data_size = size(data.data);
timesteps = data_size(1);

for i = 1:timesteps
    
end