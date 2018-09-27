function Qi = Qi_func(imu_covar, dt)

cov_a = imu_covar(1);
cov_w = imu_covar(2);
cov_ba = imu_covar(3);
cov_bw = imu_covar(4);

Qi = zeros(12, 12);
Qi(1:3,1:3) = cov_a * dt^2 * eye(3, 3);
Qi(4:6,4:6) = cov_w * dt^2 * eye(3, 3);
Qi(7:9,7:9) = cov_ba * dt * eye(3, 3);
Qi(10:12,10:12) = cov_bw * dt * eye(3, 3);


end
