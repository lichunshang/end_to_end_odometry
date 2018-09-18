function Fx = Fx_func(x_nom_prev, imu, dt)
am = imu(1:3);
% am(3) = am(3) - 9.806649999999999;
wm = imu(4:6);

ns_p = x_nom_prev(1:3);
ns_v = x_nom_prev(4:6);
ns_q = x_nom_prev(7:10);
ns_ba = x_nom_prev(11:13);
ns_bw = x_nom_prev(14:16);

Fx = zeros(15,15);

Fx(1:3,1:3) = eye(3,3);
Fx(1:3,4:6) = eye(3,3) * dt;

Fx(4:6,4:6) = eye(3,3);
Fx(4:6,7:9) = -R_quat(ns_q) * skew(am - ns_ba) * dt;
Fx(4:6,10:12) = -R_quat(ns_q) * dt;

Fx(7:9,7:9) = R_v_nonsym((wm - ns_bw)*dt).';
Fx(7:9,13:15) = -eye(3,3)*dt;

Fx(10:15,10:15) = eye(6, 6);

% % Euler's method
% F = zeros(15,15);
% F(1:3,4:6) = eye(3,3);
% 
% F(4:6,7:9) = -R_quat(ns_q) * skew(am - ns_ba);
% F(4:6,10:12) = -R_quat(ns_q);
% 
% F(7:9,7:9) = -1 * skew(wm - ns_bw);
% F(7:9,13:15) = -eye(3,3);
% 
% F_dt = eye(15, 15) + F * dt;
% 
% Fx = F_dt
end
