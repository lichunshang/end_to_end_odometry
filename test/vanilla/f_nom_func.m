function x_nom = f_nom_func(x_nom_prev, imu, dt)
am = imu(1:3);
wm = imu(4:6);

ns_p = x_nom_prev(1:3);
ns_v = x_nom_prev(4:6);
ns_q = x_nom_prev(7:10);
ns_ba = x_nom_prev(11:13);
ns_bw = x_nom_prev(14:16);
ns_g = [0 0 -9.80665].';

ns_p_kp1 = ns_p + ns_v * dt + 0.5 * dt^2 * (R_quat(ns_q) * (am - ns_ba) + ns_g);
ns_v_kp1 = ns_v + dt * (R_quat(ns_q) * (am - ns_ba) + ns_g);
ns_q_kp1 = mul_q(ns_q, q_v_nonsym((wm - ns_bw)*dt));
ns_ba_kp1 = ns_ba;
ns_bw_kp1 = ns_bw;


x_nom = [ns_p_kp1; ns_v_kp1; ns_q_kp1; ns_ba_kp1; ns_bw_kp1];
end
