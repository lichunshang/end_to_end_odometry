function [z, H] = h_func(x_nom_prev, x_nom_pred, imu, dt)


imut_rot = (imu(4:6) - x_nom_prev(14:16))*dt;
R_T = R_q(x_nom_prev(7:10)).';

H = zeros(6, 15);
H(1:3, 1:3) = R_T;
H(4:6, 7:9) = Jr_inv_nonsym(imut_rot);

dp = R_T*(x_nom_pred(1:3) - x_nom_prev(1:3));
dtheta = imut_rot;


z = [dp; dtheta];

end
