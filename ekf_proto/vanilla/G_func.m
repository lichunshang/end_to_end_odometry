function G = G_func(xk_est)
s_dtheta = xk_est(7:9);

G = eye(15, 15);
G(7:9, 7:9) = eye(3,3) - skew(0.5 * s_dtheta);

end
