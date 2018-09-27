function x_corr = f_nom_corr_func(x_nom, x_es)

x_corr = zeros(16, 1);
x_corr(1:6) = x_nom(1:6) + x_es(1:6);
x_corr(11:16) = x_nom(11:16) + x_es(10:15);
x_corr(7:10) = mul_q(x_nom(7:10), q_v_nonsym(x_es(7:9)));

end
