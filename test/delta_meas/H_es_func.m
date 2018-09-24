function H_es = H_es_func(xk_nom_pred)
ns_q = xk_nom_pred(7:10);
ns_qw = ns_q(1);
ns_qx = ns_q(2);
ns_qy = ns_q(3);
ns_qz = ns_q(4);

H_es = zeros(16, 15);
H_es(7:10,7:9) = [-ns_qx/2, -ns_qy/2, -ns_qz/2;
                   ns_qw/2, -ns_qz/2,  ns_qy/2;
                   ns_qz/2,  ns_qw/2, -ns_qx/2;
                  -ns_qy/2,  ns_qx/2,  ns_qw/2];
H_es(1:6, 1:6) = eye(6,6);
H_es(11:16,10:15) = eye(6,6);

end
