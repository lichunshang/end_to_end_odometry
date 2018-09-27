function q = q_v(v)
phi = sqrt(v(1)^2 + v(2)^2 + v(3)^2);
u = [v(1) v(2) v(3)].' / phi;
q = [cos(phi/2); u * sin(phi/2)];
end