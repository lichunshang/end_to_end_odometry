function q = q_v_nonsym(v)
phi = sqrt(v(1)^2 + v(2)^2 + v(3)^2);

if abs(phi) > 1e-15 
    u = [v(1) v(2) v(3)].' / phi;
    q = [cos(phi/2); u * sin(phi/2)];
else
    q = [1 0 0 0].';
end
end