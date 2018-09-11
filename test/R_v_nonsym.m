function m = R_v_nonsym(v)
phi = sqrt(v(1)^2 + v(2)^2 + v(3)^2);

if abs(phi) > 1e-15 
    u = [v(1) v(2) v(3)].' / phi;
    m = eye(3,3) + sin(phi)*skew(u) + (1-cos(phi))*skew(u)^2;
else
    m = eye(3,3);
end
end
