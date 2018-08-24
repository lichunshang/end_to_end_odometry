function R = R_q(q)

qw = q(1);
qx = q(2);
qy = q(3);
qz = q(4);


R = [qw^2 + qx^2 - qy^2 - qz^2,    2 * (qx*qy - qw*qz),          2 * (qx*qz + qw*qy);
     2 * (qx*qy + qw*qz),          qw^2 - qx^2 + qy^2 - qz^2,    2 * (qy*qz - qw*qx);
     2 * (qx*qz - qw*qy),          2 * (qy*qz + qw*qx),          qw^2 - qx^2 - qy^2 + qz^2];
end
