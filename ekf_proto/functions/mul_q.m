function r = mul_q(p, q)
pw = p(1);
px = p(2);
py = p(3);
pz = p(4);

qw = q(1);
qx = q(2);
qy = q(3);
qz = q(4);

r = [pw*qw - px*qx - py*qy - pz*qz;
     pw*qx + px*qw + py*qz - pz*qy;
     pw*qy - px*qz + py*qw + pz*qx;
     pw*qz + px*qy - py*qx + pz*qw];
end