syms dx dy dz dyaw dpitch droll dt wx wy wz ax ay az %Measurement & controlt
syms vx vy vz gpitch groll bax bay baz yaw pitch roll bwx bwy bwz % prev ts state

v_tm1 = [vx vy vz];
a = [ax ay az];
w = [wz wy wx];
theta_g = [gpitch groll];
ba = [bax bay baz];
bg = [bgz bgy bgx];
theta = [duaw dpitch droll];
g = [0 0 -9.80665];


