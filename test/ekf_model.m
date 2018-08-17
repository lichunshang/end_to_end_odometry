syms fcx fcy fcz fcyaw fcpitch fcroll dt wx wy wz ax ay az %Measurement & controlt
syms dx dy dz vx vy vz gpitch groll bax bay baz dyaw dpitch droll bwx bwy bwz % prev ts state

p = [dx dy dz].';
v = [vx vy vz].';
a = [ax ay az].';
w = [wz wy wx].';
gtheta = [gpitch groll].';
ba = [bax bay baz].';
bg = [bwz bwy bwx].';
theta = [dyaw dpitch droll].';
g = [0 0 -9.80665].';

u = [wz wy wx ax ay az].';
x = [p; v; gtheta; ba; theta; bg];

p_t = R(dt*(w-bg))*dt*v + (dt^2/2)*(R2(gtheta + dt*(w(2:3)-bg(2:3)))*g + a - ba);
v_t = R(dt*(w-bg))*v + dt*(R2(gtheta + dt*(w(2:3)-bg(2:3)))*g + a - ba);
gtheta_t = gtheta + dt*(w(2:3)-bg(2:3));
ba_t = ba;
theta_t = dt*(w-bg);
bg_t = bg;

f = [p_t; v_t; gtheta_t; ba_t; theta_t; bg_t];
F = jacobian(f, x);

h = [dx dy dz dyaw dpitch droll].';
H = eval(jacobian(h, x));

Ju = jacobian(f, u);
Jb = zeros(17, 6);
Jb(9:11,4:6) = eye(3,3);
Jb(15:17,1:3) = eye(3,3);

% Convert to matlab functions for faster processing
F_func = matlabFunction(F, 'Vars', {[x; u; dt]});
Ju_func = matlabFunction(Ju, 'Vars', {[x; u; dt]});
f_func = matlabFunction(f, 'Var', {[x; u; dt]});

function m = R(x)
yaw = x(1);
pitch = x(2);
roll = x(3);
cy = cos(yaw);
sy = sin(yaw);
cp = cos(pitch);
sp = sin(pitch);
cr = cos(roll);
sr = sin(roll);

m = [cy * cp,    cy * sp * sr - sy * cr,    cy * sp * cr + sy * sr;
     sy * cy,    sy * sp * sr + cy * cr,    sy * sp * sr + cy * sr;
         -sp,                   cp * sr,    cp * cr];
     
end

function m = R2(x)
m = R([0, x(1), x(2)]);
% m = eye(3,3);
end