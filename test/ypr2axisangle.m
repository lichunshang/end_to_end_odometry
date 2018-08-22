function theta = ypr2axisangle(euler)

R = eul2rotm(euler, 'ZYX');
phi = acos((trace(R) - 1) / 2);
u = unskew(R - R.') / (2*sin(phi));
theta = phi * u;

end