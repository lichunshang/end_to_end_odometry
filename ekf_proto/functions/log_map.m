function theta = log_map(R)
phi = acos((trace(R) - 1) / 2);
u = unskew(R - R.') / (2*sin(phi));
theta = phi * u;

end