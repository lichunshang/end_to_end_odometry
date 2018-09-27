function theta = log_map_nonsym(R)


phi = acos((trace(R) - 1) / 2);

if abs(phi) > 1e-15
    u = unskew(R - R.') / (2*sin(phi));
    theta = phi * u;
else
    theta = [0 0 0]';
end


end