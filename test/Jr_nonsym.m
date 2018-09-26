function Jr_inv = Jr_nonsym(v)

phi = sqrt(v(1)^2 + v(2)^2 + v(3)^2);

if abs(phi) > 1e-12
    a = [v(1) v(2) v(3)].' / phi;
    Jr_inv = (sin(phi) / phi) * eye(3,3) + (1 - (sin(phi) / phi)) * (a * a.') - ((1-cos(phi)) / phi)*skew(a);
else
    Jr_inv = eye(3,3);
end

end
