function Jr_inv = Jr_inv_nonsym(v)

phi = sqrt(v(1)^2 + v(2)^2 + v(3)^2);

if abs(phi) > 1e-12
    a = [v(1) v(2) v(3)].' / phi;
    Jr_inv = (phi/2)*cot(phi/2) * eye(3,3) + (1 - (phi/2)*cot(phi/2)) * (a * a.') + (phi/2) * skew(a);
else
    Jr_inv = eye(3,3);
end

end

function cot_t = cot(theta)
cot_t = 1/tan(theta);
end