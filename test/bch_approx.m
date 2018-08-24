function v3 = bch_approx(v1, v2)
% Baker–Campbell–Hausdorff formula
v3 = v1 + v2 + 0.5 * lb(v1,v2) + ... 
    (1.0/12) * (lb(v1,lb(v1,v2)) + lb(v2,lb(v2,v1))) - ...
    (1.0/24) * lb(v2,lb(v1,lb(v1,v2)));
end

% lie bracket for so(3)
function v3 = lb(v1, v2)
v3 = skew(v1) * v2;
end