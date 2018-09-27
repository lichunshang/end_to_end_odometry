y = 1.2;
p = 0.5;
r = -2.3;

cy = cos(y);
sy = sin(y);
cp = cos(p);
sp = sin(p);
cr = cos(r);
sr = sin(r);

%MRPT
R1 = [cy * cp,    cy * sp * sr - sy * cr,    cy * sp * cr + sy * sr;
      sy * cy,    sy * sp * sr + cy * cr,    sy * sp * sr + cy * sr;
          -sp,                   cp * sr,    cp * cr];
      
% curr
sA = sin(y);
cA = cos(y);
sB = sin(p);
cB = cos(p);
sC = sin(r);
cC = cos(r);

      
R2 = [                cB * cA,                    cB * sA,         sB;
      -cC * sA - sC * sB * cA,     cC * cA - sC * sB * sA,    sC * cB;
       sC * sA - cC * sB * cA,     -sC * cA - cC* sB * sA,    cC * cB];
   
% wikipedia
c1 = cos(r);
s1 = sin(r);
c2 = cos(p);
s2 = sin(p);
c3 = cos(y);
s3 = sin(y);

R3 = []
 