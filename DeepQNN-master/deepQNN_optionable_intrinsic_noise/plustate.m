function psi = plustate(m, phi)
if nargin < 2
    phi = 0;
end

psi = ones(2^m, length(phi));
psi = psi / sqrt(2^m);
end

