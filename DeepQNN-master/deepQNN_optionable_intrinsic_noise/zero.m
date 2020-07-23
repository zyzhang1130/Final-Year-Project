function psi = zero(m, phi)
if nargin < 2
    phi = 0;
end

psi = zeros(2^m, length(phi));
psi(1,:) = 1;

end
