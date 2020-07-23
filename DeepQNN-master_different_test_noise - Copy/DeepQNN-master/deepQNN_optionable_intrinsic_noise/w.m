function psi = w(m, phi)
% w creates w states with relative phases
%
% in: 
% m: number of qubits
% phi (optional, default = 0): array of relative phases
% out:
% psi: psi(:,j) is the state (basis00...01 + sum(exp(i*phi(j))) of the rest of basis states) / sqrt(m) 
%      in the tensor product basis

if nargin < 2
    phi = 0;
end

psi = zeros(2^m, length(phi));
psi(2,:) = 1;
if phi==0
    for ii=1:m-1
        psi(2^ii+1,:) = 1;
    end
else    
    for ii=1:m-1
        psi(2^ii+1,:) = exp(1i*phi(ii,:));
    end
end
psi = psi / sqrt(m);
end
