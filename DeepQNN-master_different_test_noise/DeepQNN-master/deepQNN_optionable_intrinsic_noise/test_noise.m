
function [test_in] = test_noise(mode, n, m)
global iii test_noise_flip_mode
dim = 2^m;
if mode == 1 
    p = iii/100;
    n0 = round(p*n);
    phi0 = zero(m, 0);
    flip0 = [0, 1; 1, 0];
    flip1 = [0, -1i; 1i, 0];
    flip2 = [1, 0; 0, -1];
    id = eye(2);
    flip = zeros(dim,dim,n);
    phi = zeros(dim,n); 
    
    if test_noise_flip_mode==1 %x-flip
        for i = 1:m 
            f = flip0;
            for j = 1:(i-1)
                f = kron(id, f);
            end
            for j = (i+1):m
                f = kron(f, id);
            end
            flip(:,:,i) = f;
            phi(:,i) = f*phi0;
        end
    end
    
    if test_noise_flip_mode==2 %y-flip
        for i = 1:m 
            f = flip1;
            for j = 1:(i-1)
                f = kron(id, f);
            end
            for j = (i+1):m
                f = kron(f, id);
            end
            flip(:,:,i) = f;
            phi(:,i) = f*phi0;
        end
    end
    
    if test_noise_flip_mode==3 %z-flip
        for i = 1:m  
            f = flip2;
            for j = 1:(i-1)
                f = kron(id, f);
            end
            for j = (i+1):m
                f = kron(f, id);
            end
            flip(:,:,i) = f;
            phi(:,i) = f*phi0;
        end
    end
    % phi_rand = phi(:,randi(m,1,3*n));
    % phi_in = phi_rand(:,1:n);
    % phi_out = phi_rand(:,n+1:2*n);
    % test_in = phi_rand(:,2*n+1:end);
    phi_rand = phi(:,randi(m,1,3*n0));
    rep_phi0 = repmat(phi0, 1, n-n0);
    %phi_in = cat(2, phi_rand(:,1:n0), rep_phi0);
    %phi_in = phi_in(:,randi(n,1,n));
    %phi_out = cat(2, phi_rand(:,n0+1:2*n0), rep_phi0);
    %phi_out = phi_out(:,randi(n,1,n));
    test_in = cat(2, phi_rand(:,2*n0+1:end), rep_phi0);
    %test_out = repmat(phi0, 1, n);
end
if mode == 2
    ghz_plus = ghz(m,0);
    ghz_minus = ghz(m,pi);
    phi_plus = repmat(ghz_plus, 1, floor(n/2));
    phi_minus = repmat(ghz_minus, 1, ceil(n/2));
    phi_in = cat(2, phi_plus, phi_minus);
    phi_out = phi_in;
    test_in = cat(2, ghz_plus, ghz_minus);
    test_out = test_in;
end
if mode == 3
    sgm = 0.5;
    phi=[];
    if m==1
        phi=0;
    else
        for i=1:m-1
            phi=[phi;normrnd(0, sgm, [1,3*n])];
        end
    end
    phi_rand = repmat(exp(1i*normrnd(0, sgm, [1,3*n])),dim,1).*w(m,phi);
    phi_in = phi_rand(:,1:n);
    phi_out = phi_rand(:,n+1:2*n);
    test_in = phi_rand(:,2*n+1:end);
    test_out = repmat(w(m,0), 1, n); 
end
end
