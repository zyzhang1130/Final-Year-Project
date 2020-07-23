U1=[-0.373-0.037i, -0.927 + 0.015i, 0; 0.008 + 0.213i, -0.013-0.085i, -0.003-0.973i; -0.017 + 0.902i, -0.035-0.363i, -0.012 + 0.230i];
U2=[0.054-0.610i, -0.790-0.042i, 0; -0.090 + 0.094i, -0.075-0.067i, 0.014-0.986i; 0.052 + 0.778i, -0.601 + 0.062i, 0.120 + 0.112i];
U3 =[0.495 + 0.298i, -0.795 + 0.185i, 0; -0.172 + 0.228i, 0.025 + 0.200i, 0.048-0.936i; -0.135 + 0.753i, 0.303 + 0.449i, 0.142 + 0.319i]; 

V1=[];
V2=[];
V3=[];
V11=[];
V22=[];
V33=[];
V4=[];
V=[];
for i=1:1000
v1=fun(1000*rand,1000*rand,1000*rand);
v2=U1*transpose(v1);
v3=v2;
absv2=abs(v2);
absv3=abs(v3);
U4=U2*U1;
v4=U4*transpose(v1);
V11=[V11;v1(1)];
V22=[V22;v1(2)];
V33=[V33;v1(3)];
V1=[V1;v3(1)];
V2=[V2;v3(2)];
V3=[V3;v3(3)];
V4=[V4;v1(3)];
end
% scatter3(abs(V1),abs(V2),abs(V3))
set(gca,'XLim',[-1 1])
set(gca,'YLim',[-1 1])
set(gca,'ZLim',[-1 1])
figure
%scatter3(abs(V11),abs(V22),abs(V33))
 stem(abs((V3)).^2)
% figure
% stem(abs(complex(V2)))
% figure
% stem(abs(real(V3)))
%histogram(V4);

function v1=fun(theta,phi,zeta)
v1=[cos(2*theta-phi)*cos(phi)-1i*sin(2*theta-phi)*sin(phi),(cos(2*theta-phi)*sin(phi)+1i*sin(2*theta-phi)*cos(phi))*cos(zeta),(cos(2*theta-phi)*sin(phi)+1i*sin(2*theta-phi)*cos(phi))*sin(zeta)];
end