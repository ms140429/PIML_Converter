
% ==================================================================================================
% Author: Shuai Zhao @ Aalborg University, szh@energy.aau.dk
% Note:
%      * The code and data accompany the paper:
%        S. Zhao, Y. Peng, Y. Zhang, and H. Wang, "Parameter Estimation of Power Electronic
%        Converters with Physics-informed Machine Learning", IEEE Trans. Power Electronics, 2022
%      * This is the code for simulation data generation. Please first run
%        the "BuckConverter.m" for voltage & current generation, and then "dataGeneration.m" for
%        preparating the format of following python code "PIML_Converter.py".
% ==================================================================================================

clear all; close all; clc

%% System parameter setting
Uin=48;
R=8;  %10.43-4.85
L=7.25e-4;
C=1.645e-4;
Rc=0.201;
Rl=0.314;
Ron=0.221;
Vd=1;

%% Initialize
Il=0;Uo=0;Uc=0;Uref=24;

Ve=0;Vc=0;Vcc=0;Ve1=0;Vc1=0;

Kp=0.003;
Ki=20;
T=1/2e4;
dt=1e-7;  %0.1us step size

% determine the workload changing time
loadPhy1 = 10009;loadD1 =(loadPhy1-1)*10+1+50000;

loadPhy2 = 20009;loadD2 =(loadPhy2-1)*10+1+50000;

loadPhy3 = 30009;loadD3 =(loadPhy3-1)*10+1+50000;

%% differential equations of Buck converter
% please refer to the below paper for the equations. This code was also
% used for the data generation part in the below paper:
% [*] Y. Peng, S. Zhao and H. Wang, "A Digital Twin Based Estimation 
%     Method for Health Indicators of DC–DC Converters," in IEEE Trans. 
%     on Power Electronics, vol. 36, no. 2, pp. 2105-2118, Feb. 2021.

a1=Uin/L;
b1=(Rc*R/(R+Rc)+Ron+Rl)/L;
c1=R/(R+Rc)*1/L;
a2=R/(R+Rc)*1/C;
b2=1/(R+Rc)*1/C;

a3=(Rc*R/(R+Rc)+Rl)/L;
b3=R/(R+Rc)*1/L;
c3=Vd/L;
a4=R/(R+Rc)*1/C;
b4=1/(R+Rc)*1/C;

n=0:1e-7:0.08;
yt=(sawtooth(2*pi*2e4*n,0.5)+1)/2;

Rload=zeros(length(n),1);
for i=1:length(n)
    Rload(i)=R;
    if(i==loadD1)
        R=10.2;  
        Rload(i)=R;
        a1=Uin/L;
        b1=(Rc*R/(R+Rc)+Ron+Rl)/L;
        c1=R/(R+Rc)*1/L;
        a2=R/(R+Rc)*1/C;
        b2=1/(R+Rc)*1/C;

        a3=(Rc*R/(R+Rc)+Rl)/L;
        b3=R/(R+Rc)*1/L;
        c3=Vd/L;
        a4=R/(R+Rc)*1/C;
        b4=1/(R+Rc)*1/C;
    end

    if(i==loadD2)
        R=6.1;
        Rload(i)=R;
        a1=Uin/L;
        b1=(Rc*R/(R+Rc)+Ron+Rl)/L;
        c1=R/(R+Rc)*1/L;
        a2=R/(R+Rc)*1/C;
        b2=1/(R+Rc)*1/C;

        a3=(Rc*R/(R+Rc)+Rl)/L;
        b3=R/(R+Rc)*1/L;
        c3=Vd/L;
        a4=R/(R+Rc)*1/C;
        b4=1/(R+Rc)*1/C;
    end

    if(i==loadD3)
        R=3.1;
        Rload(i)=R;
        a1=Uin/L;
        b1=(Rc*R/(R+Rc)+Ron+Rl)/L;
        c1=R/(R+Rc)*1/L;
        a2=R/(R+Rc)*1/C;
        b2=1/(R+Rc)*1/C;

        a3=(Rc*R/(R+Rc)+Rl)/L;
        b3=R/(R+Rc)*1/L;
        c3=Vd/L;
        a4=R/(R+Rc)*1/C;
        b4=1/(R+Rc)*1/C;
    end

    Vcbuff(i)=Vcc;
    if(yt(i)>0.9999)
        Vc1=Vc;
        Ve1=Ve;
        Ve=Uref-Uo;
        Vc=Vc1+Kp*(Ve-Ve1)+Ki*T*Ve;

        if(Vc<0.1)
            Vc=0.1;
        end
        if(Vc>0.9)
            Vc=0.9;
        end
        Vcc=Vc;
    end

    if(Vcc>=yt(i))
        s(i)=1;
    end
    if(Vcc<yt(i))
        s(i)=0;
    end

    if(s(i)==1)

        Ka1=dt*(a1-b1*Il-c1*Uc);
        Kb1=dt*(a2*Il-b2*Uc);

        Ka2=dt*(a1-b1*(Il+Ka1/2)-c1*(Uc+Kb1/2));
        Kb2=dt*(a2*(Il+Ka1/2)-b2*(Uc+Kb1/2));

        Ka3=dt*(a1-b1*(Il+Ka2/2)-c1*(Uc+Kb2/2));
        Kb3=dt*(a2*(Il+Ka2/2)-b2*(Uc+Kb2/2));

        Ka4=dt*(a1-b1*(Il+Ka3/2)-c1*(Uc+Kb3/2));
        Kb4=dt*(a2*(Il+Ka3/2)-b2*(Uc+Kb3/2));

    end

    if(s(i)==0)

        Ka1=dt*(-a3*Il-b3*Uc-c3);
        Kb1=dt*(a4*Il-b4*Uc);

        Ka2=dt*(-a3*(Il+Ka1/2)-b3*(Uc+Kb1/2)-c3);
        Kb2=dt*(a4*(Il+Ka1/2)-b4*(Uc+Kb1/2));

        Ka3=dt*(-a3*(Il+Ka2/2)-b3*(Uc+Kb2/2)-c3);
        Kb3=dt*(a4*(Il+Ka2/2)-b4*(Uc+Kb2/2));

        Ka4=dt*(-a3*(Il+Ka3/2)-b3*(Uc+Kb3/2)-c3);
        Kb4=dt*(a4*(Il+Ka3/2)-b4*(Uc+Kb3/2));

    end

    Il=Il+(Ka1+2*Ka2+2*Ka3+Ka4)/6;
    if(Il<0)
        Il=0;
    end
    Uc=Uc+(Kb1+2*Kb2+2*Kb3+Kb4)/6;
    Uo=(R/(R+Rc)*Il-Uc/(R+Rc))*Rc+Uc;
    Ill(i)=Il;
    Ucc(i)=Uc;
    Uoo(i)=Uo;
end


figure(1);subplot(2,1,1);hold on;
plot(n*1e3,Ill);
grid minor; xlabel('Time');ylabel('Inductor current (A)');

figure(1);subplot(2,1,2);
hold on;
plot(n*1e3,Uoo);
grid minor; xlabel('Time');ylabel('Output voltage (V)');

%% Data saving
% idxSelection=5e4:1:5e5;
idxSelection=1:1:5e5;
tout=n(idxSelection)';
IL=Ill(idxSelection)';
Dswitch=s(idxSelection)';
Vf=Vd*ones(length(tout),1);
Vo=Uoo(idxSelection)';
Rload=Rload(idxSelection);

SystemSettings.Vin=Uin;
SystemSettings.Rload=R;  
SystemSettings.L=L;
SystemSettings.C=C;
SystemSettings.RC=Rc;
SystemSettings.RL=Rl;
SystemSettings.Rdson=Ron;

tout=tout-tout(1);

save simulationDataset.mat tout IL Dswitch Vf Vo Rload SystemSettings



