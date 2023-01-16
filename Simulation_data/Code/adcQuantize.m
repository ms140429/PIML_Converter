function [sigQuantized,interv] = adcQuantize(sig, adcBit, lowerB, upperB)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Ts=0.0001;
% t=0:.00001:20*Ts;
% sig=sin(2000*pi*t)+cos(2000*pi*t);
interv=(upperB-lowerB)/(2^adcBit-1); %interval length for 8 levels resolution
u=upperB+interv;
partition = [lowerB:interv:upperB]; 
codebook = [lowerB:interv:u]; 
[index,sigQuantized] = quantiz(sig,partition,codebook); % Quantize.
sigQuantized=sigQuantized';
% plot([1:length(sig)],sig,'b.-',[1:length(sig)],sigQuantized,'r.-');
% legend('Original signal','Quantized signal');
end

