clear;
[y,fs] = audioread('Test-Data/Adhiraj_1.wav');
mel_high = 2595*log10(1+ fs/(2*700));
n_filt = 40;
mel = 0:(n_filt+2):mel_high;
hz = 700*(10.^(mel./2595)-1);
k = (512.*hz)./fs;
filt = zeros(length(2:length(mel)-1),3);
for i=2:length(mel)-1
    f = k(i-1);
    F = k(i+1);
    filt(i-1,1)=0;
    filt(i-1,2)=1;
    filt(i-1,3)=0;
end
plot(filt);
    