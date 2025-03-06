clear;
[y,fs] = audioread('Test-Data/Adhiraj_1.wav');
[y1,fs1]= audioread('Train-Data/Adhiraj-speech/Adhiraj-1.wav');
t = 0:1/fs:(length(y)-1)/fs;
t1 = 0:1/fs1:(length(y1)-1)/fs1;
subplot(4,2,1);
plot(t1,y1(:,1));
xlabel('time','FontWeight','bold');
ylabel('audio signal','FontWeight','bold');
title('Train Audio without noise');
grid on;
A = [6,4,2];
j = 3;

for i = A
    subplot(4,2,j);
    n = sqrt(10^(-i)).*randn(size(y1));
    y2 = y1 + n;
    plot(t1,y2(:,1));
    grid on;
    xlabel('time','FontWeight','bold');
    ylabel('audio signal','FontWeight','bold');
    title(strcat('Train Audio with noise of variance 10^{-',num2str(i),'}'));
    j = j+2;
end

subplot(4,2,2);
plot(t,y(:,1));
xlabel('time','FontWeight','bold');
ylabel('audio signal','FontWeight','bold');
title('Test Audio without noise');
grid on;
A = [6,4,2];
j = 4;

for i = A
    subplot(4,2,j);
    n = sqrt(10^(-i)).*randn(size(y));
    y2 = y + n;
    plot(t,y2(:,1));
    grid on;
    xlabel('time','FontWeight','bold');
    ylabel('audio signal','FontWeight','bold');
    title(strcat('Test Audio with noise of variance 10^{-',num2str(i),'}'));
    j = j+2;
end

    