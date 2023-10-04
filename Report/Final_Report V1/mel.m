clear;
f = 0:10000;
m = 2595*log10(1+f/700);
plot(f,m,'color','b','linewidth',2);
grid on;
xlabel('Frequency in Hz','FontSize',15,'FontWeight','bold');
ylabel('Mel-Frequency','FontSize',15,'FontWeight','bold');
title('Mel-frequency and Frequency(Hz) relationship','FontSize',15);
