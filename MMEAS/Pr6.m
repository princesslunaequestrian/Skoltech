n = 8;
w = exp(-1i * 2 * pi / n);
E4 = eye(4);
D4 = diag([1 w w ^ 2 w ^ 3]);
F4 = [1 1 1 1;
      1 -i -1 i;
      1 -1 1 -1;
      1 i -1 -i];
P8 = [1 0 0 0 0 0 0 0;
      0 0 1 0 0 0 0 0;
      0 0 0 0 1 0 0 0;
      0 0 0 0 0 0 1 0;
      0 1 0 0 0 0 0 0;
      0 0 0 1 0 0 0 0;
      0 0 0 0 0 1 0 0;
      0 0 0 0 0 0 0 1];
F8 = [E4 D4;
      E4 -D4] * [F4 zeros(4);
                 zeros(4) F4] * P8;
% disp(F8);

f = zeros(8,1);
disp(f)
f(1) = 1;
disp(f)
res = F8 * f;
power1 = res.*conj(res);
% plot((abs(res)).^2);
% title('Fourier Transform of $\hat{f}$','interpreter','latex');
% xlabel('x');
% ylabel('y');

f = zeros(8,1);
f(1:2) = 1;
disp(f)
res = F8 * f;
power2 = res.*conj(res);

f = zeros(8,1);
f(1:3) = 1;
disp(f)
res = F8 * f;
% plot((abs(res)).^2);
title('Fourier Transform of $\hat{f}$','interpreter','latex');
xlabel('x');
ylabel('y');

power3 = res.*conj(res);