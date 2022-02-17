clear variables;
clear figures;


h = pi/10;

[U, u_an, x] = numeric_calc(h);

figure(); hold on;
m1 = plot(x, U, 'c--', 'LineWidth', 1, 'DisplayName', 'Approximation');
m2 = plot(x, u_an, 'r', 'LineWidth', 1, 'DisplayName', 'Analytical Solution');
legend(); grid(); hold off;


H = 2*pi/1000:2*pi/10000:2*pi/10;

pM = zeros(length(H), 1);
RMS = zeros(length(H), 1);

for i = 1:length(H)
    disp(H(i));
    [U, u_an, x] = numeric_calc(H(i));
    pM(i) = max(abs(U - u_an));
    RMS(i) = rms(U - u_an);
end

figure(); hold on;
m3 = plot(H, pM, 'LineWidth', 1, 'DisplayName', 'Pointwise Maximum Error');
m4 = plot(H, RMS, 'LineWidth', 1, 'DisplayName', 'Root-mean-square Error');
legend(); grid(); hold off;



function [U, u_an, x] = numeric_calc(h)

    x = h:h:2*pi;
    steps = length(x);
    
    A = zeros(steps, steps);
    
    A(1, steps-1) = 1/36;
    A(1, steps) = -2/9;
    A(1, 1) = -11/9;
    A(1, 2) = 20/9;
    A(1, 3) = -37/36;
    A(1, 4) = 2/9;
    
    
    for i = 2:steps
        A(i, :) = circshift(A(i-1, :), 1);
    end
    
    g = zeros(steps, steps);
    
    for i = 1:steps
        g(i, i) = (cos(x(i)) / (2 + sin(x(i))));
    end
    
    A = A/h + g;
    
    f = zeros(steps, 1);
    
    for i = 1:steps
        x_i = x(i);
        f(i) = (cos(2*x_i) - 2*sin(x_i)) / (2 + sin(x_i));
    end
    
    U = A\f;
    
    u_an = zeros(steps, 1);
    for i = 1:steps
        x_i = x(i);
        u_an(i) = cos(x_i);
    end

end

