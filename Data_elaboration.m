%% SECTION FOR ROBOT STATES
clear, clc
% Inside the Test_(i) files: time in [s], PWM left and right, Angular Position
% from encoders in [deg] and Angular Velocity from encoders in [rad/s]

load('Test_1.mat')
% load('Test_2.mat')
% load('Test_3.mat')
% load('Test_4.mat')

r = 0.019;              % [m] radius   (between 0.0215 and 0.018)
B = 0.185;              % [m] distance between tracks

V = r/2*(omega_r+omega_l);
psidot = r/B*(omega_r-omega_l);

psi = zeros(length(t),1);
VX = zeros(length(t),1);
VY = zeros(length(t),1);
X = zeros(length(t),1);
Y = zeros(length(t),1);

for i=1:length(t)-1
    psi(i+1) = psi(i)+(t(i+1)-t(i))*psidot(i);
    if psi(i+1)>pi
        psi(i+1) = psi(i+1)-2*pi;
    end
    if psi(i+1)<-pi
        psi(i+1) = 2*pi+psi(i+1);
    end 
end

for i=1:length(t)-1
    M=[cos(psi(i)) -sin(psi(i)); sin(psi(i)) cos(psi(i))];
    velocity=M*[V(i) 0]';
    VX(i)=velocity(1);
    VY(i)=velocity(2);
end

for i=1:length(t)-1
    X(i+1) = X(i)+(t(i+1)-t(i))*VX(i);
end

for i=1:length(t)-1
    Y(i+1) = Y(i)+(t(i+1)-t(i))*VY(i);
end

% PWM
figure()
plot(t,pwm_l,'LineWidth',1.25)
grid on
hold on
plot(t,pwm_r,'LineWidth',1.25)
xlabel('Time [s]')
ylabel('PWM [\mus]')
sgtitle('PWM')
legend('PWM_l','PWM_r')

figure()
plot(t,omega_l,'LineWidth',1.25)
hold on, grid on
plot(t, omega_r,'LineWidth',1.25)
xlabel('Time [s]')
ylabel('Angular velocity [rad/s]')
legend('Left','Right')
title('Angular velocity')

% Space
figure()
plot(X,Y,'LineWidth',1.25,'Color',[0, 0.5 , 0.9])
grid on
hold on
plot(X(1), Y(1),'or')
plot(X(end), Y(end),'og')
title('Position')
xlabel('X [m]')
ylabel('Y [m]')

figure()
plot(t,X,'LineWidth',1.25,'Color',[0.7, 0 ,0])
grid on
title('X position')
xlabel('Time [s]')
ylabel('X [m]')

figure()
plot(t,Y,'LineWidth',1.25,'Color',[0, 0.7 ,0])
title('Y position')
xlabel('Time [s]')
ylabel('Y [m]')
grid on

% Speed
figure()
plot(t,VX,'LineWidth',1.25)
grid on
hold on
plot(t,VY,'LineWidth',1.25)
xlabel('Time [s]')
ylabel('V [m/s]')
sgtitle('Speed')
legend('VX','VY')
ylim([-0.3 0.5])

figure()
plot(t,psidot,'LineWidth',1.25,'Color',[1 0.5 0])
grid on
xlabel('Time [s]')
ylabel('\psidot [rad/s]')
sgtitle('Angular speed')
ylim([-2 2.5])

% Orientation
figure()
plot(t,psi*180/pi,'LineWidth',1.25,'Color',[0 0.7 0])
grid on
xlabel('Time [s]')
ylabel('\psi [deg]')
title('Angular position')
ylim([-180 180])


