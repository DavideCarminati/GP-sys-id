%% GPNARX devastator
clear, clc, close all

% The aim is to obtain the model identification of the Devastator Robot

% STEP 1: SISO model
% y(t) = f(x,u,t)
% with 
% u -> input  of PWM             [us]       in R^1
% y -> output of angular speed   [rad/s]    in R^1

% Experiments
% load('Test_1.mat')
% load('Test_2.mat')
% load('Test_3.mat')
% load('Test_4.mat')
% load('test_5.mat')
% load('test_7.mat')
% load('test_8.mat')
% load('test_9.mat')
load('test_10.mat')

% In every test,
% t       -> time                        [s]
% pwm_l   -> PWM values for Left Motor   [us]
% pwm_r   -> PWM values for Right Motor  [us]
% omega_l -> Angular speed for LM        [rad/s]
% omega_r -> Angular speed for RM        [rad/s]

%% DATA SELECTION

% Suppose to create a model for the LEFT side
% Output
y = omega_l./20;
% Input
u = pwm_l./20000;

% The selected model is a NARK model
% Input training vector
% x = [y(k-1) , y(k-2), ..., y(k-delay_y), u(k-1), u(k-2), ..., u(k-delay_u)]

% Selected delay: it is an arbitrary choise
delay_y = 3;
delay_u = 3;
delay_max = max(delay_y, delay_u);

% Creation of the input training vector 
% As many rows as the tests data (considering the delay)
% As many columns as the elements in the regressor at time k 
dim_t = size(u,1) - delay_max;
x = zeros(dim_t,delay_y+delay_u); 

% Training points for the model: 
% First iteration
% -> x(1,:) = [y(3:-1:1), u(1:-1:1)] = [y(3) y(2) y(1) u(1)]
% Second iteration
% -> x(2,:) = [y(4:-1:2), u(2:-1:2)] = [y(4) y(3) y(2) u(2)]
for ii = 1:dim_t
    x(ii,:) = [y(delay_y+ii-1:-1:ii)', u(delay_u+ii-1:-1:ii)'];
end

% Output and time
% The last output is the one that it is possible to predict considering the chosen delay 
y = y(delay_y+1:end);
t_y = t(delay_y+1:end);

% TRAINING DATA
% x -> input of the model       in R^(dim_t,delay_y+delay_u)
% y -> output of the model      in R^(dim_t,1)

% READY to FIT the model!
% But first...
% HYPOTHESIS and THEORY

% A Gaussian process regression problem can be written as
%  f(x) = GP(m(x), k(x,x'))
%  yk = f(xk) + epsilon
%  GP Gaussian process of m(x) mean and k(x,x') covariance function or kernel 
%  epsilon an indipendent noise of 0 mean and sigma^2 variance

% It is a choise to select MEAN and COVARIANCE
% Mean usually set to 0
% Covariance chosen between different kernel
% Suppose to use a squared exponential covariance function as kernel
% k(x,x') = s^2 exp( -1/2 ||x-x'||^2/l^2)
% s^2 covariance of the regressor fuctions
% l length scales of the regressor functions

% The Gaussian process regression is concerned with the following problem: 
% Given a set of observed (training) input-output data D = {(xk, yk) : k = 1, . . . , N}
% from an unknown function y = f(x), predict the values of the function at new (test) inputs
% {xk*: k = 1, . . . , M}
% Fundamental equations of Gaussian process regression
% E[f(x*)|y]   = m(x*) + k(x*)(k+sigma^2I)^(-1)(y-m(x)) 
% Cov[f(x*)|y] = k(x*,x*) - k(x)(k+sigma^2)^(-1) k'(x*)

% The mean value represents the most likely output 
% The variance can be interpreted as the measure of its confidence

% f(x) can be modeled as a basis function expansion of the Gaussian process
% In this case coefficients beta are introduced


%% TRAINING 

% GP MODEL CREATION
%fitrgp   Fit a Gaussian Process Regression (GPR) model.
%         fitrgp(x,y) accepts
%         - X as an N-by-P matrix of predictors with
%                - one row per observation   
%                - one column per predictor
%         - Y is the response vector

% The RGP Model is characterized by:
% 'FitMethod': Method used to estimate the basis function coefficients, β;
%              noise standard deviation, σ; 
%              and kernel parameters, θ.
% 'BasisFunction': Explicit basis function used in the GPR model
% 'Beta': Estimated coefficients for the explicit basis functions
% 'Sigma': Estimated noise standard deviation of the GPR model
% 'LogLikelihood': Maximized marginal log likelihood of the GPR model
% 'KernelFunction': Form of the covariance function 
% 'KernelInformation': Information about the parameters of the kernel function
% 'PredictMethod': Method that predict uses to make predictions from the GPR model
% 'Alpha': Weights used to make predictions
% 'ActiveSetVectors: Subset of training data used to make predictions from the GPR model


%     Actually used options: 
%     - 'FitMethod'      -> 'Exact' 
%     - 'BasisFunction'  -> 'Constant', ie H=1
%     - 'KernelFunction' -> 'SquaredExponential'
%     - 'PredictMethod'  -> 'Exact'
      
gpr = fitrgp(x, y);

% GP MODEL obtained!
% Let's go to see how good it is

%% VALIDATION ON TRAINING

% Predict on train data
[ypred_train, ysd_train, yint_train] = predict(gpr, x);

figure
hold on
grid on
% scatter(x,y_observed2,'xr')                   % Observed data points
% fplot(@(x) x.*sin(x),[0,10],'--r')            % Function plot of x*sin(x)
scatter(t_y, ypred_train,'g')                   % GPR predictions
plot(t_y, y, 'r');
plot(t_y, ypred_train, 'g');
patch([t_y;flipud(t_y)],[yint_train(:,1);flipud(yint_train(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction', 'Real data');
title('Validation on TRAINING');
xlabel('Time [s]');
ylabel('Angular speed [rad/s]')
hold off

%%  VALIDATION ON TEST

% load('Test_1.mat')
% load('Test_2.mat')
% load('Test_3.mat')
% load('Test_4.mat')
% load('test_5.mat')
load('test_7.mat')
% load('test_8.mat')
% load('test_9.mat')
% load('test_10.mat')

% Data preparation
delay_max = max(delay_y, delay_u);
dim_v = size(pwm_l,1) - delay_max;
xs = zeros(dim_v,delay_y+delay_u);
ys = omega_l(delay_y+1:end,1);
ts = t(delay_y+1:end);

% Test points 
for ii = 1:dim_v
    xs(ii,:) = [omega_l(delay_y+ii-1:-1:ii,1)'./20, pwm_l(delay_u+ii-1:-1:ii)'./20000];
end

% Prediction
% -ypred    ->  predicted responses for the Gaussian process regression model
% -ysd      ->  standard deviation of the process
% -yint     ->  95% prediction intervals of the response variable
[ypred, ysd, yint] = predict(gpr, xs);

figure
hold on
grid on
scatter(ts, ypred*20,'g')                
plot(ts, ys, 'r');
plot(ts, ypred*20, 'g');
patch([ts;flipud(ts)],[yint(:,1);flipud(yint(:,2))].*20,'k','FaceAlpha',0.1); 
title('Validation');
legend('Predicted', 'Real');
xlabel('Time [s]');
ylabel('Angular speed [rad/s]')
hold off

figure
title('Variance of y');
hold on, grid on
plot(ts, (ysd*20).^2);
xlabel('Time [s]');
