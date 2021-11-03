%% GPNARX devastator
clear
% load('Test_1.mat')
% load('Test_2.mat')
% load('Test_3.mat')
% load('Test_4.mat')
% load('test_5.mat')
% load('test_7.mat')
load('test_8.mat')
% load('test_9.mat')
% load('test_10.mat')

%%
y = omega_l/20;
u = pwm_l/20000;
delay_y = 3;
delay_u = 1;
delay_max = max(delay_y, delay_u);
x = zeros(size(u,1) - delay_max,delay_y+delay_u); % Training data

for ii = 1:size(y,1) - delay_max
    % Training points for the model
    x(ii,:) = [y(delay_y+ii-1:-1:ii)', u(delay_u+ii-1:-1:ii)'];
end

y = y(delay_y+1:end);
t_y = t(delay_y+1:end);

gpr = fitrgp(x, y);

% Predict on train data
[ypred_train, ysd_train, yint_train] = predict(gpr, x);

figure
hold on
grid on
% scatter(x,y_observed2,'xr') % Observed data points
% fplot(@(x) x.*sin(x),[0,10],'--r')   % Function plot of x*sin(x)
scatter(t_y, ypred_train,'g')                   % GPR predictions
plot(t_y, y, 'r');
plot(t_y, ypred_train, 'g');
patch([t_y;flipud(t_y)],[yint_train(:,1);flipud(yint_train(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction', 'Real data');
title('Validation on TRAINING dataset');
xlabel('Time [s]');
ylabel('Angular speed [rad/s]')
hold off

%%
% load('Test_1.mat')
% load('Test_2.mat')
load('Test_3.mat')
% load('Test_4.mat')
% load('test_5.mat')
% load('test_7.mat')
% load('test_8.mat')
% load('test_9.mat')
% load('test_10.mat')

omega_l = omega_l(1:1:60)/20;
pwm_l = pwm_l(1:1:60)/20000;
t = t(1:1:60);

xs = zeros(size(pwm_l,1) - delay_max,delay_y+delay_u);
ts = t(delay_y+1:end);

for ii = 1:size(omega_l,1)-delay_max
    % Test points 
    xs(ii,:) = [omega_l(delay_y+ii-1:-1:ii,1)', pwm_l(delay_u+ii-1:-1:ii)'];
end

% But what if I use a ndgrid?? Worst results ever...

% x_int = linspace(0, 1, 5);
% [ x1, x2, x3, x4 ] = ndgrid(x_int, x_int, x_int, x_int);
% xs = [ x1(:), x2(:), x3(:), x4(:) ];
% ts = [1:size(xs,1)]';

ys = omega_l(delay_y+1:end,1);

% xs = xs(slct,:);
% ts = ts(slct);

% ts = 0:0.05:20
% xs = zeros(1, size(ts,1));
% xs(20:end) = -50;
[ypred, ysd, yint] = predict(gpr, xs);

rmse = sqrt(sum((ypred - ys).^2)/length(ys));

figure
hold on
grid on
% scatter(x,y_observed2,'xr') % Observed data points
% fplot(@(x) x.*sin(x),[0,10],'--r')   % Function plot of x*sin(x)
scatter(ts, ypred,'g')                   % GPR predictions
plot(ts, ys, 'r');
plot(ts, ypred, 'g');
patch([ts;flipud(ts)],[yint(:,1);flipud(yint(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
title('Validation');
legend('Predicted', 'Real');
xlabel('Time [s]');
ylabel('Angular speed [rad/s]')
hold off

figure
title('Variance of y');
hold on, grid on
plot(ts, ysd.^2);
xlabel('Time [s]');
ylabel('Variance [rad²/s²]')

%% Adding recursion... 
% mu_g_old = ypred;
% Cg_old = diag(ysd);
% ypred = 0;
% clear
% As a basis vector, I would use the whole test dataset

% load('Test_1.mat')
% load('Test_2.mat')
load('Test_3.mat')
% load('Test_4.mat')
% load('test_5.mat')
% load('test_7.mat')
% load('test_8.mat')
% load('test_9.mat')
% load('test_10.mat')

% Normalize
omega_l = omega_l/20;
pwm_l = pwm_l/20000;

% figure
% plot(t, omega_l);
% figure
% plot(t, pwm_l);
% return

% omega_l = omega_l(1:2:60);
% pwm_l = pwm_l(1:2:60);
% t = t(1:2:60);

% Sys Id parameters
delay_y = 2;%3;
delay_u = 2;%1;
delay_max = max(delay_y, delay_u);

% Building basis vector
basisVector = zeros(size(pwm_l,1) - delay_max,delay_y+delay_u);
ts = t(delay_y+1:end);

for ii = 1:size(omega_l,1)-delay_max
    % Basis vector points 
    basisVector(ii,:) = [omega_l(delay_y+ii-1:-1:ii,1)', pwm_l(delay_u+ii-1:-1:ii)'];
end

y_basis = omega_l(delay_y+1:end,1);

% But what if I use a ndgrid??

x_int = linspace(0, 0.5, 5);
[ x1, x2, x3, x4 ] = ndgrid(x_int, x_int, x_int, x_int);
basisVector = [ x1(:), x2(:), x3(:), x4(:) ];
ts = 1:length(basisVector);

% Building batch vector

% load('Test_1.mat')
% load('Test_2.mat')
% load('Test_3.mat')
% load('Test_4.mat')
% load('test_5.mat')
% load('test_7.mat')
% load('test_8.mat')
% load('test_9.mat')
% omegal1 = omega_l;
% pwm_l1 = pwm_l;
% t1 = t;
load('test_10.mat')
% omega_l = [omegal1; omega_l];
% pwm_l = [pwm_l1; pwm_l];
% t = [ t1; t ];

% Normalize
omega_l = omega_l/20;
pwm_l = pwm_l/20000;

batch_size = 20;
% batchVector = zeros(ceil((size(pwm_l,1) - delay_max)/batch_size), batch_size, delay_y+delay_u);
batchVector = zeros(batch_size, delay_y+delay_u);

% this if the batch is one
% for ii = 1:size(omega_l,1)-delay_max
%     batchVector(ii,:) = [omega_l(delay_y+ii-1:-1:ii,1)', pwm_l(delay_u+ii-1:-1:ii)'];
% end


% jj = 1; 
% idx = 1;
% for ii = 1:size(omega_l,1)-delay_max
%     % Basis vector points 
% %     batchVector(jj,idx,:) = [omega_l(delay_y+ii-1:-1:ii,1)', pwm_l(delay_u+ii-1:-1:ii)'];
%     batchVector(idx,:) = [omega_l(delay_y+ii-1:-1:ii,1)', pwm_l(delay_u+ii-1:-1:ii)'];
%     idx = idx + 1;
%     if mod(ii, batch_size) == 0
%         batchVector2{jj} = batchVector;
%         jj = jj + 1;
%         idx = 1;
%     end
% end

ii = 1;
jj = 1;
flag = true;
while flag == true

%     batchVector = zeros(batch_size, delay_y+delay_u);
    for idx = 1:batch_size
        try
            batchVector(idx,:) = [omega_l(delay_y+ii-1:-1:ii,1)', pwm_l(delay_u+ii-1:-1:ii)'];
%             batchVector(jj,:) = [omega_l(delay_y+ii-1:-1:ii,1)', pwm_l(delay_u+ii-1:-1:ii)'];
            ii = ii + 1;
        catch
            flag = false;
            break;
        end
    end
    batchVector2{jj} = batchVector;
    jj = jj + 1;
end

% Randomizing batches
rndjj = randperm(jj-1);
for kk = 1:length(rndjj)
    batchVector2{kk} = batchVector2{rndjj(kk)};
end

ys = zeros(batch_size*size(batchVector2,2),1);
% ys = zeros(batch_size*size(batchVector,1),1);

ys(1:length(omega_l(delay_y+1:end,1))) = omega_l(delay_y+1:end,1);
ys = reshape(ys, batch_size, []);


% GP parameters
% l = 1.06e4;
% l = 0.8899;
l = 5;
l = [2 2 2 0.8];
% l = 5.6067e3;
sigm = 0.8;
mu_g_old = zeros(length(ts),1); % Initial condition on ypred
% mu_g_old = zeros(size(basisVector, 1),1);
% mu_g_old(1:length(ypred)) = ypred;
Cg_old = 10e2*eye(length(ts));
% Cg_old = 10e0*eye(size(basisVector, 1));
% Cg_old(1:length(ysd),1:length(ysd)) = diag(ysd);

% Recursion
K = kernelFnct(basisVector, basisVector, l, sigm);
K = K + 4e-4*ones(size(K,1)); % Adding noise to kernel
inv_K = inv(K);
figure
for ii = 1:size(batchVector2,2)-1
%     batch = reshape(batchVector(ii,:,:), [], delay_y+delay_u);
    batch = batchVector2{ii};
    % Provo a togliere gli zero... Ma non cambia nulla...
%     zro = any(batch,2);
%     batch = batch(zro,:);
%     Y = ys(zro,ii);
    Y = ys(:,ii);
    % Inference
%     Ks = kernelFnct(batchVector(ii,:), basisVector, l);
%     Kss = kernelFnct(batchVector(ii,:), batchVector(ii,:), l);
    Ks = kernelFnct(batch, basisVector, l, sigm);
    Kss = kernelFnct(batch, batch, l, sigm) + 1e-3*ones(size(batch,1));
    J = Ks*inv_K;
    mu_p = J*(mu_g_old - 0);
    B = Kss - J*Ks';
    Cp = B + J*Cg_old*J';
    % Update
    G = Cg_old*J'/(Cp + 1*ones(size(Cp))); % gain matrix, inversion of a matrix big as the batch used
    mu_g = mu_g_old + G*(Y - mu_p);
    Cg = Cg_old - G*J*Cg_old;
    mu_g_old = mu_g;
    Cg_old = Cg;
    
    % Plot
    subplot(2,1,1)
    cla
    hold on
    grid on
%     scatter(1:length(mu_g), mu_g,'g')                   % GPR predictions
    scatter(ts, mu_g,'g') 
%     plot(ts, y_basis, 'r');
%     plot(1:length(mu_g), mu_g, 'g');
    plot(ts, mu_g, 'g');
%     patch([ts;flipud(ts)],[yint(:,1);flipud(yint(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
    title('Validation');
    subtitle(['Batch provided: ', num2str(ii)]);
    legend('Predicted', 'Real');
    xlabel('Time [s]');
    ylabel('Angular speed [rad/s]')
%     drawnow
    hold off
    
    subplot(2,1,2)
    cla
    plot(1:length(mu_g), max(real(sqrt(diag(Cg))), 0));
    title('Standard deviation')
    drawnow
    pause(0.5);
end



function K = kernelFnct(x1, x2, l, sigm)

%     K = sigm*exp(-pdist2(x1, x2).^2/(2*l^2)); % Basis function (RBF kernel)
%     K = sigm*exp(-(x1 - x2)*l*(x1 - x2)');
    % Trying different kernels...
    % Matern 5/2
%     K = (1 + sqrt(5)/l*pdist2(x1, x2) + 5/(3*l)*pdist2(x1, x2).^2).*exp(-sqrt(5)/l*pdist2(x1, x2));
    % NN kernel
    K = zeros(size(x1,1),size(x2,1));
    for ii = 1:size(x1,1)
        for jj = 1:size(x2,1)
            K(ii,jj) = sigm*exp(-1/2*sum((x1(ii,:) - x2(jj,:)).^2.*l.^2));
%             denom = sqrt((1 + x1(ii,:)*x1(ii,:)'*l^(-2))*(1 + x2(jj,:)*x2(jj,:)'*l^(-2)));
%             Knn(ii,jj) = sigm*asin(x1(ii,:)*x2(jj,:)'*l^(-2)./denom);
        end
    end
%     denom = sqrt((1 + x1*x1'*l^(-2))*(1 + x2*x2*l^(-2)));
%     K = K + Knn*0;
    
end