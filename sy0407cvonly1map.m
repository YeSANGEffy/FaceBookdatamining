clear;
clc;
addpath('matlab');
spec = csvread('dataset_fb_f4_nume_csv.txt');
% last column is label
Label_ori = spec(:,end);
Inst_ori = spec(:,1:8);
%inst_sparse = sparse(inst);
%spectf = (spec - repmat(min(spec,[],1),size(spec,1),1))*spdiags(1./(max(spec,[],1)-min(spec,[],1))',0,size(spec,2),size(spec,2));

% data preprocessing for normalization
Label = Label_ori';
Inst = Inst_ori';

% label normalization
[~,labelps] = mapminmax(Label);
labelps.ymin = 0;
labelps.ymax = 1;
[label,labelps] = mapminmax(Label,labelps);
label = label';

% inst normalization
[~,instps] = mapminmax(Inst);
instps.ymin = 0;
instps.ymax = 1;
[inst,instps] = mapminmax(Inst,instps);
inst = inst';

% transfer inst to meet the libsvm format require
%inst = fliplr(inst);
inst=inst(: , [end,1:end-1]);
inst_sparse = sparse(inst);
%%
%
ind = 300;
traindata = inst_sparse(1:ind,:);
trainlabel = label(1:ind,:);
testdata = inst_sparse(ind+1:end,:);
testlabel = label(ind+1:end,:);

%
% Find the optimize value of c,g paramter
% Approximately choose the parameters:
% the scale of c is 2^(-5),2^(-4),...,2^(10)
% the scale of g is 2^(-5),2^(-4),...,2^(5)
[bestmse,bestc,bestg] = svmregress(trainlabel,traindata,-5,10,-5,5,3,1,1,0.0005);

% Display the approximate result
disp('Display the approximate result');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

fprintf('\nFinish the first round tuning and begin the final round regression and print the final parameters.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


%Do training by using svmtrain of libsvm
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.001 -h 0 -t 2'];
%cmd = ['-c ',num2str(bestc), ' -g ', num2str(bestg) , ' -s 3'];
model = svmtrain(trainlabel,traindata,cmd);

%Do predicting by using svmpredict of libsvm
[Predict_ori, acc, dec] = svmpredict(testlabel,testdata,model);
%%
%
Predict = Predict_ori';
%
predict_fin = mapminmax('reverse',Predict,labelps);
%
predict_fin = predict_fin';
%
testlabel_fin = mapminmax('reverse',testlabel,labelps);
%% 
% compute MAPE
test_err = predict_fin-testlabel_fin; 
test_err_abs = abs(test_err);
test_errpct = abs(test_err)./testlabel_fin*100; 
test_MAPE = mean(test_errpct(~isinf(test_errpct)))

%
