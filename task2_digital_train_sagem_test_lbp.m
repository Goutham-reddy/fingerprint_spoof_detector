clear
load('data_main.mat')

%LBP from digital is trained and LBP from sagem is tested

%Prepare LBP features data 
%Train_All_Data_LBP = horzcat(Train_All_Data_DigiLBP, Train_All_Data_SageLBP)
Train_All_Data_LBP = Train_All_Data_DigiLBP
Train_All_Data_LBP = num2cell(Train_All_Data_LBP, 1)
Train_All_Label_LBP = Train_All_Label_DigiLBP

Test_All_Data_LBP = Test_All_Data_SageLBP
Test_All_Label_LBP = Test_All_Label_SageLBP


%rng('default')
hiddenSize1 = 35;
autoenc1 = trainAutoencoder(Train_All_Data_LBP,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

%view(autoenc1)
feat1 = encode(autoenc1,Train_All_Data_LBP);


hiddenSize1 = 20;
autoenc2 = trainAutoencoder(feat1,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.10, ...
    'ScaleData', false);

%view(autoenc1)
feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,Train_All_Label_LBP,'MaxEpochs',400);
%view(softnet)

deepnet = stack(autoenc1,autoenc2, softnet);
view(deepnet)

% plot roc and confusion 
% For testing
%y = deepnet(Test_All_Data_LBP);
%figure(1)
%plotconfusion(Test_All_Label_LBP,y);
%figure(2)
%plotroc(Test_All_Label_LBP,y)

% For training
%y = deepnet(Train_All_Data_LBP);
%figure(3)
%plotconfusion(Train_All_Label_LBP,y);
%figure(4)
%plotroc(Train_All_Label_LBP,y)

% Perform fine tuning
inputSize = 54
xTrain = zeros(inputSize,numel(Train_All_Data_LBP));
for i = 1:numel(Train_All_Data_LBP)
    xTrain(:,i) = Train_All_Data_LBP{i}(:);
end


deepnet_bp = train(deepnet,xTrain,Train_All_Label_LBP);

% plot roc and confusion 
% For testing
y = deepnet_bp(Test_All_Data_LBP);
figure(5)
plotconfusion(Test_All_Label_LBP,y);
%figure(6)
%plotroc(Test_All_Label_LBP,y)
ezroc3(y,Test_All_Label_LBP,2,'',1);


% For training
Train_All_Data_LBP_rev = cell2mat(Train_All_Data_LBP)
y = deepnet_bp(Train_All_Data_LBP_rev);
figure(7)
plotconfusion(Train_All_Label_LBP,y);
%figure(8)
 %plotroc(Train_All_Label_LBP,y)
ezroc3(y,Train_All_Label_LBP,2,'',1);


save('data_task2_digital_train_sagem_test_lbp.mat')


