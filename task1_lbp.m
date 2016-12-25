clear
load('data_main.mat')

%combine data from Digital and Sagen sensors

% combine LBP features data
Train_All_Data_LBP = horzcat(Train_All_Data_DigiLBP, Train_All_Data_SageLBP)
Train_All_Data_LBP = num2cell(Train_All_Data_LBP, 1)
Train_All_Label_LBP = horzcat(Train_All_Label_DigiLBP, Train_All_Label_SageLBP)

% Prepare validation and testing data
Gen_All_Data_LBP = horzcat(Test_All_Data_DigiLBP, Test_All_Data_SageLBP)
Gen_All_Label_LBP = horzcat(Test_All_Label_DigiLBP, Test_All_Label_SageLBP)

Val_All_Data_LBP  = Gen_All_Data_LBP(: , 1:2018)
Val_All_Label_LBP = Gen_All_Label_LBP(: , 1:2018)

Test_All_Data_LBP  = Gen_All_Data_LBP(: , 2019:4036)
Test_All_Label_LBP = Gen_All_Label_LBP(: , 2019:4036)

%rng('default')
hiddenSize1 = 800;
autoenc1 = trainAutoencoder(Train_All_Data_LBP,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.09, ...
    'ScaleData', false);

%view(autoenc1)
feat1 = encode(autoenc1,Train_All_Data_LBP);

hiddenSize1 = 400;

autoenc2 = trainAutoencoder(feat1,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.003, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.08, ...
    'ScaleData', false);

%view(autoenc2)
feat2 = encode(autoenc2,feat1);

hiddenSize1 = 200;

autoenc3 = trainAutoencoder(feat2,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.003, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.06, ...
    'ScaleData', false);

%view(autoenc3)
feat3 = encode(autoenc3,feat2);

hiddenSize1 = 100;

autoenc4 = trainAutoencoder(feat3,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.02, ...
    'ScaleData', false);

%view(autoenc4)
feat4 = encode(autoenc4,feat3);

softnet = trainSoftmaxLayer(feat4,Train_All_Label_LBP,'MaxEpochs',400);
%view(softnet)

deepnet = stack(autoenc1,autoenc2,autoenc3,autoenc4,softnet);
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
figure(1)
plotconfusion(Test_All_Label_LBP,y);
%figure(2)
%plotroc(Test_All_Label_LBP,y)
ezroc3(y,Test_All_Label_LBP,2,'',1);

% For validation
y = deepnet_bp(Val_All_Data_LBP);
figure(5)
plotconfusion(Val_All_Label_LBP,y);
%figure(6)
%plotroc(Val_All_Label_LBP,y)
ezroc3(y,Val_All_Label_LBP,2,'',1);

% For training
Train_All_Data_LBP_rev = cell2mat(Train_All_Data_LBP)
y = deepnet_bp(Train_All_Data_LBP_rev);
figure(7)
plotconfusion(Train_All_Label_LBP,y);
%figure(8)
%plotroc(Train_All_Label_LBP,y)

ezroc3(y,Train_All_Label_LBP,2,'',1);

save('data_task1_lbp_case2.mat')


