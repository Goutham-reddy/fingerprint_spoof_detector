clear
load('data_main.mat')

%combine data from Digital and Sagen sensors

% combine BGP features data
Train_All_Data_BGP  = Train_All_Data_SageBGP
Train_All_Data_BGP  = num2cell(Train_All_Data_BGP, 1)
Train_All_Label_BGP = Train_All_Label_SageBGP

Test_All_Data_BGP = Test_All_Data_DigiBGP
Test_All_Label_BGP = Test_All_Label_DigiBGP

%rng('default')
hiddenSize1 = 50;
autoenc1 = trainAutoencoder(Train_All_Data_BGP,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

%view(autoenc1)
feat1 = encode(autoenc1,Train_All_Data_BGP);



softnet = trainSoftmaxLayer(feat1,Train_All_Label_BGP,'MaxEpochs',400);
%view(softnet)

deepnet = stack(autoenc1,softnet);
view(deepnet)

% plot roc and confusion 
% For testing
%y = deepnet(Test_All_Data_BGP);
%figure(1)
%plotconfusion(Test_All_Label_BGP,y);
%figure(2)
%plotroc(Test_All_Label_BGP,y)

% For training
%y = deepnet(Train_All_Data_BGP);
%figure(3)
%plotconfusion(Train_All_Label_BGP,y);
%figure(4)
%plotroc(Train_All_Label_BGP,y)

% Perform fine tuning
inputSize = 216
xTrain = zeros(inputSize,numel(Train_All_Data_BGP));
for i = 1:numel(Train_All_Data_BGP)
    xTrain(:,i) = Train_All_Data_BGP{i}(:);
end


deepnet_bp = train(deepnet,xTrain,Train_All_Label_BGP);

% plot roc and confusion 
% For testing
y = deepnet_bp(Test_All_Data_BGP);
figure(5)
plotconfusion(Test_All_Label_BGP,y);
figure(6)
%plotroc(Test_All_Label_BGP,y)
ezroc3(y,Test_All_Label_BGP,2,'',1);

% For training
Train_All_Data_BGP_rev = cell2mat(Train_All_Data_BGP)
y = deepnet_bp(Train_All_Data_BGP_rev);
figure(7)
plotconfusion(Train_All_Label_BGP,y);
%figure(8)
%plotroc(Train_All_Label_BGP,y)
ezroc3(y,Train_All_Label_BGP,2,'',1);


save('data_task2_sagem_train_digital_test_bgp.mat')