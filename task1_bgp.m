clear
load('data_main.mat')

%combine data from Digital and Sagen sensors

% combine BGP features data
Train_All_Data_BGP = horzcat(Train_All_Data_DigiBGP, Train_All_Data_SageBGP)
Train_All_Data_BGP = num2cell(Train_All_Data_BGP, 1)
Train_All_Label_BGP = horzcat(Train_All_Label_DigiBGP, Train_All_Label_SageBGP)

Test_All_Data_BGP = horzcat(Test_All_Data_DigiBGP, Test_All_Data_SageBGP)
Test_All_Label_BGP = horzcat(Test_All_Label_DigiBGP, Test_All_Label_SageBGP)

rng('default')
hiddenSize1 = 800;
autoenc1 = trainAutoencoder(Train_All_Data_BGP,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.20, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.010, ...
    'ScaleData', false);

view(autoenc1)
feat1 = encode(autoenc1,Train_All_Data_BGP);

hiddenSize1 = 400;

autoenc2 = trainAutoencoder(feat1,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.18, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.008, ...
    'ScaleData', false);

view(autoenc2)
feat2 = encode(autoenc2,feat1);

hiddenSize1 = 200;

autoenc3 = trainAutoencoder(feat2,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.16, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.006, ...
    'ScaleData', false);

view(autoenc3)
feat3 = encode(autoenc3,feat2);

hiddenSize1 = 100;

autoenc4 = trainAutoencoder(feat3,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.12, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.004, ...
    'ScaleData', false);

view(autoenc4)
feat4 = encode(autoenc4,feat3);

softnet = trainSoftmaxLayer(feat4,Train_All_Label_BGP,'MaxEpochs',400);
view(softnet)

deepnet = stack(autoenc1,autoenc2,autoenc3,autoenc4,softnet);
view(deepnet)

% plot roc and confusion 
% For testing
y = deepnet(Test_All_Data_BGP);
figure(1)
plotconfusion(Test_All_Label_BGP,y);
figure(2)
plotroc(Test_All_Label_BGP,y)

% For training
y = deepnet(Train_All_Data_BGP);
figure(3)
plotconfusion(Train_All_Label_BGP,y);
figure(4)
plotroc(Train_All_Label_BGP,y)

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
plotroc(Test_All_Label_BGP,y)

% For training
y = deepnet_bp(Train_All_Data_BGP);
figure(7)
plotconfusion(Train_All_Label_BGP,y);
figure(8)
plotroc(Train_All_Label_BGP,y)

save('data_task1_bgp.mat')


