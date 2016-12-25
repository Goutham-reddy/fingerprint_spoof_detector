clear
load('data_main.mat')

%combine data from Digital and Sagen sensors

% combine BSIF features data
Train_All_Data_BSIF = horzcat(Train_All_Data_DigiBSIF, Train_All_Data_SageBSIF)
Train_All_Data_BSIF = num2cell(Train_All_Data_BSIF, 1)
Train_All_Label_BSIF = horzcat(Train_All_Label_DigiBSIF, Train_All_Label_SageBSIF)

Test_All_Data_BSIF = horzcat(Test_All_Data_DigiBSIF, Test_All_Data_SageBSIF)
Test_All_Label_BSIF = horzcat(Test_All_Label_DigiBSIF, Test_All_Label_SageBSIF)

%rng('default')
hiddenSize1 = 400;
autoenc1 = trainAutoencoder(Train_All_Data_BSIF,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.20, ...
    'ScaleData', false);

%view(autoenc1)
feat1 = encode(autoenc1,Train_All_Data_BSIF);

hiddenSize1 = 200;

autoenc2 = trainAutoencoder(feat1,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.003, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.18, ...
    'ScaleData', false);

%view(autoenc2)
feat2 = encode(autoenc2,feat1);

hiddenSize1 = 100;

autoenc3 = trainAutoencoder(feat2,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.16, ...
    'ScaleData', false);

%view(autoenc2)
feat3 = encode(autoenc3,feat2);

hiddenSize1 = 50;

autoenc4 = trainAutoencoder(feat3,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.14, ...
    'ScaleData', false);

%view(autoenc2)
feat4 = encode(autoenc4,feat3);

softnet = trainSoftmaxLayer(feat4,Train_All_Label_BSIF,'MaxEpochs',400);
%view(softnet)

deepnet = stack(autoenc1,autoenc2,autoenc3,autoenc4,softnet);
view(deepnet)

% plot roc and confusion 
% For testing
y = deepnet(Test_All_Data_BSIF);
figure(1)
plotconfusion(Test_All_Label_BSIF,y);
figure(2)
plotroc(Test_All_Label_BSIF,y)

% For training
y = deepnet(Train_All_Data_BSIF);
figure(3)
plotconfusion(Train_All_Label_BSIF,y);
figure(4)
plotroc(Train_All_Label_BSIF,y)

% Perform fine tuning
inputSize = 512
xTrain = zeros(inputSize,numel(Train_All_Data_BSIF));
for i = 1:numel(Train_All_Data_BSIF)
    xTrain(:,i) = Train_All_Data_BSIF{i}(:);
end


deepnet_bp = train(deepnet,xTrain,Train_All_Label_BSIF);

% plot roc and confusion 
% For testing
y = deepnet_bp(Test_All_Data_BSIF);
figure(5)
plotconfusion(Test_All_Label_BSIF,y);
figure(6)
plotroc(Test_All_Label_BSIF,y)

% For training
y = deepnet_bp(Train_All_Data_BSIF);
figure(7)
plotconfusion(Train_All_Label_BSIF,y);
figure(8)
plotroc(Train_All_Label_BSIF,y)

save('data_bsif.mat')