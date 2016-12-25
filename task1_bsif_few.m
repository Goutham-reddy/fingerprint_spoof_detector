clear
load('data_main.mat')

%combine data from Digital and Sagen sensors

% combine BSIF features data
Train_All_Data_BSIF = horzcat(Train_All_Data_DigiBSIF, Train_All_Data_SageBSIF)
Train_All_Data_BSIF = num2cell(Train_All_Data_BSIF, 1)
Train_All_Label_BSIF = horzcat(Train_All_Label_DigiBSIF, Train_All_Label_SageBSIF)

Gen_All_Data_BSIF = horzcat(Test_All_Data_DigiBSIF, Test_All_Data_SageBSIF)
Gen_All_Label_BSIF = horzcat(Test_All_Label_DigiBSIF, Test_All_Label_SageBSIF)

Val_All_Data_BSIF  = Gen_All_Data_BSIF(: , 1:2018)
Val_All_Label_BSIF = Gen_All_Label_BSIF(: , 1:2018)

Test_All_Data_BSIF  = Gen_All_Data_BSIF(: , 2019:4036)
Test_All_Label_BSIF = Gen_All_Label_BSIF(: , 2019:4036)


rng('default')
hiddenSize1 = 125;
autoenc1 = trainAutoencoder(Train_All_Data_BSIF,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.12, ...
    'ScaleData', false);

%view(autoenc1)
feat1 = encode(autoenc1,Train_All_Data_BSIF);


softnet = trainSoftmaxLayer(feat1,Train_All_Label_BSIF,'MaxEpochs',400);
%view(softnet)

deepnet = stack(autoenc1,softnet);
view(deepnet)

% plot roc and confusion 
% For testing
%%y = deepnet(Test_All_Data_BSIF);
%figure(1)
%plotconfusion(Test_All_Label_BSIF,y);
%figure(2)
%plotroc(Test_All_Label_BSIF,y)

% For training
%y = deepnet(Train_All_Data_BSIF);
%figure(3)
%plotconfusion(Train_All_Label_BSIF,y);
%figure(4)
%plotroc(Train_All_Label_BSIF,y)

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
figure(4)
plotconfusion(Test_All_Label_BSIF,y);
%figure(6)
%plotroc(Test_All_Label_BSIF,y)
ezroc3(y,Test_All_Label_BSIF,2,'',1);

% For validation
y = deepnet_bp(Val_All_Data_BSIF);
figure(5)
plotconfusion(Val_All_Label_BSIF,y);
%figure(6)
%plotroc(Test_All_Label_BSIF,y)
ezroc3(y,Val_All_Label_BSIF,2,'',1);


% For training
Train_All_Data_BSIF_rev = cell2mat(Train_All_Data_BSIF)
y = deepnet_bp(Train_All_Data_BSIF_rev);
figure(7)
plotconfusion(Train_All_Label_BSIF,y);
%figure(8)
%plotroc(Train_All_Label_BSIF,y)
ezroc3(y,Train_All_Label_BSIF,2,'',1);

save('data_task1_bsif_few.mat')