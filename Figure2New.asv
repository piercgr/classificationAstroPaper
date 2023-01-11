dbstop if error; clear; clc; close all;


addpath('//Applications/MATLAB_R2022a.app/toolbox/tight_subplot/');
addpath('/Applications/MATLAB_R2021b.app/toolbox/confmat/');
addpath ('/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/');
addpath ('//Applications/MATLAB_R2021b.app/toolbox/random_forests_generic/');
addpath ('/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/');
addpath '/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/randomForestMatlab'
% load AstrocyteDataComplete.mat;
load AstrocyteDataOnlyCTRL;

rng(3);
RemoveUnusedPred;
data(isundefined(data.MorfologiaLMNB),:)=[];
numTrees=150;
 

%% define outcome 

outcome  = 'MorfologiaLMNB';
%% define X and y
y=data.(outcome);
data.(outcome)=[];
X=data;


%% classification tree using crossval 
clearvars -except X y numTrees data
ypredT = fitctree(X,y,'CrossVal','on');
ypredtree=kfoldPredict(ypredT);
loss = kfoldLoss(ypredT);

figure(55)
view(ypredT.Trained{2},'Mode','graph')


%% treeensemble using crossval 
Mdl=fitcensemble(X,y,'NumLearningCycles',numTrees, "CrossVal","on");


% Mdl=fitcensemble(X,y,'NumLearningCycles',numTrees, "Leaveout","on");
ypredtreeE=kfoldPredict(Mdl);



%% treebagger
Mdl = TreeBagger(100,X,y,'OOBPrediction','On','Method','classification')
oobErrorBaggedEnsemble = oobError(Mdl);
 
OObPred=oobPredict(Mdl);
predictionOOb=categorical(OObPred);

 
figure(2);
[ha, pos] = tight_subplot (1,3,[ .05 0.1],[.2 0.1],[.05 0.1]);
axes(ha(1)); 
confusionchart(y,ypredtree,'Normalization','row-normalized','RowSummary','row-normalized');
cm1=confusionmat(y,ypredtree);
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title('Classification Tree'); % Set Title with correct Position
x0=10;
y0=10;
width=1600;
height=400;
set(gcf,'position',[x0,y0,width,height]);
 

[wF, wSe, wSp]=calculatef5(cm1);

annotation('textbox',[0.10 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);


axes(ha(2)); 
confusionchart(y,predictionOOb,'Normalization','row-normalized','RowSummary','row-normalized');
cm2=confusionmat(y,predictionOOb);
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title ('Treebagger');
% 
% [m,n]=size (cm2);
% diagcm2= diag(cm2);
% idx = eye(m,n);
% offdiagcm2=cm2(idx==0);
% errcm2= sum(offdiagcm2)/(sum(diagcm2)+sum(offdiagcm2));

[wF, wSe, wSp]=calculatef5(cm2);

annotation('textbox',[0.42 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);

axes(ha(3));
confusionchart(y,ypredtreeE,'Normalization','row-normalized','RowSummary','row-normalized');
cm3=confusionmat(y,ypredtreeE);
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title ('Tree Ensemble');
xlabel('Predicted Class');
ylabel('True Class');

% [m,n]=size (cm3);
% diagcm3= diag(cm3);
% idx = eye(m,n);
% offdiagcm3=cm3(idx==0);
% errcm3= sum(offdiagcm3)/(sum(diagcm3)+sum(offdiagcm3));

[wF, wSe, wSp]=calculatef5(cm3);

annotation('textbox',[0.74 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);
 set(gcf,'color','w');


annotation('textbox',[0.01  0.9 0.1 0.1], ...
    'String','Figure 2','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.01 0.8 0.1 0.1], ...
    'String','A','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.32 0.8 0.1 0.1], ...
    'String','B','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.64 0.8 0.1 0.1], ...
    'String','C','EdgeColor','none','fontweight','bold','FontSize', 14);

 








