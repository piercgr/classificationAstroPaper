dbstop if error;

addpath('//Applications/MATLAB_R2022a.app/toolbox/tight_subplot/');
addpath('/Applications/MATLAB_R2021b.app/toolbox/confmat/');
clear; clc; close all;

addpath ('/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/');
addpath ('/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/');
addpath '/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/randomForestMatlab'
% load AstrocyteDataComplete.mat;

rng(1);

%% panel 1
load AstrocyteDataOnlyCTRL;

RemoveUnusedPred;

ThreeClassLMNB;
y=data.ConsMorfLMNB;
data.ConsMorfLMNB=[];


X=data;

nReps=2500;
numTrees=150;


[cmmean, CI, cmmed, cms]=bootEnsembleRUS(X, y,nReps, numTrees);
cmmean2=cmmean;
[ha, pos] = tight_subplot (1,3,[ .05 0.1],[.2 0.1],[.05 0.1]);
annotation('textbox',[0.01  0.9 0.1 0.1], ...
    'String','Figure 4','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.01 0.8 0.1 0.1], ...
    'String','A','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.32 0.8 0.1 0.1], ...
    'String','B','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.64 0.8 0.1 0.1], ...
    'String','C','EdgeColor','none','fontweight','bold','FontSize', 14);
set(gcf,'color','w');


axes(ha(1)); 

cm = confusionchart(round(cmmean2*1000),'Normalization','row-normalized','RowSummary','row-normalized');
% cmbin3=confusionmat(testy,decision);
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title ('Tree Ensemble Bootstrap');
x0=10;
y0=10;
width=1600;
height=400;
set(gcf,'position',[x0,y0,width,height]);
ax = gca; % current axes
ax.FontSize = 12;



[wF, wSe, wSp]=calculatef3(cmmean2);

annotation('textbox',[0.10 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);




%% panel 2
clearvars -except ha numTrees
load AstrocyteDataOnlyCTRL;

RemoveUnusedPred;

ThreeClassLMNB;
y=data.ConsMorfLMNB;
data.ConsMorfLMNB=[];


X=data;
X.zIntDenLMNB=[]


%% impo=[3 4 6 7];
%% Mdl=fitcensemble(X(:,impo),y,'Method', 'RUSBoost','NumLearningCycles',numTrees, "Leaveout","on");

Mdl=fitcensemble(X,y,'Method', 'RUSBoost','NumLearningCycles',numTrees, "Leaveout","on");
ypred=kfoldPredict(Mdl);
 
axes(ha(2)); 

cm = confusionchart(y,ypred,'Normalization','row-normalized','RowSummary','row-normalized');
cm4=confusionmat(y,ypred);
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title ('Tree Ensemble LOOCV');
ax = gca; % current axes
ax.FontSize = 12;


[wF, wSe, wSp]=calculatef3(cm4);

annotation('textbox',[0.42 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);



%% panel 3
clearvars -except ha numTrees
load AstrocyteDataOnlyCTRL;

RemoveUnusedPred;

ThreeClassLMNB;
y=data.ConsMorfLMNB;
data.ConsMorfLMNB=[];


X=data;

c = cvpartition(y,'KFold',5,'Stratify',true);

Mdl=fitcensemble(X,y,'Method', 'RUSBoost','NumLearningCycles',numTrees, "cvpartition",c);
ypred=kfoldPredict(Mdl);

axes(ha(3)); 

cm = confusionchart(y,ypred,'Normalization','row-normalized','RowSummary','row-normalized');
cm5=confusionmat(y,ypred);
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title ('Tree Ensemble Stratified CV');
ax = gca; % current axes
ax.FontSize = 12;
 

[wF, wSe, wSp]=calculatef3(cm5);

annotation('textbox',[0.74 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);


