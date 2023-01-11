dbstop if error;

addpath('//Applications/MATLAB_R2022a.app/toolbox/tight_subplot/');
addpath('/Applications/MATLAB_R2021b.app/toolbox/confmat/');
clear; clc; 

addpath ('/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/');
addpath ('//Applications/MATLAB_R2021b.app/toolbox/random_forests_generic/');
addpath ('/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/');
addpath '/Users/pc/Google Drive/current data/plated Astrocytes Annalisa/randomForestMatlab'
% load AstrocyteDataComplete.mat;
load AstrocyteDataOnlyCTRL;

rng(2);
figure(3);

data(isundefined(data.MorfologiaLMNB),:)=[];


%% remove predictors
RemoveUnusedPred;

numTrees = 150;

%% creates X and y

y=data.MorfologiaLMNB;
data.MorfologiaLMNB=[];
X=data;




Mdl=fitcensemble(X,y, 'Method','RUSboost','NumLearningCycles',numTrees, 'CrossVal', 'on');
ypred=kfoldPredict(Mdl);
loss = kfoldLoss(Mdl);

% figure(55)
% view(Mdl.Trained{1},'Mode','graph')
% 

close all;
%[ha, pos] = tight_subplot(rows,columns,[vspace_between_plots hspace_between_plots],[bottom_margin top_margin],[left_margin right_margin]); 
[ha, pos] = tight_subplot (1,3,[ .05 0.1],[.2 0.1],[.05 0.1]);

annotation('textbox',[0.01  0.9 0.1 0.1], ...
    'String','Figure 3','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.01 0.8 0.1 0.1], ...
    'String','A','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.32 0.8 0.1 0.1], ...
    'String','B','EdgeColor','none','fontweight','bold','FontSize', 14);

annotation('textbox',[0.64 0.8 0.1 0.1], ...
    'String','C','EdgeColor','none','fontweight','bold','FontSize', 14);
set(gcf,'color','w');


axes(ha(1)); 
label = {'0','1','2','3','4'};
label = categorical(label);
confusionchart(y,ypred,'Normalization','row-normalized','RowSummary','row-normalized'); 
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title('Tree Ensemble - RUSboost'); % Set Title with correct Position
x0=10;
y0=10;
width=1600;
height=400;
set(gcf,'position',[x0,y0,width,height]);
ax = gca; % current axes
ax.FontSize = 12;

cm3=confusionmat(y,ypred);


[wF, wSp, wSe]=calculatef5(cm3);

annotation('textbox',[0.10 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);

%% panel 2
%% predict 3 class Morfologia LMNB  

load AstrocyteDataOnlyCTRL;

RemoveUnusedPred;

ThreeClassLMNB;
y=data.ConsMorfLMNB;
data.ConsMorfLMNB=[];


X=data;

Mdl=fitcensemble(X,y, 'Method','RUSboost','NumLearningCycles',numTrees, 'CrossVal', 'on');

% t = templateTree('Reproducible',true);
% Mdl = fitcensemble(X,y,'OptimizeHyperparameters','auto','Learners',t, ...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'))
% 
% Mdl=crossval(Mdl)

ypred=kfoldPredict(Mdl);
%[cmmean, CI, cmmed, Mdl]=bootEnsembleRUS(X,y,nReps, numTrees);

axes(ha(2)); 
% label = {'0','1','2','3','4'};
% label = categorical(label);
confusionchart(y,ypred,'Normalization','row-normalized','RowSummary','row-normalized');
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title ('Tree Ensemble - Consolidated categories');
ax = gca; % current axes
ax.FontSize = 12;

cm4=confusionmat(y,ypred);

[wF, wSe, wSp]=calculatef3(cm4);

annotation('textbox',[0.42 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);


%% third panel
%% panel 3
clearvars -except ha numTrees
load AstrocyteDataOnlyCTRL;

RemoveUnusedPred;

ThreeClassLMNB;
y=data.ConsMorfLMNB;
data.ConsMorfLMNB=[];


X=data;

ncat=length(unique(y));
for i=1:length(y)

    if y(i)=="1"
        y1(i,1)=1;
    else y1(i,1) = 0;
    end
    if y(i)=="2"
        y2(i,1)=1;
    else y2(i,1) = 0;
    end
    if y(i)=="3"
        y3(i,1)=1;
    else y3(i,1) = 0;
    end
end

y1=categorical (y1);
y2=categorical (y2);
y3=categorical (y3);

Mdl=fitcensemble(X,y1,'Method', 'RUSBoost','NumLearningCycles',numTrees, "CrossVal","on");[ypred1,Score1,Cost1]=kfoldPredict(Mdl);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mdl=fitcensemble(X,y2,'Method', 'RUSBoost','NumLearningCycles',numTrees, "CrossVal","on");
[ypred2,Score2,Cost2]=kfoldPredict(Mdl);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mdl=fitcensemble(X,y3,'Method', 'RUSBoost','NumLearningCycles',numTrees, "CrossVal","on");
[ypred3,Score3,Cost3]=kfoldPredict(Mdl);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
scores(:,1)=Score1(:,2)-Score1(:,1);
scores(:,2)=Score2(:,2)-Score2(:,1);
scores(:,3)=Score3(:,2)-Score3(:,1);

[M,decision]= max(scores,[],2);
decision=categorical(decision);

axes(ha(3)); 

cm = confusionchart(y,decision,'Normalization','row-normalized','RowSummary','row-normalized');
cm5=confusionmat(y,decision);
cm.RowSummary = 'row-normalized';
loscm.ColumnSummary = 'column-normalized';
title ('Tree Ensemble - Binary Decisions');
ax = gca; % current axes
ax.FontSize = 12;


[wF, wSe, wSp]=calculatef3(cm5);

annotation('textbox',[0.74 0 0.1 0.1], ...
    'String',['Se = ' num2str(round(wSe,2)) ', Sp = ' num2str(round(wSp,2))...
    ', F1 = ' num2str(round(wF,2))],'EdgeColor','none','FontSize', 14);


save Figure4.mat


 