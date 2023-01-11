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



Mdl=fitcensemble(X,y,'Method', 'RUSBoost','NumLearningCycles',numTrees, "Leaveout","on");
% ypred=kfoldPredict(Mdl);


figure(5);
%% find predictor importance of cvmodel
for i=1:length(y)
    predImp(i,:)=predictorImportance(Mdl.Trained{i});
end

predImpMean= mean (predImp);
predImpSE = std(predImp) / sqrt(length(predImp));

xbar=[1:1:length(Mdl.PredictorNames)];
annotation('textbox',[0.01  0.9 0.1 0.1], ...
    'String','Figure 5','EdgeColor','none','fontweight','bold','FontSize', 14);


bar(xbar, predImpMean, 'FaceColor',[.5 .5 .5],'EdgeColor',[0 0 0],'LineWidth',2);
hold on 
errorbar(xbar, predImpMean,predImpSE,'o','Color','k', LineWidth=2);
hold on
% yline(0.006, 'k','Threshold', 'LineWidth',2);


xticks([1:1:length(Mdl.PredictorNames)]);
xticklabels({'GFAP','Cell Morphology','nRamifications','Cellular Area',...
    'Soma Area','Nuclear Area',...
    'Norm. Int. LMNB','Norm. Int. GFAP','Norm. Int. AQP4'});
% [ha, pos] = tight_subplot (1,3,[ .05 0.1],[.2 0.1],[.05 0.1]);

ylabel('Predictor Importance Score')
ax = gca; % current axes
ax.FontSize = 12;
ax.TickDir = 'out';
ax.LineWidth=2;
box off;
title('Important Predictors'); % Set Title with correct Position
x0=10;
y0=10;
width=1600;
height=400;
% set(gcf,'position',[x0,y0,width,height]);
ax = gca; % current axes
ax.FontSize = 12;



