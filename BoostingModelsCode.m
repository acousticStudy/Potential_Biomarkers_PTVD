clc;
clear all;
close all;
% warning off;
% warning('off', 'stats:lassoglm:IterationLimit');

% To enhance transparency and reproducibility, code was provided with inline comments. 
% For brevity, some repetitive blocks were not modularized but have been rigorously tested.

% The 'Method' parameter in fitcensemble can be adjusted to switch between different boosting algorithms.
% While 'GentleBoost' is used by default in this script, it can be replaced with 'LogitBoost' or any other supported method.
% Additionally, the hyperparameters 'NumLearningCycles' and 'LearnRate' can be modified within the ranges specified in the manuscript,
% or set to any desired values to tune the model's complexity and learning behavior.

tic;
seed = 2;
rng(seed, 'twister');

numFolds = 3;
numRuns = 100;

treeTemplate = templateTree(...
    'MaxNumSplits', 10, ...               
    'MinLeafSize', 3, ...                 
    'MinParentSize', 6, ...               
    'PredictorSelection', 'allsplits', ...
    'Reproducible', true);                


testTP_GentleBoost_values = zeros(numRuns, 1);
testFP_GentleBoost_values = zeros(numRuns, 1);
testTN_GentleBoost_values = zeros(numRuns, 1);
testFN_GentleBoost_values = zeros(numRuns, 1);

test_confmat_GentleBoost_all = zeros(2, 2, numRuns);

allScoresGentleBoost = [];
allLabelsGentleBoost = [];

validationScoresGentleBoost = [];
validationLabelsGentleBoost = [];

validationGentleBoostAccuracies = zeros(numFolds, numRuns);
validationGentleBoostPrecisions = zeros(numFolds, numRuns);
validationGentleBoostRecalls = zeros(numFolds, numRuns);
validationGentleBoostSpecificities = zeros(numFolds, numRuns);
validationGentleBoostF1Scores = zeros(numFolds, numRuns);

testGentleBoostAccuracies = zeros(numRuns, 1);
testGentleBoostPrecisions = zeros(numRuns, 1);
testGentleBoostRecalls = zeros(numRuns, 1);
testGentleBoostSpecificities = zeros(numRuns, 1);
testGentleBoostF1Scores = zeros(numRuns, 1);


data = readtable('Dataset.xlsx');
Xall = table2array(data(:, 2:end));
yall = table2array(data(:, 1));

successfulRuns = 0;
attemptedRuns = 0;

allLabelsValGentleBoost = [];
allPredsValGentleBoost = [];

numSelectedFeatures = NaN(numRuns, 1); 

allFeatureNames = data.Properties.VariableNames(2:end);
numAllFeatures = numel(allFeatureNames);
shapleyResultsFull = NaN(numRuns, numAllFeatures);

shapCorrelation = zeros(numRuns, 1);
shapCorrelationPval = zeros(numRuns, 1);

allAverageShapTrain = NaN(numRuns, numAllFeatures);
allAverageShapTest  = NaN(numRuns, numAllFeatures);

featureSelectionCount = zeros(1, numAllFeatures);

while successfulRuns < numRuns
    run = successfulRuns + 1;
    attemptedRuns = attemptedRuns + 1;

    fprintf('\n Running iteration %d of %d...\n', run, numRuns);


    runSuccessful = false;

    cv_outer = cvpartition(yall, 'HoldOut', 0.2, 'Stratify', true);
    trainIdx = cv_outer.training;
    testIdx  = cv_outer.test;

    Xtrain = table2array(data(trainIdx, 2:end));
    Xtest = table2array(data(testIdx, 2:end));
    ytrain = yall(trainIdx);
    ytest = yall(testIdx);

    % Class control
    if numel(unique(ytrain)) < 2 || numel(unique(ytest)) < 2
        fprintf('Contains a single class, iteration skipped.\n');
        continue;
    end

 
    %% Z-score normalization
    Xtrain_z = zscore(Xtrain);
    mu = mean(Xtrain);
    sigma = std(Xtrain);
    Xtest_z = (Xtest - mu) ./ sigma;

    %% LASSO
    alpha = 1;  % LASSO

    [B, FitInfo] = lassoglm(Xtrain_z, ytrain, ...
        'binomial', ...
        'Alpha', 1, ...
        'CV', 5, ...
        'NumLambda', 15, ...              
        'LambdaRatio', 1e-2, ...          
        'MaxIter', 5e4, ...               
        'Standardize', false);           

    idxLambdaMin = FitInfo.IndexMinDeviance;
    selectedLasso = B(:, idxLambdaMin) ~= 0;
    selectedFeatureIndices = find(selectedLasso);
   
    if sum(selectedLasso) < 2
        fprintf('Not enough features left after LASSO.\n');
        continue;
    end


    % Names of all features
    allFeatureNames = data.Properties.VariableNames(2:end);
    numAllFeatures = numel(allFeatureNames);

    selectedFeatureNames = allFeatureNames(selectedFeatureIndices);

    Xtrain_final = Xtrain(:, selectedFeatureIndices);
    Xtest_final  = Xtest(:, selectedFeatureIndices);

    numSelectedFeatures(run) = numel(selectedFeatureIndices);


    fprintf('Number of samples in the training set: %d\n', size(Xtrain_final,1));
    fprintf('Class distribution of training set:\n');
    tabulate(ytrain);

    successfulFolds = 0;
    attemptedFolds = 0;

    while successfulFolds < numFolds
        attemptedFolds = attemptedFolds + 1;

        cv_inner = cvpartition(ytrain, 'KFold', numFolds, 'Stratify', true);


        for fold = 1:cv_inner.NumTestSets
            trainIdx_fold = cv_inner.training(fold);
            validationIdx_fold = cv_inner.test(fold);
            Xtrain_fold = Xtrain(trainIdx_fold, :);
            ytrain_fold = ytrain(trainIdx_fold, :);
            Xvalidation = Xtrain(validationIdx_fold, :);
            yvalidation = ytrain(validationIdx_fold, :);

            if numel(unique(ytrain_fold)) < 2 || numel(unique(yvalidation)) < 2
                fprintf('Fold %d skipped due to single class.\n', fold);
                continue;
            end
            
            Xtrain_fold_selected = Xtrain_fold(:, selectedFeatureIndices);
            Xvalidation_selected = Xvalidation(:, selectedFeatureIndices);

            tblTrainFold = array2table(Xtrain_fold_selected, 'VariableNames', selectedFeatureNames);
            tblValidation = array2table(Xvalidation_selected, 'VariableNames', selectedFeatureNames);
            
            % GentleBoost 
            ModelGentleBoost = fitcensemble(tblTrainFold, ytrain_fold, 'Method', 'GentleBoost', ...
                'NumLearningCycles', 100, ...
                'LearnRate', 0.1, ...
                'Learners', treeTemplate);
            [ypred_validation_GentleBoost, scoresGentleBoostValidation] = predict(ModelGentleBoost, tblValidation);

            validation_confmat_GentleBoost = confusionmat(yvalidation, ypred_validation_GentleBoost);

            % If the confusion matrix is not 2x2, fill in the missing cells.
            if size(validation_confmat_GentleBoost, 1) < 2
                validation_confmat_GentleBoost(2, 2) = 0;
            end
            if size(validation_confmat_GentleBoost, 2) < 2
                validation_confmat_GentleBoost(2, 2) = 0;
            end

            validationTP_GentleBoost = validation_confmat_GentleBoost(2, 2);
            validationFP_GentleBoost = validation_confmat_GentleBoost(1, 2);
            validationTN_GentleBoost = validation_confmat_GentleBoost(1, 1);
            validationFN_GentleBoost = validation_confmat_GentleBoost(2, 1);

            validationAccuracy_GentleBoost = (validationTP_GentleBoost + validationTN_GentleBoost) / (validationTP_GentleBoost + validationFP_GentleBoost + validationTN_GentleBoost + validationFN_GentleBoost);
            validationprecision_GentleBoost = validationTP_GentleBoost / (validationTP_GentleBoost + validationFP_GentleBoost);
            validationrecall_GentleBoost = validationTP_GentleBoost / (validationTP_GentleBoost + validationFN_GentleBoost);
            validationspecificity_GentleBoost = validationTN_GentleBoost / (validationTN_GentleBoost + validationFP_GentleBoost);
            validationF1_GentleBoost = 2 * (validationprecision_GentleBoost * validationrecall_GentleBoost) / (validationprecision_GentleBoost + validationrecall_GentleBoost);

            if validationAccuracy_GentleBoost == 0
                validationAccuracy_GentleBoost = NaN;
            end

            if validationprecision_GentleBoost == 0
                validationprecision_GentleBoost = NaN;
            end

            if validationrecall_GentleBoost == 0
                validationrecall_GentleBoost = NaN;
            end

            if validationspecificity_GentleBoost == 0
                validationspecificity_GentleBoost = NaN;
            end

            if validationF1_GentleBoost == 0
                validationF1_GentleBoost = NaN;
            end

            % NaN check
            skipIteration1 = false;

            if isnan(validationAccuracy_GentleBoost) || isnan(validationprecision_GentleBoost) || isnan(validationrecall_GentleBoost) || isnan(validationspecificity_GentleBoost) || isnan(validationF1_GentleBoost)
                skipIteration1 = true;
            else

                validationGentleBoostAccuracies(fold, run) = validationAccuracy_GentleBoost*100;
                validationGentleBoostPrecisions(fold, run) = validationprecision_GentleBoost*100;
                validationGentleBoostRecalls(fold, run) = validationrecall_GentleBoost*100;
                validationGentleBoostSpecificities(fold, run) = validationspecificity_GentleBoost*100;
                validationGentleBoostF1Scores(fold, run) = validationF1_GentleBoost*100;

                validationScoresGentleBoost = [validationScoresGentleBoost; scoresGentleBoostValidation(:,2)];
                validationLabelsGentleBoost = [validationLabelsGentleBoost; yvalidation];
            end

            % If NaN value is present skip this iteration
            if skipIteration1
                fprintf('Run %d, Fold %d skipped due to NaN values.\n', attemptedRuns, fold);
                continue;
            end

  
            % If all skipIteration flags are false, increase the count of successful iterations.
            if ~(skipIteration1)
                successfulFolds = successfulFolds + 1;
                allLabelsValGentleBoost = [allLabelsValGentleBoost; yvalidation];
            end

        end
    end


    %% For the Test Data
    Xtrain_final_selected = Xtrain(:, selectedFeatureIndices);
    Xtest_final_selected = Xtest(:, selectedFeatureIndices);

    tblTrainFinal = array2table(Xtrain_final_selected, 'VariableNames', selectedFeatureNames);
    tblTestFinal = array2table(Xtest_final_selected, 'VariableNames', selectedFeatureNames);
    
    ModelGentleBoostTest = fitcensemble(tblTrainFinal, ytrain, 'Method', 'GentleBoost', ...
        'NumLearningCycles', 100, ...
        'LearnRate', 0.1, ...
        'Learners', treeTemplate);

    [~, scoresGentleBoost] = predict(ModelGentleBoostTest, tblTestFinal);

    % Store the scores and the labels.
    allScoresGentleBoost = [allScoresGentleBoost; scoresGentleBoost(:,2)];
    allLabelsGentleBoost = [allLabelsGentleBoost; ytest];

    ypred_test_GentleBoost = predict(ModelGentleBoostTest, tblTestFinal);

    confmat_test_GentleBoost = confusionmat(ytest, ypred_test_GentleBoost);
    
    % If the confusion matrix is not 2x2, fill in the missing cells.

    if size(confmat_test_GentleBoost, 1) < 2
        confmat_test_GentleBoost(2, 2) = 0;
    end
    if size(confmat_test_GentleBoost, 2) < 2
        confmat_test_GentleBoost(2, 2) = 0;
    end

    testTP_GentleBoost = confmat_test_GentleBoost(2, 2);
    testFP_GentleBoost = confmat_test_GentleBoost(1, 2);
    testTN_GentleBoost = confmat_test_GentleBoost(1, 1);
    testFN_GentleBoost = confmat_test_GentleBoost(2, 1);

    accuracy_test_GentleBoost = (testTP_GentleBoost + testTN_GentleBoost) / (testTP_GentleBoost + testFP_GentleBoost + testTN_GentleBoost + testFN_GentleBoost);
    precision_test_GentleBoost = testTP_GentleBoost / (testTP_GentleBoost + testFP_GentleBoost);
    recall_test_GentleBoost = testTP_GentleBoost / (testTP_GentleBoost + testFN_GentleBoost);
    specificity_test_GentleBoost = testTN_GentleBoost / (testTN_GentleBoost + testFP_GentleBoost);
    testF1_GentleBoost = 2 * (precision_test_GentleBoost * recall_test_GentleBoost) / (precision_test_GentleBoost + recall_test_GentleBoost);

    % After replacing zero-valued metrics with NaN (to indicate structurally undefined or uninformative results),
    % a NaN check is performed before counting the iteration as successful.
    % This ensures that only iterations with valid and interpretable performance metrics are included in the final evaluation.

    if accuracy_test_GentleBoost == 0
        accuracy_test_GentleBoost = NaN;
    end

    if precision_test_GentleBoost == 0
        precision_test_GentleBoost = NaN;
    end

    if recall_test_GentleBoost == 0
        recall_test_GentleBoost = NaN;
    end

    if specificity_test_GentleBoost == 0
        specificity_test_GentleBoost = NaN;
    end

    if testF1_GentleBoost == 0
        testF1_GentleBoost = NaN;
    end


    fprintf('GentleBoost  Test Accuracy: %.2f\n', accuracy_test_GentleBoost*100);
    fprintf('GentleBoost  Test Precision: %.2f\n', precision_test_GentleBoost*100);
    fprintf('GentleBoost  Test Recall: %.2f\n', recall_test_GentleBoost*100);
    fprintf('GentleBoost  Test Specificity: %.2f\n', specificity_test_GentleBoost*100);
    fprintf('GentleBoost  Test F1-score: %.2f\n', testF1_GentleBoost*100);

    % NaN check
    skipIteration3 = false;

    if isnan(accuracy_test_GentleBoost) || isnan(precision_test_GentleBoost) || isnan(recall_test_GentleBoost) || isnan(specificity_test_GentleBoost) || isnan(testF1_GentleBoost)
        skipIteration3 = true;
    else
        testGentleBoostAccuracies(run) = accuracy_test_GentleBoost*100;
        testGentleBoostPrecisions(run) = precision_test_GentleBoost*100;
        testGentleBoostRecalls(run) = recall_test_GentleBoost*100;
        testGentleBoostSpecificities(run) = specificity_test_GentleBoost*100;
        testGentleBoostF1Scores(run) = testF1_GentleBoost*100;
    end

    if skipIteration3
        fprintf('Run %d skipped due to NaN values.\n', attemptedRuns);
        continue;
    end

    successfulRuns = successfulRuns + 1;

    test_confmat_GentleBoost_all(:, :, run) = confmat_test_GentleBoost;
   
    %% Shapley
   
    selectedNames = selectedFeatureNames;

    blackbox_test = ModelGentleBoostTest;  % Use the trained model in SHAP
    
    % Train SHAP
    
    Xtrain_selected = Xtrain_final;  % Selected features used during training

    if any(any(ismissing(Xtrain_selected)))
        error('There are missing (NaN) values in Xtrain_selected; SHAP cannot be computed.');
    end

    predictorNames = selectedFeatureNames;
    dataTrain = array2table(Xtrain_selected, 'VariableNames', predictorNames);

    assert(isequal(predictorNames, dataTrain.Properties.VariableNames), ...
    'Predictor names and table columns do not match.');

    numTrainPoints = size(dataTrain, 1);
    numFeatures = length(predictorNames);
    
    explainerTrain = shapley(ModelGentleBoostTest, 'QueryPoints', dataTrain);
    shapMatrixTrain = table2array(explainerTrain.ShapleyValues(:, "1"));  % For Class 1
    averageShapleyTrain = mean(shapMatrixTrain, 2)';  

    shapleyTableTrain = array2table(averageShapleyTrain, 'VariableNames', predictorNames);


    % Test SHAP

    Xtest_selected = Xtest_final;
    

    dataTest = array2table(Xtest_selected, 'VariableNames', predictorNames);
    numQueryPoints = size(dataTest, 1);

    explainerTest = shapley(ModelGentleBoostTest, 'QueryPoints', dataTest);
    shapMatrixTest = table2array(explainerTest.ShapleyValues(:, "1"));
    averageShapleyTest = mean(shapMatrixTest, 2)';

    for iFeat = 1:numel(selectedFeatureNames)
        featName = selectedFeatureNames{iFeat};
        idx = find(strcmp(allFeatureNames, featName));
        if ~isempty(idx)
            allAverageShapTrain(run, idx) = averageShapleyTrain(iFeat);
            allAverageShapTest(run, idx)  = averageShapleyTest(iFeat);
            shapleyResultsFull(run, idx)  = averageShapleyTest(iFeat);
            featureSelectionCount(idx) = featureSelectionCount(idx) + 1;
        end
    end


     
[rhoSHAP, pvalSHAP] = corr(averageShapleyTrain(:), averageShapleyTest(:), 'Type', 'Spearman');

shapCorrelation(run) = rhoSHAP;
shapCorrelationPval(run) = pvalSHAP;


end

% Select common features
threshold = 75;
selectedIdx = find(featureSelectionCount >= threshold);
stableFeatureNames = allFeatureNames(selectedIdx);  

% Identify valid iterations (with at least 2 SHAP values)
validRuns = sum(~isnan(allAverageShapTrain(:, selectedIdx)), 2) >= 2;

% Recalculate SHAP correlations (only for common features)
stable_rho = NaN(numRuns,1);
stable_pval = NaN(numRuns,1);

for i = 1:numRuns
    trainVals = allAverageShapTrain(i, selectedIdx);
    testVals  = allAverageShapTest(i,  selectedIdx);
    
    if all(isnan(trainVals)) || all(isnan(testVals))
        continue;
    end

    [rho, pval] = corr(trainVals(:), testVals(:), 'Type', 'Spearman');
    stable_rho(i) = rho;
    stable_pval(i) = pval;
end

mean_rho = mean(stable_rho(validRuns), 'omitnan');
significant_ratio = mean(stable_pval(validRuns) < 0.05) * 100;

fprintf('Mean SHAP correlation for common features: %.3f\n', mean_rho);
fprintf('Ratio of significant correlations (p < 0.05): %.1f%%\n', significant_ratio);

shapTrainTbl = array2table(allAverageShapTrain(validRuns, selectedIdx), 'VariableNames', stableFeatureNames);
shapTestTbl  = array2table(allAverageShapTest(validRuns,  selectedIdx), 'VariableNames', stableFeatureNames);

shapCorrTable = table(find(validRuns), stable_rho(validRuns), stable_pval(validRuns), ...
    'VariableNames', {'Run', 'SpearmanRho', 'PValue'});

summaryStats = table(mean(stable_rho(validRuns), 'omitnan'), ...
                     mean(stable_pval(validRuns) < 0.05) * 100, ...
                     'VariableNames', {'MeanRho', 'SignificantPvalRatio'});

writetable(shapTrainTbl,  'SHAP_Results_Stable.xlsx', 'Sheet', 'Train_SHAP');
writetable(shapTestTbl,   'SHAP_Results_Stable.xlsx', 'Sheet', 'Test_SHAP');
writetable(shapCorrTable, 'SHAP_Results_Stable.xlsx', 'Sheet', 'SHAP_Correlation');
writetable(summaryStats,  'SHAP_Results_Stable.xlsx', 'Sheet', 'SHAP_Summary');




fileID = fopen('GentleBoost_results1-100.txt','w');

% Compute the average results
validation_avgGentleBoostAccuracy = mean(validationGentleBoostAccuracies(:));
validation_avgGentleBoostPrecision = mean(validationGentleBoostPrecisions(:));
validation_avgGentleBoostRecall = mean(validationGentleBoostRecalls(:));
validation_avgGentleBoostSpecificity = mean(validationGentleBoostSpecificities(:));
validation_avgGentleBoostF1 = mean(validationGentleBoostF1Scores(:));

validation_stdGentleBoostAccuracy = std(validationGentleBoostAccuracies(:));
validation_stdGentleBoostPrecision = std(validationGentleBoostPrecisions(:));
validation_stdGentleBoostRecall = std(validationGentleBoostRecalls(:));
validation_stdGentleBoostSpecificity = std(validationGentleBoostSpecificities(:));
validation_stdGentleBoostF1 = std(validationGentleBoostF1Scores(:));

fprintf(fileID, '(Validation) GentleBoost  Accuracy: Mean = %.2f\n, STD = %2f\n', validation_avgGentleBoostAccuracy, validation_stdGentleBoostAccuracy);
fprintf(fileID, '(Validation) GentleBoost  Precision: Mean = %.2f\n, STD = %2f\n', validation_avgGentleBoostPrecision, validation_stdGentleBoostPrecision);
fprintf(fileID, '(Validation) GentleBoost  Recall: Mean = %.2f\n, STD = %2f\n', validation_avgGentleBoostRecall, validation_stdGentleBoostRecall);
fprintf(fileID, '(Validation) GentleBoost  Specificity: Mean = %.2f\n, STD = %2f\n', validation_avgGentleBoostSpecificity, validation_stdGentleBoostSpecificity);
fprintf(fileID, '(Validation) GentleBoost  F1-score: Mean = %.2f\n, STD = %2f\n', validation_avgGentleBoostF1, validation_stdGentleBoostF1);


meanTestGentleBoostAcc = mean(testGentleBoostAccuracies(:));
stdTestGentleBoostAcc = std(testGentleBoostAccuracies(:));

meanTestGentleBoostPrec = mean(testGentleBoostPrecisions(:));
stdTestGentleBoostPrec = std(testGentleBoostPrecisions(:));

meanTestGentleBoostRecall = mean(testGentleBoostRecalls(:));
stdTestGentleBoostRecall = std(testGentleBoostRecalls(:));

meanTestGentleBoostSpec = mean(testGentleBoostSpecificities(:));
stdTestGentleBoostSpec = std(testGentleBoostSpecificities(:));

meanTestGentleBoostF1 = mean(testGentleBoostF1Scores(:));
stdTestGentleBoostF1 = std(testGentleBoostF1Scores(:));

fileID = fopen('GentleBoost_results1-100.txt','a');

fprintf(fileID, '\nGentleBoost Test Accuracy: Mean = %f, Std = %f\n', meanTestGentleBoostAcc, stdTestGentleBoostAcc);
fprintf(fileID, 'GentleBoost Test Precision: Mean = %f, Std = %f\n', meanTestGentleBoostPrec, stdTestGentleBoostPrec);
fprintf(fileID, 'GentleBoost Test Recall: Mean = %f, Std = %f\n', meanTestGentleBoostRecall, stdTestGentleBoostRecall);
fprintf(fileID, 'GentleBoost Test Specificity: Mean = %f, Std = %f\n', meanTestGentleBoostSpec, stdTestGentleBoostSpec);
fprintf(fileID, 'GentleBoost Test F1-score: Mean = %f, Std = %f\n', meanTestGentleBoostF1, stdTestGentleBoostF1);

fclose(fileID);

testAUCValuesGentleBoost = zeros(numRuns, 1);
validationAUCValuesGentleBoost = zeros(numRuns, 1);

% Compute the AUC values for each run
for i = 1:numRuns
    step = floor(length(allLabelsGentleBoost) / numRuns);
    startIndexGentleBoost = (i - 1) * step + 1;
    endIndexGentleBoost = i * step;

    labels = allLabelsGentleBoost(startIndexGentleBoost:endIndexGentleBoost);
    scores = allScoresGentleBoost(startIndexGentleBoost:endIndexGentleBoost);

    if numel(unique(labels)) < 2
        fprintf('Run %d skipped: ROC AUC could not be calculated (single class).\n', i);
        testAUC_GentleBoost = NaN;
        continue; 
    else
        [~, ~, ~, testAUC_GentleBoost] = perfcurve(labels, scores, 1);
    end

    testAUCValuesGentleBoost(i) = testAUC_GentleBoost;

    stepVal = floor(length(validationLabelsGentleBoost) / numRuns);
    startIndexValGentleBoost = (i - 1) * stepVal + 1;
    endIndexValGentleBoost = i * stepVal;

    val_labels = validationLabelsGentleBoost(startIndexValGentleBoost:endIndexValGentleBoost);
    val_scores = validationScoresGentleBoost(startIndexValGentleBoost:endIndexValGentleBoost);


    if numel(unique(val_labels)) < 2
        fprintf('Run %d skipped: ROC AUC could not be calculated (single class).\n', i);
        validationAUC_GentleBoost = NaN;
        continue; 
    else
        [~, ~, ~, validationAUC_GentleBoost] = perfcurve(val_labels, val_scores, 1);
    end

    validationAUCValuesGentleBoost(i) = validationAUC_GentleBoost;

end

meanTestAUC_GentleBoost = mean(testAUCValuesGentleBoost);
stdTestAUC_GentleBoost = std(testAUCValuesGentleBoost);

meanValidationAUC_GentleBoost = mean(validationAUCValuesGentleBoost);
stdValidationAUC_GentleBoost = std(validationAUCValuesGentleBoost);

fprintf('GentleBoost Test Mean AUC: %.4f\n', meanTestAUC_GentleBoost);
fprintf('GentleBoost Test AUC Standard Deviation: %.4f\n', stdTestAUC_GentleBoost);
fprintf('GentleBoost Validation Mean AUC: %.4f\n', meanValidationAUC_GentleBoost);
fprintf('GentleBoost Validation AUC Standard Deviation: %.4f\n', stdValidationAUC_GentleBoost);

fileID = fopen('auc_results_GentleBoost.txt', 'w');
fprintf(fileID, 'GentleBoost Test Mean AUC: %.4f\n', meanTestAUC_GentleBoost);
fprintf(fileID, 'GentleBoost Test AUC Standard Deviation: %.4f\n', stdTestAUC_GentleBoost);
fprintf(fileID, 'GentleBoost Validation Mean AUC: %.4f\n', meanValidationAUC_GentleBoost);
fprintf(fileID, 'GentleBoost Validation AUC Standard Deviation: %.4f\n', stdValidationAUC_GentleBoost);
fclose(fileID);


%% Confusion matrix analysis
empty_idx = [];
for i = 1:size(test_confmat_GentleBoost_all, 3)
    if all(test_confmat_GentleBoost_all(:,:,i) == 0, 'all')
        empty_idx(end+1) = i;
    end
end

nan_idx = [];
for i = 1:size(test_confmat_GentleBoost_all, 3)
    if any(isnan(test_confmat_GentleBoost_all(:,:,i)), 'all')
        nan_idx(end+1) = i;
    end
end

valid_idx = setdiff(1:size(test_confmat_GentleBoost_all, 3), union(empty_idx, nan_idx));
valid_confmats = test_confmat_GentleBoost_all(:,:,valid_idx);
avg_confmat_test_GentleBoost = mean(valid_confmats, 3);

% Confusion Matrix - GentleBoost 
figure('Units', 'pixels', 'Position', [100, 100, 1200, 1000]);
imagesc(avg_confmat_test_GentleBoost);
colormap(turbo);
colorbar;
title('GentleBoost - Mean Confusion Matrix','FontSize', 44, 'FontWeight', 'bold');

xlabel('Predicted Class', 'FontSize', 40, 'FontWeight', 'bold');
ylabel('Actual Class','FontSize', 40, 'FontWeight', 'bold');

set(gca, 'FontSize', 34, 'FontWeight', 'bold');

xticks([1 2]);
yticks([1 2]);
xticklabels({'Negative (N)', 'Positive (P)'});
yticklabels({'Negative (N)', 'Positive (P)'});

text(1,1, sprintf('%.0f', avg_confmat_test_GentleBoost(1,1)), 'FontSize', 60, 'HorizontalAlignment', 'center', 'Color', 'w');
text(1,2, sprintf('%.0f', avg_confmat_test_GentleBoost(1,2)), 'FontSize', 60, 'HorizontalAlignment', 'center', 'Color', 'w');
text(2,1, sprintf('%.0f', avg_confmat_test_GentleBoost(2,1)), 'FontSize', 60, 'HorizontalAlignment', 'center', 'Color', 'w');
text(2,2, sprintf('%.0f', avg_confmat_test_GentleBoost(2,2)), 'FontSize', 60, 'HorizontalAlignment', 'center', 'Color', 'w');

set(gcf, 'Color', 'w');

print(gcf, 'GentleBoost_ConfusionMatrix', '-dpng', '-r300');

save('GentleBoost_results1-100.mat')

% Overall Evaluation of SHAP Results
% For each feature: mean SHAP, standard deviation, and selection count

meanShap = nanmean(shapleyResultsFull, 1);
stdShap = nanstd(shapleyResultsFull, 0, 1);
selectionCount = sum(~isnan(shapleyResultsFull), 1);

[~, sortIdx] = sort(meanShap, 'descend');
sortedFeatureNames = allFeatureNames(sortIdx);
sortedMeanShap = meanShap(sortIdx);
sortedStdShap = stdShap(sortIdx);
sortedSelectionCount = selectionCount(sortIdx);

shapTable = table(sortedFeatureNames', sortedMeanShap', sortedStdShap', sortedSelectionCount', ...
    'VariableNames', {'Feature', 'MeanSHAP', 'StdSHAP', 'SelectionCount'});

writetable(shapTable, 'SHAP_Feature_Importance_AllRuns.xlsx');

% selection count >=75 features (Consistently Contributive Features)
kararli_idx = (selectionCount >= 75);
biomarkerFeatures = allFeatureNames(kararli_idx);
biomarkerShapValues = meanShap(kararli_idx);

[shapSorted, sortBiomIdx] = sort(biomarkerShapValues, 'descend');
biomarkerFeaturesSorted = biomarkerFeatures(sortBiomIdx);

figure('Position', [100, 100, 900, 600]);
barh(shapSorted);
set(gca, 'ytick', 1:length(biomarkerFeaturesSorted), ...
    'yticklabel', biomarkerFeaturesSorted, ...
    'YDir', 'reverse');
xlabel('Mean SHAP Value');
ylabel('Feature');
title('Consistently Contributive Features');
saveas(gcf, 'SHAP_Feature_Importance_Biomarkers.png');

T = array2table(shapleyResultsFull, 'VariableNames', allFeatureNames);

iterationLabels = strcat("Iter_", string(1:100))';
T = addvars(T, iterationLabels, 'Before', 1, 'NewVariableNames', 'Iteration');

writetable(T, 'SHAP_Iterative_Values.xlsx');

save('GentleBoost_results1-100.mat')


% Heatmap of SHAP Values for Features Selected ≥ 75%
threshold = 75;

selectedIdx = find(selectionCount >= threshold);

filteredSHAP = shapleyResultsFull(:, selectedIdx);
filteredNames = allFeatureNames(selectedIdx);

figure('Units', 'pixels', 'Position', [100 100 1600 900]); 
imagesc(filteredSHAP);
colorbar;
xlabel('Features','FontSize', 24, 'FontWeight', 'bold');
ylabel('Iterations','FontSize', 24, 'FontWeight', 'bold');
xticks(1:length(filteredNames));
xticklabels(filteredNames);
xtickangle(45);
title('Heatmap of SHAP Values for Features Selected ≥ 75%','FontSize', 28, 'FontWeight', 'bold');

ax = gca;
ax.FontSize = 18;              
ax.XAxis.FontSize = 20;        

ax.YAxis.FontSize = 20;        


box on;
set(gcf, 'Color', 'w');

print(gcf, 'SHAP_Heatmap_SelectedFeatures', '-dpng', '-r300');

save('GentleBoost_results1-100.mat')


save('GentleBoost_NLC100_LR0.1.mat', 'allScoresGentleBoost', 'allLabelsGentleBoost');

save('GentleBoost_NLC100_LR0.1_AUC.mat', 'testAUCValuesGentleBoost');

save('GentleBoost_NLC100_LR0.1_accuracy.mat', 'testGentleBoostAccuracies');

save('GentleBoost_NLC100_LR0.1_F1.mat', 'testGentleBoostF1Scores');

save('GentleBoost_NLC100_LR0.1_precision.mat', 'testGentleBoostPrecisions');

save('GentleBoost_NLC100_LR0.1_recall.mat', 'testGentleBoostRecalls');

save('GentleBoost_NLC100_LR0.1_specificity.mat', 'testGentleBoostSpecificities');



%% CI + SHAP Correlation Summary
modelTypes = {'GentleBoost'};
sheetNames = {'GentleBoost_CI'};

metricNames = {'AUC', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity'};

outputExcel = 'CI_Results_AllModels.xlsx';

for i = 1:length(modelTypes)
    model = modelTypes{i};

    testMetrics.AUC         = eval(['testAUCValues' model]);
    testMetrics.Accuracy    = eval(['test' model 'Accuracies']);
    testMetrics.F1          = eval(['test' model 'F1Scores']);
    testMetrics.Precision   = eval(['test' model 'Precisions']);
    testMetrics.Recall      = eval(['test' model 'Recalls']);
    testMetrics.Specificity = eval(['test' model 'Specificities']);

    valMetrics.AUC          = eval(['validationAUCValues' model]);
    valMetrics.Accuracy     = eval(['validation' model 'Accuracies']);
    valMetrics.F1           = eval(['validation' model 'F1Scores']);
    valMetrics.Precision    = eval(['validation' model 'Precisions']);
    valMetrics.Recall       = eval(['validation' model 'Recalls']);
    valMetrics.Specificity  = eval(['validation' model 'Specificities']);


    alpha = 0.05;
    results = {};
    for j = 1:length(metricNames)
        metric = metricNames{j};

        [muTest, loTest, hiTest, methTest, pTest] = smartCI(testMetrics.(metric), alpha);
        results(end+1, :) = {metric, 'Test', muTest, loTest, hiTest, methTest, pTest};

        [muVal, loVal, hiVal, methVal, pVal] = smartCI(valMetrics.(metric), alpha);
        results(end+1, :) = {metric, 'Validation', muVal, loVal, hiVal, methVal, pVal};
    end

    CI_Table = cell2table(results, 'VariableNames', ...
        {'Metric', 'Dataset', 'Mean', 'LowerCI', 'UpperCI', 'Method', 'pNormal'});

    writetable(CI_Table, outputExcel, 'Sheet', sheetNames{i});

    shapVarName = ['shapCorrelation' model]; 
    pvalVarName = ['shapCorrelationPval' model];
    
    if evalin('base', sprintf('exist(''%s'', ''var'')', shapVarName))
        rhoAll = eval(shapVarName);
        pvalAll = eval(pvalVarName);

        mean_rho = mean(rhoAll);
        significant_ratio = mean(pvalAll < 0.05) * 100;

        shapSummary = table({'SHAP Correlation'}, mean_rho, significant_ratio, ...
            'VariableNames', {'Metric', 'MeanRho', 'SignificantPvalRatio'});

        writetable(shapSummary, outputExcel, 'Sheet', [sheetNames{i} '_SHAP']);
        fprintf('SHAP correlation summary added Page: %s_SHAP\n', sheetNames{i});
    end

    fprintf('%s all tables saved for the model.\n', model);
end


T_features = table((1:numRuns)', numSelectedFeatures, ...
    'VariableNames', {'Run', 'NumSelectedFeatures'});
writetable(T_features, 'selected_feature_counts.xlsx');


elapsed = toc;          
minutes = elapsed / 60;
disp(['Elapsed time: ', num2str(minutes), ' minutes']);

save('GentleBoost_results1-100.mat')

