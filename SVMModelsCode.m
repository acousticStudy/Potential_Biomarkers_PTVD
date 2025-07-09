clc;
clear all;
close all;
% warning off;

% To enhance transparency and reproducibility, code was provided with inline comments. 
% For brevity, some repetitive blocks were not modularized but have been rigorously tested.

% The SVM model configuration can be flexibly adjusted by modifying the kernel-related parameters.
% For instance, setting 'KernelFunction' to 'polynomial' with 'PolynomialOrder' set to 3 yields a Cubic SVM,
% while setting the same to 2 results in a Quadratic SVM. Alternatively, specifying 'KernelFunction' as 'rbf'
% enables the use of the Radial Basis Function (Gaussian) kernel.
% Furthermore, the regularization strength of the SVM is controlled via the 'BoxConstraint' parameter,
% which can be set to values such as 0.01, 0.1, or 1 to regulate the trade-off between margin maximization and classification error.

tic;
seed = 2;
rng(seed, 'twister');

numFolds = 3;
numRuns = 100;

testTP_Cubic_values = zeros(numRuns, 1);
testFP_Cubic_values = zeros(numRuns, 1);
testTN_Cubic_values = zeros(numRuns, 1);
testFN_Cubic_values = zeros(numRuns, 1);

test_confmat_Cubic_all = zeros(2, 2, numRuns);

allScoresCubic = [];
allLabelsCubic = [];

validationScoresCubic = [];
validationLabelsCubic = [];

validationCubicAccuracies = zeros(numFolds, numRuns);
validationCubicPrecisions = zeros(numFolds, numRuns);
validationCubicRecalls = zeros(numFolds, numRuns);
validationCubicSpecificities = zeros(numFolds, numRuns);
validationCubicF1Scores = zeros(numFolds, numRuns);

testCubicAccuracies = zeros(numRuns, 1);
testCubicPrecisions = zeros(numRuns, 1);
testCubicRecalls = zeros(numRuns, 1);
testCubicSpecificities = zeros(numRuns, 1);
testCubicF1Scores = zeros(numRuns, 1);

data = readtable('Dataset.xlsx');
Xall = table2array(data(:, 2:end));
yall = table2array(data(:, 1));

successfulRuns = 0;
attemptedRuns = 0;

allLabelsValCubic = [];
allPredsValCubic = [];

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

    if sum(selectedLasso) < 10
        fprintf('Not enough features left after LASSO.\n');
        continue;
    end


    numSelectedFeatures(run) = sum(selectedLasso);  

    % Names of all features
    allFeatureNames = data.Properties.VariableNames(2:end);
    numAllFeatures = numel(allFeatureNames);

    selectedFeatureIndices = find(selectedLasso);
    selectedFeatureNames = allFeatureNames(selectedFeatureIndices);

    % Filter last selected features
    Xtrain_final = Xtrain_z(:, selectedLasso);
    Xtest_final = Xtest_z(:, selectedLasso);

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
            
            % Cubic SVM
            SVMModelCubic = fitcsvm(tblTrainFold, ytrain_fold, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'BoxConstraint', 1, 'KernelScale', 'auto', 'Standardize', false, 'Solver', 'SMO', 'ClassNames', unique(ytrain_fold));
            [ypred_validation_Cubic, scoresCubicValidation] = predict(SVMModelCubic, tblValidation);

            validation_confmat_Cubic = confusionmat(yvalidation, ypred_validation_Cubic);

            % If the confusion matrix is not 2x2, fill in the missing cells.
            if size(validation_confmat_Cubic, 1) < 2
                validation_confmat_Cubic(2, 2) = 0;
            end
            if size(validation_confmat_Cubic, 2) < 2
                validation_confmat_Cubic(2, 2) = 0;
            end

            validationTP_Cubic = validation_confmat_Cubic(2, 2);
            validationFP_Cubic = validation_confmat_Cubic(1, 2);
            validationTN_Cubic = validation_confmat_Cubic(1, 1);
            validationFN_Cubic = validation_confmat_Cubic(2, 1);

            validationAccuracy_Cubic = (validationTP_Cubic + validationTN_Cubic) / (validationTP_Cubic + validationFP_Cubic + validationTN_Cubic + validationFN_Cubic);
            validationprecision_Cubic = validationTP_Cubic / (validationTP_Cubic + validationFP_Cubic);
            validationrecall_Cubic = validationTP_Cubic / (validationTP_Cubic + validationFN_Cubic);
            validationspecificity_Cubic = validationTN_Cubic / (validationTN_Cubic + validationFP_Cubic);
            validationF1_Cubic = 2 * (validationprecision_Cubic * validationrecall_Cubic) / (validationprecision_Cubic + validationrecall_Cubic);

            if validationAccuracy_Cubic == 0
                validationAccuracy_Cubic = NaN;
            end

            if validationprecision_Cubic == 0
                validationprecision_Cubic = NaN;
            end

            if validationrecall_Cubic == 0
                validationrecall_Cubic = NaN;
            end

            if validationspecificity_Cubic == 0
                validationspecificity_Cubic = NaN;
            end

            if validationF1_Cubic == 0
                validationF1_Cubic = NaN;
            end

            % NaN check
            skipIteration1 = false;

            if isnan(validationAccuracy_Cubic) || isnan(validationprecision_Cubic) || isnan(validationrecall_Cubic) || isnan(validationspecificity_Cubic) || isnan(validationF1_Cubic)
                skipIteration1 = true;
            else

                validationCubicAccuracies(fold, run) = validationAccuracy_Cubic*100;
                validationCubicPrecisions(fold, run) = validationprecision_Cubic*100;
                validationCubicRecalls(fold, run) = validationrecall_Cubic*100;
                validationCubicSpecificities(fold, run) = validationspecificity_Cubic*100;
                validationCubicF1Scores(fold, run) = validationF1_Cubic*100;

                validationScoresCubic = [validationScoresCubic; scoresCubicValidation(:,2)];
                validationLabelsCubic = [validationLabelsCubic; yvalidation];
            end

            % If NaN value is present skip this iteration
            if skipIteration1
                fprintf('Run %d, Fold %d skipped due to NaN values.\n', attemptedRuns, fold);
                continue;
            end

  
            % If all skipIteration flags are false, increase the count of successful iterations.
            if ~(skipIteration1)
                successfulFolds = successfulFolds + 1;
                allLabelsValCubic = [allLabelsValCubic; yvalidation];
            end

        end
    end


    %% For the Test Data
    Xtrain_final_selected = Xtrain_z(:, selectedFeatureIndices);
    Xtest_final_selected = Xtest_z(:, selectedFeatureIndices);

    tblTrainFinal = array2table(Xtrain_final_selected, 'VariableNames', selectedFeatureNames);
    tblTestFinal = array2table(Xtest_final_selected, 'VariableNames', selectedFeatureNames);
    
    SVMModelCubicTest = fitcsvm(tblTrainFinal, ytrain, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'BoxConstraint', 1, 'KernelScale', 'auto', 'Standardize', false, 'Solver', 'SMO', 'ClassNames', unique(ytrain_fold));

    [~, scoresCubic] = predict(SVMModelCubicTest, tblTestFinal);

    % Store the scores and the labels.
    allScoresCubic = [allScoresCubic; scoresCubic(:,2)];
    allLabelsCubic = [allLabelsCubic; ytest];

    ypred_test_Cubic = predict(SVMModelCubicTest, tblTestFinal);

    confmat_test_Cubic = confusionmat(ytest, ypred_test_Cubic);
    
    % If the confusion matrix is not 2x2, fill in the missing cells.

    if size(confmat_test_Cubic, 1) < 2
        confmat_test_Cubic(2, 2) = 0;
    end
    if size(confmat_test_Cubic, 2) < 2
        confmat_test_Cubic(2, 2) = 0;
    end

    testTP_Cubic = confmat_test_Cubic(2, 2);
    testFP_Cubic = confmat_test_Cubic(1, 2);
    testTN_Cubic = confmat_test_Cubic(1, 1);
    testFN_Cubic = confmat_test_Cubic(2, 1);

    accuracy_test_Cubic = (testTP_Cubic + testTN_Cubic) / (testTP_Cubic + testFP_Cubic + testTN_Cubic + testFN_Cubic);
    precision_test_Cubic = testTP_Cubic / (testTP_Cubic + testFP_Cubic);
    recall_test_Cubic = testTP_Cubic / (testTP_Cubic + testFN_Cubic);
    specificity_test_Cubic = testTN_Cubic / (testTN_Cubic + testFP_Cubic);
    testF1_Cubic = 2 * (precision_test_Cubic * recall_test_Cubic) / (precision_test_Cubic + recall_test_Cubic);

    % After replacing zero-valued metrics with NaN (to indicate structurally undefined or uninformative results),
    % a NaN check is performed before counting the iteration as successful.
    % This ensures that only iterations with valid and interpretable performance metrics are included in the final evaluation.

    if accuracy_test_Cubic == 0
        accuracy_test_Cubic = NaN;
    end

    if precision_test_Cubic == 0
        precision_test_Cubic = NaN;
    end

    if recall_test_Cubic == 0
        recall_test_Cubic = NaN;
    end

    if specificity_test_Cubic == 0
        specificity_test_Cubic = NaN;
    end

    if testF1_Cubic == 0
        testF1_Cubic = NaN;
    end


    fprintf('Cubic SVM Test Accuracy: %.2f\n', accuracy_test_Cubic*100);
    fprintf('Cubic SVM Test Precision: %.2f\n', precision_test_Cubic*100);
    fprintf('Cubic SVM Test Recall: %.2f\n', recall_test_Cubic*100);
    fprintf('Cubic SVM Test Specificity: %.2f\n', specificity_test_Cubic*100);
    fprintf('Cubic SVM Test F1-score: %.2f\n', testF1_Cubic*100);

    % NaN check
    skipIteration3 = false;

    if isnan(accuracy_test_Cubic) || isnan(precision_test_Cubic) || isnan(recall_test_Cubic) || isnan(specificity_test_Cubic) || isnan(testF1_Cubic)
        skipIteration3 = true;
    else
        testCubicAccuracies(run) = accuracy_test_Cubic*100;
        testCubicPrecisions(run) = precision_test_Cubic*100;
        testCubicRecalls(run) = recall_test_Cubic*100;
        testCubicSpecificities(run) = specificity_test_Cubic*100;
        testCubicF1Scores(run) = testF1_Cubic*100;
    end

    if skipIteration3
        fprintf('Run %d skipped due to NaN values.\n', attemptedRuns);
        continue;
    end

    successfulRuns = successfulRuns + 1;

    test_confmat_Cubic_all(:, :, run) = confmat_test_Cubic;
   
    %% Shapley
   
    selectedNames = selectedFeatureNames;

    blackbox_test = SVMModelCubicTest;  % Use the trained model in SHAP
    
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
    
    explainerTrain = shapley(SVMModelCubicTest, 'QueryPoints', dataTrain);
    shapMatrixTrain = table2array(explainerTrain.ShapleyValues(:, "1"));  % For Class 1
    averageShapleyTrain = mean(shapMatrixTrain, 2)';

    shapleyTableTrain = array2table(averageShapleyTrain, 'VariableNames', predictorNames);


    % Test SHAP

    Xtest_selected = Xtest_final;
    

    dataTest = array2table(Xtest_selected, 'VariableNames', predictorNames);
    numQueryPoints = size(dataTest, 1);

    explainerTest = shapley(SVMModelCubicTest, 'QueryPoints', dataTest);
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




fileID = fopen('Cubic_svm_results1-100.txt','w');

% Compute the average results
validation_avgCubicAccuracy = mean(validationCubicAccuracies(:));
validation_avgCubicPrecision = mean(validationCubicPrecisions(:));
validation_avgCubicRecall = mean(validationCubicRecalls(:));
validation_avgCubicSpecificity = mean(validationCubicSpecificities(:));
validation_avgCubicF1 = mean(validationCubicF1Scores(:));

validation_stdCubicAccuracy = std(validationCubicAccuracies(:));
validation_stdCubicPrecision = std(validationCubicPrecisions(:));
validation_stdCubicRecall = std(validationCubicRecalls(:));
validation_stdCubicSpecificity = std(validationCubicSpecificities(:));
validation_stdCubicF1 = std(validationCubicF1Scores(:));

fprintf(fileID, '(Validation) Cubic SVM Accuracy: Mean = %.2f\n, STD = %2f\n', validation_avgCubicAccuracy, validation_stdCubicAccuracy);
fprintf(fileID, '(Validation) Cubic SVM Precision: Mean = %.2f\n, STD = %2f\n', validation_avgCubicPrecision, validation_stdCubicPrecision);
fprintf(fileID, '(Validation) Cubic SVM Recall: Mean = %.2f\n, STD = %2f\n', validation_avgCubicRecall, validation_stdCubicRecall);
fprintf(fileID, '(Validation) Cubic SVM Specificity: Mean = %.2f\n, STD = %2f\n', validation_avgCubicSpecificity, validation_stdCubicSpecificity);
fprintf(fileID, '(Validation) Cubic SVM F1-score: Mean = %.2f\n, STD = %2f\n', validation_avgCubicF1, validation_stdCubicF1);


meanTestCubicAcc = mean(testCubicAccuracies(:));
stdTestCubicAcc = std(testCubicAccuracies(:));

meanTestCubicPrec = mean(testCubicPrecisions(:));
stdTestCubicPrec = std(testCubicPrecisions(:));

meanTestCubicRecall = mean(testCubicRecalls(:));
stdTestCubicRecall = std(testCubicRecalls(:));

meanTestCubicSpec = mean(testCubicSpecificities(:));
stdTestCubicSpec = std(testCubicSpecificities(:));

meanTestCubicF1 = mean(testCubicF1Scores(:));
stdTestCubicF1 = std(testCubicF1Scores(:));

fileID = fopen('Cubic_svm_results1-100.txt','a');

fprintf(fileID, '\nCubic SVM Test Accuracy: Mean = %f, Std = %f\n', meanTestCubicAcc, stdTestCubicAcc);
fprintf(fileID, 'Cubic SVM Test Precision: Mean = %f, Std = %f\n', meanTestCubicPrec, stdTestCubicPrec);
fprintf(fileID, 'Cubic SVM Test Recall: Mean = %f, Std = %f\n', meanTestCubicRecall, stdTestCubicRecall);
fprintf(fileID, 'Cubic SVM Test Specificity: Mean = %f, Std = %f\n', meanTestCubicSpec, stdTestCubicSpec);
fprintf(fileID, 'Cubic SVM Test F1-score: Mean = %f, Std = %f\n', meanTestCubicF1, stdTestCubicF1);

fclose(fileID);

testAUCValuesCubic = zeros(numRuns, 1);
validationAUCValuesCubic = zeros(numRuns, 1);

% Compute the AUC values for each run
for i = 1:numRuns
    step = floor(length(allLabelsCubic) / numRuns);
    startIndexCubic = (i - 1) * step + 1;
    endIndexCubic = i * step;

    labels = allLabelsCubic(startIndexCubic:endIndexCubic);
    scores = allScoresCubic(startIndexCubic:endIndexCubic);

    if numel(unique(labels)) < 2
        fprintf('Run %d skipped: ROC AUC could not be calculated (single class).\n', i);
        testAUC_Cubic = NaN;
        continue; 
    else
        [~, ~, ~, testAUC_Cubic] = perfcurve(labels, scores, 1);
    end

    testAUCValuesCubic(i) = testAUC_Cubic;

    stepVal = floor(length(validationLabelsCubic) / numRuns);
    startIndexValCubic = (i - 1) * stepVal + 1;
    endIndexValCubic = i * stepVal;

    val_labels = validationLabelsCubic(startIndexValCubic:endIndexValCubic);
    val_scores = validationScoresCubic(startIndexValCubic:endIndexValCubic);


    if numel(unique(val_labels)) < 2
        fprintf('Run %d skipped: ROC AUC could not be calculated (single class).\n', i);
        validationAUC_Cubic = NaN;
        continue; 
    else
        [~, ~, ~, validationAUC_Cubic] = perfcurve(val_labels, val_scores, 1);
    end

    validationAUCValuesCubic(i) = validationAUC_Cubic;

end

meanTestAUC_Cubic = mean(testAUCValuesCubic);
stdTestAUC_Cubic = std(testAUCValuesCubic);

meanValidationAUC_Cubic = mean(validationAUCValuesCubic);
stdValidationAUC_Cubic = std(validationAUCValuesCubic);

fprintf('Cubic SVM Test Mean AUC: %.4f\n', meanTestAUC_Cubic);
fprintf('Cubic SVM Test AUC Standard Deviation: %.4f\n', stdTestAUC_Cubic);
fprintf('Cubic SVM Validation Mean AUC: %.4f\n', meanValidationAUC_Cubic);
fprintf('Cubic SVM Validation AUC Standard Deviation: %.4f\n', stdValidationAUC_Cubic);

fileID = fopen('auc_results_Cubic.txt', 'w');
fprintf(fileID, 'Cubic SVM Test Mean AUC: %.4f\n', meanTestAUC_Cubic);
fprintf(fileID, 'Cubic SVM Test AUC Standard Deviation: %.4f\n', stdTestAUC_Cubic);
fprintf(fileID, 'Cubic SVM Validation Mean AUC: %.4f\n', meanValidationAUC_Cubic);
fprintf(fileID, 'Cubic SVM Validation AUC Standard Deviation: %.4f\n', stdValidationAUC_Cubic);
fclose(fileID);


%% Confusion matrix analysis
empty_idx = [];
for i = 1:size(test_confmat_Cubic_all, 3)
    if all(test_confmat_Cubic_all(:,:,i) == 0, 'all')
        empty_idx(end+1) = i;
    end
end

nan_idx = [];
for i = 1:size(test_confmat_Cubic_all, 3)
    if any(isnan(test_confmat_Cubic_all(:,:,i)), 'all')
        nan_idx(end+1) = i;
    end
end

valid_idx = setdiff(1:size(test_confmat_Cubic_all, 3), union(empty_idx, nan_idx));
valid_confmats = test_confmat_Cubic_all(:,:,valid_idx);
avg_confmat_test_Cubic = mean(valid_confmats, 3);

% Confusion Matrix - Cubic SVM
figure('Units', 'pixels', 'Position', [100, 100, 1200, 1000]);
imagesc(avg_confmat_test_Cubic);
colormap(turbo);
colorbar;
title('Cubic SVM - Mean Confusion Matrix','FontSize', 44, 'FontWeight', 'bold');

xlabel('Predicted Class', 'FontSize', 40, 'FontWeight', 'bold');
ylabel('Actual Class','FontSize', 40, 'FontWeight', 'bold');

set(gca, 'FontSize', 34, 'FontWeight', 'bold');

xticks([1 2]);
yticks([1 2]);
xticklabels({'Negative (N)', 'Positive (P)'});
yticklabels({'Negative (N)', 'Positive (P)'});

text(1,1, sprintf('%.0f', avg_confmat_test_Cubic(1,1)), 'FontSize', 60, 'HorizontalAlignment', 'center', 'Color', 'w');
text(1,2, sprintf('%.0f', avg_confmat_test_Cubic(1,2)), 'FontSize', 60, 'HorizontalAlignment', 'center', 'Color', 'w');
text(2,1, sprintf('%.0f', avg_confmat_test_Cubic(2,1)), 'FontSize', 60, 'HorizontalAlignment', 'center', 'Color', 'w');
text(2,2, sprintf('%.0f', avg_confmat_test_Cubic(2,2)), 'FontSize', 60, 'HorizontalAlignment', 'center', 'Color', 'w');

set(gcf, 'Color', 'w');

print(gcf, 'Cubic_ConfusionMatrix', '-dpng', '-r300');

save('Cubic_svm_results1-100.mat')

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
stable_idx = (selectionCount >= 75);
biomarkerFeatures = allFeatureNames(stable_idx);
biomarkerShapValues = meanShap(stable_idx);

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

save('Cubic_svm_results1-100.mat')


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

save('Cubic_svm_results1-100.mat')


save('Cubic_SVM_C_1.mat', 'allScoresCubic', 'allLabelsCubic');

save('Cubic_C_1_AUC.mat', 'testAUCValuesCubic');

save('Cubic_C_1_accuracy.mat', 'testCubicAccuracies');

save('Cubic_C_1_F1.mat', 'testCubicF1Scores');

save('Cubic_C_1_precision.mat', 'testCubicPrecisions');

save('Cubic_C_1_recall.mat', 'testCubicRecalls');

save('Cubic_C_1_specificity.mat', 'testCubicSpecificities');



%% CI + SHAP Correlation Summary
modelTypes = {'Cubic'};
sheetNames = {'Cubic_CI'};

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

save('Cubic_svm_results1-100.mat')




