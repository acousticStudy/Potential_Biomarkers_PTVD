function generateCITable(testMetrics, valMetrics, metricNames, saveName)
% generateCITable - Computes and exports a comparative confidence interval (CI)
% table for test and validation metrics, using parametric or bootstrap methods
% based on normality assessment.
%
% This function compares the generalization performance between test and
% validation (cross-validation) results. It automatically selects the CI method
% for each metric depending on the result of a Shapiro-Wilk normality test.
%
% Inputs:
%   testMetrics   - Struct containing vectors of test set metric values
%   valMetrics    - Struct containing vectors of validation metric values
%   metricNames   - Cell array of metric names (e.g., {'AUC','Accuracy',...})
%   saveName      - Desired output Excel file name (e.g., 'CI_results.xlsx')
%
% Output:
%   An Excel file containing the following columns:
%     - Metric     : Name of the evaluated metric
%     - Dataset    : 'Test' or 'Validation'
%     - Mean       : Mean of the metric across runs
%     - LowerCI    : Lower bound of the 95% CI
%     - UpperCI    : Upper bound of the 95% CI
%     - Method     : 'parametric' or 'bootstrap'
%     - pNormal    : p-value of Shapiro-Wilk normality test
%
% Note:
%   This function requires `smartCI.m` (and `swtest.m`) to be available.

    alpha = 0.05;  % Default: 95% confidence interval
    results = {};

    for i = 1:length(metricNames)
        metric = metricNames{i};

        % -- TEST set confidence interval --
        values_test = testMetrics.(metric);
        [muTest, loTest, hiTest, methodTest, pTest] = smartCI(values_test, alpha);
        results(end+1, :) = {metric, 'Test', muTest, loTest, hiTest, methodTest, pTest};

        % -- VALIDATION set confidence interval --
        values_val = valMetrics.(metric);
        [muVal, loVal, hiVal, methodVal, pVal] = smartCI(values_val, alpha);
        results(end+1, :) = {metric, 'Validation', muVal, loVal, hiVal, methodVal, pVal};
    end

    % Convert results to a table
    T = cell2table(results, 'VariableNames', ...
        {'Metric', 'Dataset', 'Mean', 'LowerCI', 'UpperCI', 'Method', 'pNormal'});

    % Export to Excel
    writetable(T, saveName);

    fprintf('Confidence interval table successfully saved to "%s"\n', saveName);
end



% Confidence intervals for each evaluation metric (AUC, accuracy, precision, etc.) 
% were computed using a custom MATLAB function. 
% The method employed the Shapiro-Wilk test to assess normality. 
% If the metric distribution was normal (p > 0.05), a parametric 
% CI was calculated using the standard error of the mean. 
% Otherwise, a 1000-sample bootstrap CI was used. 
% All results were tabulated and exported using the generateCITable utility.

