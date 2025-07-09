function [meanVal, lowerCI, upperCI, methodUsed, pNormal] = smartCI(metricVector, alpha)
% smartCI - Computes the confidence interval (CI) for a given metric vector.
% The method automatically selects between parametric and bootstrap CI
% based on a normality test (Shapiro-Wilk).
%
% Syntax:
%   [meanVal, lowerCI, upperCI, methodUsed, pNormal] = smartCI(metricVector, alpha)
%
% Inputs:
%   metricVector - A numeric vector containing metric values (e.g., AUCs)
%   alpha        - Significance level for CI (default is 0.05 for 95% CI)
%
% Outputs:
%   meanVal      - Mean of the metric values
%   lowerCI      - Lower bound of the confidence interval
%   upperCI      - Upper bound of the confidence interval
%   methodUsed   - 'parametric' or 'bootstrap', depending on distribution
%   pNormal      - p-value from the Shapiro-Wilk normality test
%
% Requirements:
%   - The external function `swtest.m` must be available in the path
%
% Example:
%   [m, l, u, method, p] = smartCI(aucValues, 0.05)

    if nargin < 2
        alpha = 0.05;
    end

    % Ensure column vector and remove NaNs
    metricVector = metricVector(:);
    metricVector = metricVector(~isnan(metricVector));
    n = length(metricVector);

    if n < 3
        error('At least 3 observations are required to estimate a CI.');
    end

    % Normality test using Shapiro-Wilk
    [~, pNormal] = swtest(metricVector, alpha);

    meanVal = mean(metricVector);

    if pNormal > alpha
        % Normally distributed: use parametric z-based CI
        z = norminv(1 - alpha / 2);
        SEM = std(metricVector) / sqrt(n);
        lowerCI = meanVal - z * SEM;
        upperCI = meanVal + z * SEM;
        methodUsed = 'parametric';
    else
        % Not normal: use nonparametric bootstrap CI
        numBoots = 1000;
        rng(2);  % For reproducibility
        bootMeans = zeros(numBoots, 1);
        for i = 1:numBoots
            resample = metricVector(randi(n, n, 1));
            bootMeans(i) = mean(resample);
        end
        lowerCI = prctile(bootMeans, 100 * (alpha / 2));
        upperCI = prctile(bootMeans, 100 * (1 - alpha / 2));
        methodUsed = 'bootstrap';
    end
end
