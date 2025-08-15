@echo off
REM =========================================
REM 각 프로젝트 폴더 안에 notebooks, src, data, results 생성
REM =========================================

REM 최상위 경로 설정
set "BASE_PATH=C:\Users\Owner\Documents\ml-dl-hands-on"

REM 폴더 목록
setlocal enabledelayedexpansion
for %%F in (
"Unsupervised_Learning_with_Clustering_Algorithms"
"Autoencoders_for_Data_Compression_and_Anomaly_Detection"
"Convolutional_Neural_Networks_for_Image_Recognition"
"Custom_Model_Building_with_Keras_Functional_API"
"Decision_Trees_for_Predictive_Modeling"
"Deep_Neural_Network_Optimization_and_Regularization"
"Deploying_Machine_Learning_Models_to_Production"
"Dimensionality_Reduction_with_PCA_and_t-SNE"
"Ensemble_Learning_with_Random_Forests_and_Gradient_Boosting"
"Generative_Models_with_GANs"
"Image Digit Classification Pipeline"
"Introduction_to_Artificial_Neural_Networks"
"Logistic_Regression_for_Classification_Tasks"
"Natural_Language_Processing_with_RNN_and_Word_Embeddings"
"Predictive Modeling with Linear, Polynomial, and Regularized Regression"
"Real Estate Price Prediction Pipeline"
"Reinforcement_Learning_with_Deep_Q-Networks"
"Sequence_Modeling_with_RNN_LSTM_and_GRU"
) do (
    set "FOLDER=%%~F"
    if exist "%BASE_PATH%\!FOLDER!" (
        echo [!FOLDER!] 서브폴더 생성 중...
        mkdir "%BASE_PATH%\!FOLDER!\notebooks" 2>nul
        mkdir "%BASE_PATH%\!FOLDER!\src" 2>nul
        mkdir "%BASE_PATH%\!FOLDER!\data" 2>nul
        mkdir "%BASE_PATH%\!FOLDER!\results" 2>nul
    ) else (
        echo [WARN] 폴더 없음: !FOLDER!
    )
)

echo.
echo 작업 완료!
pause
