@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >NUL

REM ====== 폴더 생성 경로 설정 ======
REM 이 경로를 원하는 위치로 변경하세요.
set "ROOT=C:\projects\ml-dl-hands-on"
REM ================================

echo [INFO] 폴더 생성 위치: %ROOT%
mkdir "%ROOT%" 2>nul

REM ---------- 5~19장 폴더 + README 생성 ----------
call :mk "Logistic Regression for Classification Tasks"
call :mk "Decision Trees for Predictive Modeling"
call :mk "Ensemble Learning with Random Forests and Gradient Boosting"
call :mk "Dimensionality Reduction with PCA and t-SNE"
call :mk "Unsupervised Learning with Clustering Algorithms"
call :mk "Introduction to Artificial Neural Networks"
call :mk "Deep Neural Network Optimization and Regularization"
call :mk "Custom Model Building with Keras Functional API"
call :mk "Convolutional Neural Networks for Image Recognition"
call :mk "Sequence Modeling with RNN, LSTM, and GRU"
call :mk "Natural Language Processing with RNN and Word Embeddings"
call :mk "Reinforcement Learning with Deep Q-Networks"
call :mk "Autoencoders for Data Compression and Anomaly Detection"
call :mk "Generative Models with GANs"
call :mk "Deploying Machine Learning Models to Production"

REM ---------- docs 폴더 ----------
mkdir "%ROOT%\docs" 2>nul
if not exist "%ROOT%\docs\learning_log.md" (
    >"%ROOT%\docs\learning_log.md" (
        echo # Learning Log
        echo - Date: YYYY-MM-DD
        echo - Summary: what I tried
        echo - Results: metrics / images
        echo - Next: todo
    )
)

REM ---------- 메인 README ----------
>"%ROOT%\README.md" (
    echo # ML/DL Hands-On Practice
    echo.
    echo This repository contains my portfolio-style practice projects for machine learning and deep learning.
    echo.
    echo ## Chapters
    echo 5 - Logistic Regression for Classification Tasks
    echo 6 - Decision Trees for Predictive Modeling
    echo 7 - Ensemble Learning with Random Forests and Gradient Boosting
    echo 8 - Dimensionality Reduction with PCA and t-SNE
    echo 9 - Unsupervised Learning with Clustering Algorithms
    echo 10 - Introduction to Artificial Neural Networks
    echo 11 - Deep Neural Network Optimization and Regularization
    echo 12 - Custom Model Building with Keras Functional API
    echo 13 - Convolutional Neural Networks for Image Recognition
    echo 14 - Sequence Modeling with RNN, LSTM, and GRU
    echo 15 - Natural Language Processing with RNN and Word Embeddings
    echo 16 - Reinforcement Learning with Deep Q-Networks
    echo 17 - Autoencoders for Data Compression and Anomaly Detection
    echo 18 - Generative Models with GANs
    echo 19 - Deploying Machine Learning Models to Production
)

echo [DONE] 모든 폴더와 README 생성 완료!
pause
exit /b

:mk
REM %~1 = 폴더명
set "title=%~1"
set "slug=%title%"
set "slug=!slug: =_!"
set "slug=!slug:,=!"
set "slug=!slug:&=and!"
set "slug=!slug:/=-!"
set "slug=!slug:(=!"
set "slug=!slug:)=!"

mkdir "%ROOT%\!slug!" 2>nul
mkdir "%ROOT%\!slug!\results" 2>nul

>"%ROOT%\!slug!\README.md" (
    echo # %title%
    echo.
    echo ## Overview
    echo Brief summary of this project.
    echo.
    echo ## How to Run
    echo ```bash
    echo jupyter notebook
    echo ```
    echo.
    echo ## Results
    echo - Metrics: TBD
    echo.
    echo ![Result](results/result1.png)
)
exit /b
