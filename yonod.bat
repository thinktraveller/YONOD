@echo off
setlocal enabledelayedexpansion
:: GBK 编码保存文件，cmd.exe 按系统默认代码页读取
:: set /p 提示字符串中不使用 ( ) < > 等 cmd 特殊字符

title YONOD 通用向导

echo.
echo =====================================================
echo   YONOD - Your One-stop Notebook Of Descriptors
echo   通用向导
echo =====================================================
echo.

:: 步骤 1：CSV 路径（去除拖拽时可能带的首尾引号）
set /p CSV_PATH=[1/8] 输入CSV路径，可直接拖拽文件到此处后回车: 
set CSV_PATH=%CSV_PATH:"=%

:: 步骤 2：标签列名称
set /p LABEL_COL=[2/8] 输入标签列名称，如 yield, ee, delta_G: 

:: 步骤 3：SMILES 列（可选）
set /p SMILES_COLS=[3/8] SMILES列名，空格分隔，留空自动探测: 

:: 步骤 4：数值辅助列（可选）
set /p NUMERIC_COLS=[4/8] 数值辅助列名，空格分隔，留空跳过: 

:: 步骤 5：输出目录（可选）
set /p OUTPUT_DIR=[5/8] 输出目录路径，留空使用默认路径: 

:: 步骤 6：任务名称（可选）
set /p TASK_NAME=[6/8] 任务名称，用于报告标题，留空用CSV文件名: 

:: 步骤 7：描述符选择（可选）
echo [7/8] 可选描述符: morgan  maccs  fisd  molmetalm
set /p DESCS=       请选空格分隔，留空=全选: 

:: 步骤 8：模型选择（可选）
echo [8/8] 可选模型: xgb  rf  svm  autogluon
set /p MODELS=       请选空格分隔，留空=全选: 

:: 拼接命令（使用 !CMD! 延迟展开，避免 if 块内变量展开错误）
set CMD=python run_yonod.py --csv "%CSV_PATH%" --label-col "%LABEL_COL%"

if not "%SMILES_COLS%"=="" set CMD=!CMD! --smiles-cols %SMILES_COLS%
if not "%NUMERIC_COLS%"=="" set CMD=!CMD! --numeric-cols %NUMERIC_COLS%
if not "%OUTPUT_DIR%"=="" set CMD=!CMD! --output-dir "%OUTPUT_DIR%"
if not "%TASK_NAME%"=="" set CMD=!CMD! --task-name "%TASK_NAME%"
if not "%DESCS%"=="" set CMD=!CMD! --descriptors %DESCS%
if not "%MODELS%"=="" set CMD=!CMD! --models %MODELS%

echo.
echo =====================================================
echo   即将执行的命令:
echo   !CMD!
echo =====================================================
echo.
pause

:: 激活 conda 环境
call conda activate yonod-yield
if %ERRORLEVEL% NEQ 0 (
    echo [错误] conda activate yonod-yield 失败，请确认环境名称是否正确。
    pause
    exit /b 1
)

!CMD!

echo.
echo [完成] 任务已完成，按任意键关闭窗口。
pause