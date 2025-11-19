@echo off
setlocal enabledelayedexpansion

call :run_and_verify ./data/measurements_50mil.txt ./50mil.out b88372780b09030bf46cf490223b300248878a27497b1c92d257fb29ff15bc75
call :run_and_verify ./data/measurements_300mil.txt ./300mil.out f5a3358fb3f59c5dd2b7403f8b1de4a82ec2e3e165d98800b1d2225768d4f1ef
call :run_and_verify ./data/measurements_full.txt ./full.out 31ebb58f6ec1a90169a06ef8b4e03cedd9698e6acc4c055bb280cf6526a556a5

exit /b %ERRORLEVEL%

:run_and_verify
set INPUT_FILE=%~1
set OUTPUT_FILE=%~2
set EXPECTED_HASH=%~3

echo Running processor with input: %INPUT_FILE% output: %OUTPUT_FILE%
..\src\build\Release\processor.exe %INPUT_FILE% %OUTPUT_FILE%

if errorlevel 1 (
    echo Processor failed
    exit /b 2
)

echo Computing hash of %OUTPUT_FILE%
for /f "skip=1 tokens=1" %%h in ('certutil -hashfile %OUTPUT_FILE% SHA256') do (
    set ACTUAL_HASH=%%h
    goto :hash_done
)
:hash_done

set ACTUAL_HASH=!ACTUAL_HASH: =!

echo Expected: %EXPECTED_HASH%
echo Actual:   !ACTUAL_HASH!

if /i "!ACTUAL_HASH!"=="%EXPECTED_HASH%" (
    echo PASS
    exit /b 0
) else (
    echo FAIL
    exit /b 2
)