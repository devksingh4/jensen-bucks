@echo off
setlocal enabledelayedexpansion

call :run_and_verify ./data/measurements_100mil.txt ./100mil.out 1822092ffe421c70bf9fb5faded5dcbeb5724f02b22ca8ff9bb9051fa2ec6c2d
call :run_and_verify ./data/measurements_500mil.txt ./500mil.out 1796d3ebcd8a46137557f3f306966dda49aad17e9f8e1101f2c47d1f1afbcdf5
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