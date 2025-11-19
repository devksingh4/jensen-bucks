@echo off
setlocal enabledelayedexpansion

call :run_and_verify ./data/measurements_50mil.txt ./50mil.out 97825d5cef3cf93e079f9da1cab01ee59c0436ebafaaa364f39dde6e3ed395ee
call :run_and_verify ./data/measurements_300mil.txt ./300mil.out 46cd11387fbcac86bd3e23b506d65da0c656167cbd633afca6d9facca372af44
call :run_and_verify ./data/measurements_full.txt ./full.out 9c9cf2a419590f947f3584cbf6354e34ddb7aa8ba235540a2e3017e56f3048d3

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