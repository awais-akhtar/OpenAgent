@echo off
rem Phase 1 evidence capture for the local OLMoE Q8 route, then the closure verifier.
rem All output goes to build\phase1_capture.log.
cd /d "%~dp0.."
if not exist build mkdir build
set "LOG=%CD%\build\phase1_capture.log"
echo started %DATE% %TIME% > "%LOG%"
where python >> "%LOG%" 2>&1
python --version >> "%LOG%" 2>&1
python -c "import numpy, tokenizers; print(numpy.__version__, tokenizers.__version__)" >> "%LOG%" 2>&1
echo ==== capture ==== >> "%LOG%"
python -m openagent.adaptive_engine.phase1_capture olmoe --checkpoint models\olmoe-1b-7b-instruct --manifest models\olmoe-1b-7b-instruct\openagent-experts.q8.manifest.json --build-library build\native --prompt "The capital of France is" --max-tokens 4 >> "%LOG%" 2>&1
echo capture_exit=%ERRORLEVEL% >> "%LOG%"
echo ==== verify ==== >> "%LOG%"
python phase1_verify.py >> "%LOG%" 2>&1
echo verify_exit=%ERRORLEVEL% >> "%LOG%"
echo finished %DATE% %TIME% >> "%LOG%"
