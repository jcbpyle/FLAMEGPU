echo off
echo "This will overwrite the existing logging_and_functions.c file!"
pause
XSLTProcessor.exe XMLModelFile.xml ../FLAMEGPU/templates/logging_and_functions.xslt logging_and_functions.c
pause
echo "This will overwrite the existing experiment_template.py file!"
pause
XSLTProcessor.exe XMLExperimentFile.xml ../FLAMEGPU/templates/experiment.xslt experiment_template.py
pause