echo off
echo "This will overwrite the existing experiment_template.py file!"
pause
XSLTProcessor.exe XMLExperimentFile.xml ../FLAMEGPU/templates/experiment.xslt experiment_template.py
pause