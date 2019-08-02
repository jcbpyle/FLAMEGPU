echo off
echo "This will overwrite the existing experiment.py file!"
pause
XSLTProcessor.exe PredPreyGrass_Experiment.xml experiment.xslt experiment.py
pause