echo off
echo "This will overwrite the existing logging_and_functions.c file!"
pause
XSLTProcessor.exe XMLModelFile.xml ../FLAMEGPU/templates/logging.xslt logging.h
pause