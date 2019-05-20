echo off
echo "This will overwrite the existing functions.c file!"
pause
XSLTProcessor.exe XMLModelFile.xml ../FLAMEGPU/templates/functions.xslt functions.c
pause