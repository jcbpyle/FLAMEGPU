echo off
echo "This will overwrite the existing experiment.py file!"
pause
D:/Documents/sim_experiment_api_testing/FLAMEGPU/tools/XSLTProcessor.exe Paper_experiment_xmlfile.xml D:/Documents/sim_experiment_api_testing/FLAMEGPU/FLAMEGPU/templates/experiment.xslt reproduction_of_experiment.py
pause