@echo off

set nn_properties=props.yml
set io_arguments=io_args.yml


echo Training a translation model...
python train_conv.py %nn_properties% %io_arguments% || goto ERR

echo Building vocabularies...
python build_vocabularies.py %io_arguments% || goto ERR

echo Building dictionaries...
python build_dictionaries.py %io_arguments% || goto ERR


:ERR
echo.
echo ERROR: Process aborted!
exit /b 1
