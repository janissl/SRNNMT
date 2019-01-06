@echo off

set nn_properties=props.yml
set io_arguments=io_args.yml

echo 1. Vectorizing sentences...
python vectorize_dense.py %nn_properties% %io_arguments% || goto ERR

echo 2. Getting cosine similarity...
python dot.py %nn_properties% %io_arguments% || goto ERR

echo 3. Exporting aligned sentences...
python export_aligned_sents.py %io_arguments% || goto ERR

echo 4. Finding aligned sentence indices in the original segmented files...
python get_segment_alignments.py %io_arguments% || goto ERR

echo 5. Building parallel corpora...
python build_parallel_corpora.py %io_arguments% || goto ERR

echo 6. Extract unique segment pairs...
python extract_unique_pairs.py %io_arguments% || goto ERR

echo Done.
exit /b

:ERR
echo.
echo ERROR: Process aborted!
exit /b 1
