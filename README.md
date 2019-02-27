# SRNNMT
A set of scripts to build parallel corpora using a neural network.<br>
Based on [TurkuNLP/SRNNMT](https://github.com/TurkuNLP/SRNNMT) as well as [csr_csc_dot](https://github.com/fginter/csr_csc_dot).
<hr>

## Usage
##### File system structure:
<pre><code>
${corpus_title}
|-- source
|   |-- ${title}_${source_lang}.snt
|   |-- ${title}_${target_lang}.snt
|-- work
|   |-- ${source_lang}-${target_lang}
|       |-- ${title}_${source_lang}.snt
|       |-- ${title}_${target_lang}.snt
|       |-- ${title}_${source_lang}.snt.aligned
|       |-- ${title}_${target_lang}.snt.aligned
|-- aligned_idx
|   |-- ${source_lang}-${target_lang}
|       |-- ${title}.${source_lang}.idx
|       |-- ${title}.${target_lang}.idx
|-- result
    |-- ${corpus_title}.${source_lang}-${target_lang}.${source_lang}
    |-- ${corpus_title}.${source_lang}-${target_lang}.${target_lang}
    |-- ${corpus_title}.unique.${source_lang}-${target_lang}.${source_lang}
    |-- ${corpus_title}.unique.${source_lang}-${target_lang}.${target_lang}
</code></pre>

* Install the following dependencied using the `python -m pip install ${module_name}` command if necessary:
  * PyYAML
  * Cython
  * numpy
  * scikit-learn
  * tensorflow
  * Keras
* Before running the shell script, put your source files in _${corpus\_title}/source_ directory.
* The content of source files must be segmented in sentences (one sentence per line).
* Filenames of input files must have the following pattern: _${title}\_${lang}.snt_ (e.g. _document\_en.snt_).
* Parallel files must have identical titles (e.g. _article\_001\_en.snt_, _article\_001\_fr.snt_).
* There are two source data directories - _'original_source_data_directory'_ and _'source_data_directory'_ - specified in the YAML file.
The 'original_source_data_directory' is used for files containing sentences in natural language (i.e. unmodified sentences).
The _'source_data_directory'_ is used for additionaly preprocessed files originated from the 'original_source_data_directory'
(e.g. stemmed files, additionally tokenized files etc.).
The sentence alignment itself is done using the content from the _'source_data_directory'_.
On the contrary, the building of parallel corpora is done using the content from _'original_source_data_directory'_.
If no additional preprocessing has been made on source files, both paths must be equal.
* The _'work'_, _'aligned_idx'_ and _'result'_ directories are created automatically.
* Aligned corpora are placed in the _'result'_ directory.
<br>

__Note:__ It is not necessary to keep all automatically created subdirectories (_work_, _aligned_idx_, _result_) under
the same root but it is much easier to track the alignment process in this way.

##### An example of a neural network model properties file (YAML):
<pre><code>
ngram_length: [4, ]
feature_count: 150
gru_width: 150
max_sent_length: 200
minibatch_size: 200
epochs: 1000
</code></pre>

##### An example of a configuration file (YAML):
(for running on Windows OS; replace values in square brackets with actual paths; see also _io\_args.yml.sample_)
<pre><code>
source_language: en
target_language: fr

corpus_title: aligned_corpora

model_name: train_corpus.model
epoch_number: 100

src_train: [...]\train\train_corpus.unique.en-fr.en
trg_train: [...]\train\train_corpus.unique.en-fr.fr

src_devel: [...]\devel\devel_corpus.unique.en-fr.en
trg_devel: [...]\devel\devel_corpus.unique.en-fr.fr

dictionary_directory: [...]\dict
dictionary_name: en-fr.lex

vocabulary_directory: [...]\voc
vocabulary_name: train_corpus.unique.tokens.en-fr

original_source_data_directory: [...]\aligned_corpora\source
source_data_directory: [...]\aligned_corpora\source
work_directory: [...]\aligned_corpora\work
alignment_index_directory: [...]\aligned_corpora\aligned_idx
output_data_directory: [...]\aligned_corpora\result

alignment_threshold: 0.92
</code></pre>

_Notes:_
* The _dictionary\_name_ value must not contain a filename extension i.e. .e2f or .f2e.
The same also applies to the _vocabulary\_name_.
* The epoch_number matches the epoch when the best model was saved.

### Training the model
* Enter the desired values for parameters in the neural network properties file and the alignment configuration file (see above).
* Specify the names of the neural network properties and alignment configuration files in the _run\_training.bat_ file
(the values of _nn_properties_ and _io_arguments_). The YAML files must reside in the script directory.
* Execute the following command (on Windows):<br>
`.\run_training.bat`

### Running the alignment
* Check the actual values for parameters in the alignment configuration file (see above).
* Specify the names of the neural network properties and alignment configuration files in the _run\_alignment.bat_ file
(the values of _nn_properties_ and _io_arguments_). The YAML files must reside in the script directory.
* Execute the following command (on Windows):<br>
`.\run_alignment.bat`
<br><br>

__Note:__ The current set of scripts may be also run under UNIX/Linux OS.
For this purpose, Bash scripts similar to _run\_training.bat_ and _run\_alignment.bat_ must be executed.
<br><br>

TODO: continue development of _build\_dictionaries.py_<br>
At the moment, this script is just a placeholder. Dictionaries can be built using [bsa-wrapper](https://github.com/janissl/bsa-wrapper):
execute the step 1 in the _run\_bsa.bat_ script for both language directions i.e. en-fr and fr-en for instance,
find the _model-one_ files in the _[...]\work\sent\_align\en-fr_ and _[...]\work\sent\_align\fr-en_ directories, reorder
columns into the following order: source_word, target_word, probability, and save the output files as _${dictionary\_name}.e2f_
(for the fr-en language direction) and _${dictionary\_name}.f2e_ (for the en-fr language direction).
<hr>

##### References:
[Finding Translation Pairs from Unordered Internet Text](https://blogs.helsinki.fi/language-technology/files/2017/09/FINMT2017_Kanerva.pdf)
