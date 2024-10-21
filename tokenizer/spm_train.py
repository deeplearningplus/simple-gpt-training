#!/opt/software/install/miniconda37/bin/python
import sentencepiece as spm

#character_coverage: amount of characters covered by the model, 
#+good defaults are: 0.9995 for languages with rich character set 
#+like Japanese or Chinese and 1.0 for other languages with small character set.

spm.SentencePieceTrainer.Train(
    input='../data/text_reports.txt',
    model_prefix='tokenizer',
    pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    vocab_size=8888,
    model_type='bpe',
    #user_defined_symbols='`,~,!,@,#,$,%,^,&,*,(,),_,+,|,/,<,>,.,},{,],[,\\,?,\\n',
    character_coverage=1.0,
    shuffle_input_sentence=True,
    num_threads=8,
    #input_sentence_size=1000000-1,
    unk_surface=r" \342\201\207 ", # according to ../../llama2.c-master/tinystories.py
)
