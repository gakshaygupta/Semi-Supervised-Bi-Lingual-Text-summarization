# Repurposed the code form Mikel Artetxe <artetxem@gmail.com> with appropiate modifications
from undreamt import devices
from undreamt.encoder import RNNEncoder
from undreamt.decoder import RNNAttentionDecoder
from undreamt.generator import *
from undreamt.translator import Translator
from undreamt.discriminator import *
import argparse
import numpy as np
import sys
import time
import os
import torch

def main_train():
    # Build argument parser
    parser = argparse.ArgumentParser(description='Train a neural machine translation model')

    # Training corpus
    corpora_group = parser.add_argument_group('training corpora', 'Corpora related arguments; specify either monolingual or parallel training corpora (or both)')
    corpora_group.add_argument('--src', help='the source language monolingual corpus')
    corpora_group.add_argument('--trg', help='the target language monolingual corpus')
    corpora_group.add_argument('--summary', metavar=('ARITICLE', 'SUMMARY'), nargs=2, help='the source-to-source-summary parallel corpora')  #edit
    corpora_group.add_argument('--src2trg', metavar=('SRC', 'TRG'), nargs=2, help='the source-to-target parallel corpus')
    corpora_group.add_argument('--trg2src', metavar=('TRG', 'SRC'), nargs=2, help='the target-to-source parallel corpus')
    corpora_group.add_argument('--max_sentence_length', type=int, default=50, help='the maximum sentence length for Translation training (defaults to 50)')
    corpora_group.add_argument('--max_sentence_source_long_length', type=int, default=200, help='the maximum sentence length for summarization training (defaults to 200)')
    corpora_group.add_argument('--cache', type=int, default=1000000, help='the cache size (in sentences) for corpus reading (defaults to 1000000)')
    corpora_group.add_argument('--cache_parallel', type=int, default=None, help='the cache size (in sentences) for parallel corpus reading')

    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings', 'Embedding related arguments; either give pre-trained cross-lingual embeddings, or a vocabulary and embedding dimensionality to randomly initialize them')
    embedding_group.add_argument('--src_embeddings', help='the source language word embeddings')
    embedding_group.add_argument('--trg_embeddings', help='the target language word embeddings')
    embedding_group.add_argument('--src_vocabulary', help='the source language vocabulary')
    embedding_group.add_argument('--trg_vocabulary', help='the target language vocabulary')
    embedding_group.add_argument('--embedding_size', type=int, default=0, help='the word embedding size')
    embedding_group.add_argument('--cutoff', type=int, default=0, help='cutoff vocabulary to the given size')
    embedding_group.add_argument('--learn_encoder_embeddings', action='store_true', help='learn the encoder embeddings instead of using the pre-trained ones')
    embedding_group.add_argument('--fixed_decoder_embeddings', action='store_true', help='use fixed embeddings in the decoder instead of learning them from scratch')
    embedding_group.add_argument('--fixed_generator', action='store_true', help='use fixed embeddings in the output softmax instead of learning it from scratch')

    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2, help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--discriminator_type', type=str, default="linear", help='the type of discriminator layer to use (defaults to linear layer)')#
    architecture_group.add_argument('--discriminator_hidden_size', type=int, default=100, help='the number of discriminator hidden size (defaults to 100)')#
    architecture_group.add_argument('--discriminator_filter_size', type=int, default=5, help='the number of discriminator filter size (defaults to 5)')#
    architecture_group.add_argument('--discriminator_filter_num', type=int, default=64, help='the number of discriminator filter number (defaults to 64)')#
    architecture_group.add_argument('--hidden', type=int, default=600, help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--disable_bidirectional', action='store_true', help='use a single direction encoder')
    architecture_group.add_argument('--disable_denoising', action='store_true', help='disable random swaps')
    architecture_group.add_argument('--denoise_prob',type=float, default=1, help='ammount of noisey sample (1 means all the samples are swapped and 0 means none are swapped) (default=1)')
    architecture_group.add_argument('--disable_backtranslation', action='store_true', help='disable backtranslation')

    # Training phase
    training_group = parser.add_argument_group('training','Training phase related argumnets')
    training_group.add_argument('--phase', type=int, default=3, help='the training phase to be executed. Initialization phase = 1 \n Summarization phase =2 \n Both phase = 3 (defaults to 3)')
    training_group.add_argument('--model_path', help='Path to the saved model folder if you want to continue from initialization phase')

    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch', type=int, default=50, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0002, help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--dropout', metavar='PROB', type=float, default=0.3, help='dropout probability for the encoder/decoder (defaults to 0.3)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1, help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--initialization_iterations', type=int, default=300000, help='the number of training iterations for initialization phase (defaults to 300000)')
    optimization_group.add_argument('--summarizationn_iterations', type=int, default=300000, help='the number of training iterations for summarization phase (defaults to 300000)')

    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--save', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=0, help='save intermediate models at this interval')

    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=1000, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--validation', nargs='+', default=(), help='use parallel corpora for translation validation ')
    logging_group.add_argument('--s_validation', nargs='+', default=(), help='use parallel corpora for source summarization validation ')
    logging_group.add_argument('--validation_directions', nargs='+', default=['src2src', 'trg2trg', 'src2trg', 'trg2src','el2es','el2hs'], help='validation directions')  # add el2s
    logging_group.add_argument('--validation_output', metavar='PREFIX', help='output validation translations with the given prefix')
    logging_group.add_argument('--validation_beam_size', type=int, default=0, help='use beam search for validation')
    # change el2s to el2es and add el2hs also we have to validate the translation of the english if parallel data is not available
    # conditions to add for arguments:- el2es,el2hs,s_validation,discriminator_hidden_size,discriminator_type
    # Other
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if args.src_embeddings is None and args.src_vocabulary is None or args.trg_embeddings is None and args.trg_vocabulary is None:
        print(args.src_embeddings,args.src_vocabulary,args.trg_embeddings,args.trg_vocabulary)
        print('Either an embedding or a vocabulary file must be provided')
        sys.exit(-1)
    if (args.src_embeddings is None or args.trg_embeddings is None) and (not args.learn_encoder_embeddings or args.fixed_decoder_embeddings or args.fixed_generator):
        print('Either provide pre-trained word embeddings or set to learn the encoder/decoder embeddings and generator')
        sys.exit(-1)
    if args.src_embeddings is None and args.trg_embeddings is None and args.embedding_size == 0:
        print('Either provide pre-trained word embeddings or the embedding size')
        sys.exit(-1)
    if len(args.validation) % 2 != 0:
        print('--validation should have an even number of arguments (one pair for each validation set)')
        sys.exit(-1)

    # Select device
    device = devices.gpu if args.cuda else devices.cpu

    # Create optimizer lists
    src2src_optimizers = []
    trg2trg_optimizers = []
    src2trg_optimizers = []
    trg2src_optimizers = []
    el2es_optimizers = []
    el2hs_optimizers = []                 #add el2s
    initialization_optimizers = []
    summarization_optimizers = []
    # Method to create a module optimizer and add it to the given lists
    def add_optimizer(module, module_name, directions=()):
        if args.param_init != 0.0:
            for param in module.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate)
        for direction in directions:
            direction.append([optimizer,module_name])
        return optimizer

    # Load word embeddings
    src_words = trg_words = src_embeddings = trg_embeddings = src_dictionary = trg_dictionary = None
    embedding_size = args.embedding_size
    if args.src_vocabulary is not None:
        f = open(args.src_vocabulary, encoding=args.encoding, errors='surrogateescape')
        src_words = [line.strip() for line in f.readlines()]
        if args.cutoff > 0:
            src_words = src_words[:args.cutoff]
        src_dictionary = data.Dictionary(src_words)
    if args.trg_vocabulary is not None:
        f = open(args.trg_vocabulary, encoding=args.encoding, errors='surrogateescape')
        trg_words = [line.strip() for line in f.readlines()]
        if args.cutoff > 0:
            trg_words = trg_words[:args.cutoff]
        trg_dictionary = data.Dictionary(trg_words)
    if args.src_embeddings is not None:
        f = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
        src_embeddings, src_dictionary = data.read_embeddings(f, args.cutoff, src_words)
        src_embeddings = device(src_embeddings)
        src_embeddings.requires_grad = False
        if embedding_size == 0:
            embedding_size = src_embeddings.weight.data.size()[1]
        if embedding_size != src_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)
    if args.trg_embeddings is not None:
        trg_file = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
        trg_embeddings, trg_dictionary = data.read_embeddings(trg_file, args.cutoff, trg_words)
        trg_embeddings = device(trg_embeddings)
        trg_embeddings.requires_grad = False
        if embedding_size == 0:
            embedding_size = trg_embeddings.weight.data.size()[1]
        if embedding_size != trg_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)
    if args.learn_encoder_embeddings:
        src_encoder_embeddings = device(data.random_embeddings(src_dictionary.size(), embedding_size))
        trg_encoder_embeddings = device(data.random_embeddings(trg_dictionary.size(), embedding_size))
        add_optimizer(src_encoder_embeddings,'src_encoder_embeddings', (src2src_optimizers, src2trg_optimizers,initialization_optimizers))
        add_optimizer(trg_encoder_embeddings, 'trg_encoder_embeddings', (trg2trg_optimizers, trg2src_optimizers,initialization_optimizers))
    else:
        src_encoder_embeddings = src_embeddings
        trg_encoder_embeddings = trg_embeddings
    if args.fixed_decoder_embeddings:
        src_decoder_embeddings = src_embeddings
        trg_decoder_embeddings = trg_embeddings
    else:
        src_decoder_embeddings = device(data.random_embeddings(src_dictionary.size(), embedding_size))
        trg_decoder_embeddings = device(data.random_embeddings(trg_dictionary.size(), embedding_size))
        add_optimizer(src_decoder_embeddings, (src2src_optimizers, 'src_decoder_embeddings', trg2src_optimizers, el2es_optimizers,initialization_optimizers))
        add_optimizer(trg_decoder_embeddings, (trg2trg_optimizers, 'trg_decoder_embeddings', src2trg_optimizers, el2hs_optimizers,initialization_optimizers))
    if args.fixed_generator:
        src_embedding_generator = device(EmbeddingGenerator(hidden_size=args.hidden, embedding_size=embedding_size))
        trg_embedding_generator = device(EmbeddingGenerator(hidden_size=args.hidden, embedding_size=embedding_size))
        add_optimizer(src_embedding_generator, 'src_embedding_generator', (src2src_optimizers, trg2src_optimizers, el2es_optimizers,initialization_optimizers))
        add_optimizer(trg_embedding_generator, 'trg_embedding_generator', (trg2trg_optimizers, src2trg_optimizers, el2hs_optimizers,initialization_optimizers))
        src_generator = device(WrappedEmbeddingGenerator(src_embedding_generator, src_embeddings))
        trg_generator = device(WrappedEmbeddingGenerator(trg_embedding_generator, trg_embeddings))
    else:
        src_generator = device(LinearGenerator(args.hidden, src_dictionary.size()))
        trg_generator = device(LinearGenerator(args.hidden, trg_dictionary.size()))
        add_optimizer(src_generator,'src2src_optimizers', (src2src_optimizers, trg2src_optimizers,el2es_optimizers,initialization_optimizers))
        add_optimizer(trg_generator, 'trg2trg_optimizers', (trg2trg_optimizers, src2trg_optimizers,el2hs_optimizers,initialization_optimizers))

    # Build encoder
    encoder = device(RNNEncoder(embedding_size=embedding_size, hidden_size=args.hidden,
                                bidirectional=not args.disable_bidirectional, layers=args.layers, dropout=args.dropout))
    add_optimizer(encoder, 'encoder', (src2src_optimizers, trg2trg_optimizers, src2trg_optimizers, trg2src_optimizers,initialization_optimizers))
    # Build discriminator
    if args.discriminator_type=="linear":
        discriminator=device(LinearDiscriminator(device = device, hidden_size=args.hidden,linear_size=args.discriminator_hidden_size))
    else:
        discriminator=device(ConvDiscriminator(device = device,hidden_size = args.hidden, filter_size = args.discriminator_filter_size, num_filters = args.discriminator_filter_num))
    add_optimizer(discriminator, 'discriminator', (src2src_optimizers, trg2trg_optimizers, src2trg_optimizers, trg2src_optimizers,initialization_optimizers))
    # Build decoders
    src_decoder = device(RNNAttentionDecoder(embedding_size=embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
    trg_decoder = device(RNNAttentionDecoder(embedding_size=embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
    add_optimizer(src_decoder, 'src_decoder', (src2src_optimizers, trg2src_optimizers,el2es_optimizers,initialization_optimizers))#
    add_optimizer(trg_decoder, 'trg_decoder', (trg2trg_optimizers, src2trg_optimizers,el2hs_optimizers,initialization_optimizers))#

    if args.phase == 2 and args.model_path != None:
        dir=os.listdir(args.model_path)
        models = ['encoder','discriminator','decoder','src_encoder_embeddings','src_decoder_embeddings','src_generator','trg_encoder_embeddings','trg_decoder_embeddings','trg_generator']
        path = {}
        for model in models:
            index = [dir[x] for x in range(0,len(dir)) if model+".pth" in dir[x] ]
            path[model] = index[0]

        encoder=torch.load(os.path.join(args.model_path,path("encoder")))
        discriminator=torch.load(os.path.join(args.model_path,path("discriminator")))
        decoder=torch.load(os.path.join(args.model_path,path("decoder")))
        src_encoder_embeddings=torch.load(os.path.join(args.model_path,path("src_encoder_embeddings")))
        src_decoder_embeddings=torch.load(os.path.join(args.model_path,path("src_decoder_embeddings")))
        src_generator=torch.load(os.path.join(args.model_path,path("src_generator")))
        trg_encoder_embeddings=torch.load(os.path.join(args.model_path,path("trg_encoder_embeddings")))
        trg_decoder_embeddings=torch.load(os.path.join(args.model_path,path("trg_decoder_embeddings")))
        trg_generator=torch.load(os.path.join(args.model_path,path("trg_generator")))


    # Build translators src_encoder_embeddings,src_decoder_embeddings,src_generator,trg_encoder_embeddings,trg_decoder_embeddings,trg_generator
    src2src_translator = Translator(encoder_embeddings=src_encoder_embeddings,
                                    decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                    src_dictionary=src_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                    decoder=src_decoder, denoising=not args.disable_denoising, device=device,discriminator=discriminator,source=1,denoise_prob=args.denoise_prob)#
    src2trg_translator = Translator(encoder_embeddings=src_encoder_embeddings,
                                    decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                    src_dictionary=src_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                    decoder=trg_decoder, denoising=not args.disable_denoising, device=device,discriminator=discriminator,source=1,denoise_prob=args.denoise_prob)#
    trg2trg_translator = Translator(encoder_embeddings=trg_encoder_embeddings,
                                    decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                    src_dictionary=trg_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                    decoder=trg_decoder, denoising=not args.disable_denoising, device=device,discriminator=discriminator,source=0,denoise_prob=args.denoise_prob)#
    trg2src_translator = Translator(encoder_embeddings=trg_encoder_embeddings,
                                    decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                    src_dictionary=trg_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                    decoder=src_decoder, denoising=not args.disable_denoising, device=device,discriminator=discriminator,source=0,denoise_prob=args.denoise_prob)#
    el2es_summarizer = Translator(encoder_embeddings=src_encoder_embeddings,
                                    decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                    src_dictionary=src_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                    decoder=src_decoder, denoising=False, device=device,rec=False,discriminator=None,source=1)#
    el2hs_summarizer = Translator(encoder_embeddings=src_encoder_embeddings,
                                    decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                    src_dictionary=src_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                    decoder=trg_decoder, denoising=False, device=device,rec=False,discriminator=None,source=1)#

    # Build trainers
    trainers_initialization = []
    trainers_summarization = []
    src2src_trainer = trg2trg_trainer = src2trg_trainer = trg2src_trainer = None #might have to be changed
    srcback2trg_trainer = trgback2src_trainer = None
    el2hs_trainer = el2es_trainer=None
    if args.src is not None:
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f, max_src_sentence_length=args.max_sentence_length,max_trg_sentence_length=args.max_sentence_length, cache_size=args.cache)
        src2src_trainer = Trainer(translator=src2src_translator, optimizers=src2src_optimizers, corpus=corpus, batch_size=args.batch)
        trainers_initialization.append(src2src_trainer)
        if not args.disable_backtranslation:
            trgback2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus, translator=src2trg_translator), batch_size=args.batch)
            trainers_initialization.append(trgback2src_trainer)
    if args.trg is not None:
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f, max_src_sentence_length=args.max_sentence_length, max_trg_sentence_length=args.max_sentence_length, cache_size=args.cache)
        trg2trg_trainer = Trainer(translator=trg2trg_translator, optimizers=trg2trg_optimizers, corpus=corpus, batch_size=args.batch)
        trainers_initialization.append(trg2trg_trainer)
        if not args.disable_backtranslation:
            srcback2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus, translator=trg2src_translator), batch_size=args.batch)
            trainers_initialization.append(srcback2trg_trainer)
    if args.summary is not None:
        #el2es supervised # make encoder parameters static
        f1 = open(args.summary[0], encoding=args.encoding, errors='surrogateescape')
        f2 = open(args.summary[1], encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f1 ,f2, max_src_sentence_length=args.max_sentence_source_long_length, max_trg_sentence_length=args.max_sentence_length, cache_size=args.cache)
        el2es_trainer = Trainer(translator=el2es_summarizer, optimizers=el2es_optimizers, corpus=corpus, batch_size=args.batch,summary = True)
        trainers_summarization.append(el2es_trainer)
        #el2hs unsupervised
        corpus = data.CorpusReader(f1, f2, max_src_sentence_length=args.args.max_sentence_source_long_length, max_trg_sentence_length=args.max_sentence_length, cache_size=args.cache)
        if not args.disable_backtranslation:
            el2hs_trainer = Trainer(translator=el2hs_summarizer, optimizers=el2hs_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus, translator=src2trg_translator,summary = True), batch_size=args.batch,summary=True)
            trainers_summarization.append(el2hs)        ## check again
    if args.src2trg is not None:
        f1 = open(args.src2trg[0], encoding=args.encoding, errors='surrogateescape')
        f2 = open(args.src2trg[1], encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f1, f2, max_src_sentence_length=args.max_sentence_length, max_trg_sentence_length=args.max_sentence_length, cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
        src2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers, corpus=corpus, batch_size=args.batch)
        trainers_summarization.append(src2trg_trainer)
    if args.trg2src is not None:
        f1 = open(args.trg2src[0], encoding=args.encoding, errors='surrogateescape')
        f2 = open(args.trg2src[1], encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f1, f2, max_src_sentence_length=args.max_sentence_length, max_trg_sentence_length=args.max_sentence_length, cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
        trg2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers, corpus=corpus, batch_size=args.batch)
        trainers_summarization.append(trg2src_trainer)

    # Build validators
    src2src_validators = []
    trg2trg_validators = []
    src2trg_validators = []
    trg2src_validators = []
    el2es_validators = []#
    el2hs_validators = []#
    for i in range(0, len(args.validation), 2):
        src_validation = open(args.validation[i],   encoding=args.encoding, errors='surrogateescape').readlines()
        trg_validation = open(args.validation[i+1], encoding=args.encoding, errors='surrogateescape').readlines()
        if len(src_validation) != len(trg_validation):
            print('Validation sizes do not match')
            sys.exit(-1)
        map(lambda x: x.strip(), src_validation)
        map(lambda x: x.strip(), trg_validation)
        if 'src2src' in args.validation_directions:
            src2src_validators.append(Validator(src2src_translator, src_validation, src_validation, args.batch, args.validation_beam_size))
        if 'trg2trg' in args.validation_directions:
            trg2trg_validators.append(Validator(trg2trg_translator, trg_validation, trg_validation, args.batch, args.validation_beam_size))
        if 'src2trg' in args.validation_directions:
            src2trg_validators.append(Validator(src2trg_translator, src_validation, trg_validation, args.batch, args.validation_beam_size))
        if 'trg2src' in args.validation_directions:
            trg2src_validators.append(Validator(trg2src_translator, trg_validation, src_validation, args.batch, args.validation_beam_size))

    for i in range(0, len(args.s_validation), 2):#
        src_validation = open(args.validation[i],   encoding=args.encoding, errors='surrogateescape').readlines()
        trg_validation = open(args.validation[i+1], encoding=args.encoding, errors='surrogateescape').readlines()
        if len(src_validation) != len(trg_validation):
            print('Validation sizes do not match')
            sys.exit(-1)
        map(lambda x: x.strip(), src_validation)
        map(lambda x: x.strip(), trg_validation)
        if 'el2es' in args.validation_directions:
            el2es_validators.append(Validator(el2es_summarizer, scr_validation, trg_validation, args.batch, args.validation_beam_size))#
        """if 'el2hs' in args.validation_directions:
            el2hs_validators.append(Validator(trg2trg_translator, trg_validation, trg_validation, args.batch, args.validation_beam_size))"""#
        #create a validator from summary to hindi and then hindi to english then check the error
        # Build loggers
    loggers_initialization = []
    loggers_summarization = []
    src2src_output = trg2trg_output = src2trg_output = trg2src_output= el2es_output= el2hs_output = None#
    if args.validation_output is not None:
        src2src_output = '{0}.src2src'.format(args.validation_output)
        trg2trg_output = '{0}.trg2trg'.format(args.validation_output)
        src2trg_output = '{0}.src2trg'.format(args.validation_output)
        trg2src_output = '{0}.trg2src'.format(args.validation_output)
        el2es_output = '{0}.el2es'.format(args.validation_output)#
        el2hs_output = '{0}.el2hs'.format(args.validation_output)#
        #gotta add for english to english summarization also

    loggers_initialization.append(Logger('Source to target (backtranslation)', srcback2trg_trainer, args.log_interval, [], None, args.encoding))
    loggers_initialization.append(Logger('Target to source (backtranslation)', trgback2src_trainer, args.log_interval, [], None, args.encoding))
    loggers_initialization.append(Logger('Source to source', src2src_trainer, args.log_interval, src2src_validators, src2src_output, args.encoding))
    loggers_initialization.append(Logger('Target to target', trg2trg_trainer, args.log_interval, trg2trg_validators, trg2trg_output, args.encoding))
    loggers_initialization.append(Logger('Source to target', src2trg_trainer, args.log_interval, src2trg_validators, src2trg_output, args.encoding))
    loggers_initialization.append(Logger('Target to source', trg2src_trainer, args.log_interval, trg2src_validators, trg2src_output, args.encoding))
    loggers_summarization.append(Logger('source long to source summary', el2es_trainer, args.log_interval, el2es_validators, el2es_output, args.encoding))#
    loggers_summarization.append(Logger('source long to traget summary', el2hs_trainer, args.log_interval, el2hs_validators, el2hs_output, args.encoding))#
    loggers_summarization.append(Logger('Source to target', src2trg_trainer, args.log_interval, src2trg_validators, src2trg_output, args.encoding))
    loggers_summarization.append(Logger('Target to source', trg2src_trainer, args.log_interval, trg2src_validators, trg2src_output, args.encoding))

    # Method to save models
    def save_models(name ,initialization = True):
        if initialization:
            torch.save(src2src_translator, '{0}.{1}.src2src.pth'.format(args.save, name))
            torch.save(trg2trg_translator, '{0}.{1}.trg2trg.pth'.format(args.save, name))
            torch.save(encoder,'{0}.{1}.encoder.pth'.format(args.save, name))
            torch.save(discriminator,'{0}.{1}.discriminator.pth'.format(args.save, name))
            torch.save(decoder,'{0}.{1}.decoder.pth'.format(args.save, name))
            torch.save(src_encoder_embeddings,'{0}.{1}.src_encoder_embeddings.pth'.format(args.save, name))
            torch.save(src_decoder_embeddings,'{0}.{1}.src_decoder_embeddings.pth'.format(args.save, name))
            torch.save(src_generator,'{0}.{1}.src_generator.pth'.format(args.save, name))
            torch.save(trg_encoder_embeddings,'{0}.{1}.trg_encoder_embeddings.pth'.format(args.save, name))
            torch.save(trg_decoder_embeddings,'{0}.{1}.trg_decoder_embeddings.pth'.format(args.save, name))
            torch.save(trg_generator,'{0}.{1}.trg_generator.pth'.format(args.save, name))

        else:
            torch.save(src2trg_translator, '{0}.{1}.src2trg.pth'.format(args.save, name))
            torch.save(trg2src_translator, '{0}.{1}.trg2src.pth'.format(args.save, name))
            torch.save(el2es_summarizer, '{0}.{1}.el2es.pth'.format(args.save, name))#
            torch.save(el2hs_summarizer, '{0}.{1}.el2hs.pth'.format(args.save, name))#
            torch.save(encoder,'{0}.{1}.s_encoder.pth'.format(args.save, name))
            torch.save(discriminator,'{0}.{1}.s_discriminator.pth'.format(args.save, name))
            torch.save(decoder,'{0}.{1}.s_decoder.pth'.format(args.save, name))
            torch.save(src_encoder_embeddings,'{0}.{1}.s_src_encoder_embeddings.pth'.format(args.save, name))
            torch.save(src_decoder_embeddings,'{0}.{1}.s_src_decoder_embeddings.pth'.format(args.save, name))
            torch.save(src_generator,'{0}.{1}.s_src_generator.pth'.format(args.save, name))
            torch.save(trg_encoder_embeddings,'{0}.{1}.s_trg_encoder_embeddings.pth'.format(args.save, name))
            torch.save(trg_decoder_embeddings,'{0}.{1}.s_trg_decoder_embeddings.pth'.format(args.save, name))
            torch.save(trg_generator,'{0}.{1}.s_trg_generator.pth'.format(args.save, name))


    # Training  We can implement the initialization and summarization phase
    #initialization phase
    def initialization_training(l):
            loggers = l
            for step in range(1, args.initialization_iterations + 1):
                lossEG = lossD = lossG = 0
                for trainer in trainers_initialization:#
                    loss = trainer.step()
                    lossEG += loss[0]
                    lossD += loss[1]
                    lossG += loss[2]
                t = time.time()
                optimizers = initialization_optimizers
                for optimizer in optimizers:
                    optimizer[0].zero_grad()
                #lossEG, lossD, lossG = loss
                lossT= lossEG + lossG
                lossT.div(args.batch).backward(retain_graph = True)
                """lossEG.div(args.batch).backward(retain_graph = True)
                lossG.div(args.batch).backward(retain_graph = True)"""
                for optimizer in optimizers:
                    if optimizer[1] != "discriminator":
                        optimizer[0].step()
                for optimizer in optimizers:
                    if optimizer[1] == "discriminator":
                        optimizer[0].zero_grad()
                #lossD = lossD
                lossD.div(args.batch).backward()
                for optimizer in optimizers:           # can be optimised much more:- a dictionary can be used
                    if optimizer[1] == "discriminator":
                        optimizer[0].step()
                backward_time = time.time() - t
                for trainer in trainers_initialization:#
                     trainer.backward_time += backward_time/len(trainers_initialization)
                if args.save is not None and args.save_interval > 0 and step % args.save_interval == 0:
                    save_models('it{0}'.format(step))

                if step % args.log_interval == 0:
                    print()
                    print('STEP {0} x {1}'.format(step, args.batch))
                    for logger in loggers:
                        logger.log(step)

                step += 1

            save_models('final_UNMT')#
        #summarization phase
    def summarization_training(l):
        loggers = l
        for step in range(1, args.summarizationn_iterations + 1):#
            lossEG = lossD = lossG = 0
            loss_sum = 0
            for trainer in trainers_summarization:
                loss = trainer.step()
                if trainer.summary:
                    loss_sum += loss
                else:
                    lossEG += loss[0]
                    lossD += loss[1]
                    lossG += loss[2]
            t = time.time()
            optimizers = summarization_optimizers
            for optimizer in optimizers:
                optimizer[0].zero_grad()

            #lossEG, lossD, lossG = loss
            lossT= lossEG + lossG
            lossT.div(args.batch).backward(retain_graph = True)
            """lossEG.div(args.batch).backward(retain_graph = True)
            lossG.div(args.batch).backward(retain_graph = True)"""
            for optimizer in optimizers:
                if optimizer[1] != "discriminator":
                    optimizer[0].step()
            for optimizer in optimizers:
                if optimizer[1] == "discriminator":
                    optimizer[0].zero_grad()
            #lossD = lossD
            lossD.div(args.batch).backward()
            for optimizer in optimizers:           # can be optimised much more:- a dictionary can be used
                if optimizer[1] == "discriminator":
                    optimizer[0].step()
            backward_time = time.time() - t
            if args.save is not None and args.save_interval > 0 and step % args.save_interval == 0:
                save_models('it{0}'.format(step) ,initialization = False)

            if step % args.log_interval == 0:
                print()
                print('STEP {0} x {1}'.format(step, args.batch))
                for logger in loggers:
                    logger.log(step)

            step += 1

        save_models('final_NMT_SUMMARIZATION' ,initialization=False)#
    if args.phase == 1:#
        initialization_training(loggers_initialization)
    if args.phase == 2:
        summarization_training(loggers_summarization)
    if args.phase == 3:
        initialization_training(loggers_initialization)
        summarization_training(loggers_summarization)  #

class Trainer:
    def __init__(self, corpus, optimizers, translator, batch_size=50, summary = False):
        self.corpus = corpus
        self.translator = translator
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.summary = summary
        self.reset_stats()

    def step(self):
        # Reset gradients
        """for optimizer in self.optimizers:
            optimizer[0].zero_grad()"""

        # Read input sentences
        t = time.time()
        src, trg = self.corpus.next_batch(self.batch_size)
        self.src_word_count += sum([len(data.tokenize(sentence)) + 1 for sentence in src])  # TODO Depends on special symbols EOS/SOS
        self.trg_word_count += sum([len(data.tokenize(sentence)) + 1 for sentence in trg])  # TODO Depends on special symbols EOS/SOS
        self.io_time += time.time() - t

        # Compute loss
        t = time.time()

        loss = self.translator.score(src, trg, train=True)
        if not self.summary:
            lossEG, lossD, lossG = loss
            self.lossEG += lossEG.item()
            self.lossD  += lossD.item()
            self.lossG  += lossG.item()                 #generalize: should depend on summary
        else:
            self.lossEG += loss.item()
        self.forward_time += time.time() - t

        # Backpropagate error + optimize            #TODO Edit this for discriminator
        """t = time.time()
        if not self.summary:
            lossEG, lossD, lossG = loss
            lossT= lossEG + lossG
            lossT.div(self.batch_size).backward(retain_graph = True)
            lossEG.div(self.batch_size).backward(retain_graph = True)
            lossG.div(self.batch_size).backward(retain_graph = True)
            for optimizer in self.optimizers:
                if optimizer[1] != "discriminator":
                    optimizer[0].step()
            for optimizer in self.optimizers:
                if optimizer[1] == "discriminator":
                    optimizer[0].zero_grad()
            lossD = lossD
            lossD.div(self.batch_size).backward()
            for optimizer in self.optimizers:           # can be optimised much more:- a dictionary can be used
                if optimizer[1] == "discriminator":
                    optimizer[0].step()

        else:
            loss.div(self.batch_size).backward()
            for optimizer in self.optimizers:
                optimizer[0].step()
        self.backward_time += time.time() - t"""
        return loss

    def reset_stats(self):
        self.src_word_count = 0
        self.trg_word_count = 0
        self.io_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.lossEG = 0
        self.lossD = 0
        self.lossG = 0

    def perplexity_per_word(self):
        if self.summary:
            return np.exp(self.lossEG/self.trg_word_count)
        return np.exp(self.lossEG/self.trg_word_count), self.lossD/self.batch_size, self.lossG/self.batch_size

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / self.total_time(),  self.trg_word_count / self.total_time()


class Validator:
    def __init__(self, translator, source, reference, batch_size=50, beam_size=0):
        self.translator = translator
        self.source = source
        self.reference = reference
        self.sentence_count = len(source)
        self.reference_word_count = sum([len(data.tokenize(sentence)) + 1 for sentence in self.reference])  # TODO Depends on special symbols EOS/SOS
        self.batch_size = batch_size
        self.beam_size = beam_size

        # Sorting
        lengths = [len(data.tokenize(sentence)) for sentence in self.source]
        self.true2sorted = sorted(range(self.sentence_count), key=lambda x: -lengths[x])
        self.sorted2true = sorted(range(self.sentence_count), key=lambda x: self.true2sorted[x])
        self.sorted_source = [self.source[i] for i in self.true2sorted]
        self.sorted_reference = [self.reference[i] for i in self.true2sorted]

    def perplexity(self):
        lossEG = 0

        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            EG = self.translator.score(self.sorted_source[i:j], self.sorted_reference[i:j], train=False)
            lossEG += EG.item()

        return np.exp(lossEG/self.reference_word_count)

    def translate(self):
        translations = []
        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            batch = self.sorted_source[i:j]
            if self.beam_size <= 0:
                translations += self.translator.greedy(batch, train=False)
            else:
                translations += self.translator.beam_search(batch, train=False, beam_size=self.beam_size)
        return [translations[i] for i in self.sorted2true]


class Logger:
    def __init__(self, name, trainer, log_interval, validators=(), output_prefix=None, encoding='utf-8'):
        self.name = name
        self.trainer = trainer
        self.validators = validators
        self.output_prefix = output_prefix
        self.encoding = encoding
        self.log_interval = log_interval
    def log(self, step=0):
        if self.trainer is not None or len(self.validators) > 0:
            print('{0}'.format(self.name))
        if self.trainer is not None:
            if not self.trainer.summary :
                lossEG ,lossD ,lossG = self.trainer.perplexity_per_word()
                print('  - Training:   {0:10.2f}, -Discriminator:  {1:10.2f}, -Generator  {2:10.2f}  ({3:.2f}s: {4:.2f}tok/s src, {5:.2f}tok/s trg; epoch {6}),'
                    .format(lossEG ,lossD/self.log_interval ,lossG/self.log_interval, self.trainer.total_time(),
                    self.trainer.words_per_second()[0], self.trainer.words_per_second()[1], self.trainer.corpus.epoch))  # have to log generator and discriminator loss also
                self.trainer.reset_stats()
            else:
                lossEG = self.trainer.perplexity_per_word()
                print('  - Training:   {0:10.2f}  ({3:.2f}s: {4:.2f}tok/s src, {5:.2f}tok/s trg; epoch {6}),'
                    .format(lossEG ,self.trainer.total_time(),
                    self.trainer.words_per_second()[0], self.trainer.words_per_second()[1], self.trainer.corpus.epoch))  # have to log generator and discriminator loss also
                self.trainer.reset_stats()
        for id, validator in enumerate(self.validators):
            t = time.time()
            perplexity = validator.perplexity()
            print('  - Validation: {0:10.2f}   ({1:.2f}s)'.format(perplexity, time.time() - t))
            if self.output_prefix is not None:
                f = open('{0}.{1}.{2}.txt'.format(self.output_prefix, id, step), mode='w',
                         encoding=self.encoding, errors='surrogateescape')
                for line in validator.translate():
                    print(line, file=f)
                f.close()
        sys.stdout.flush()
