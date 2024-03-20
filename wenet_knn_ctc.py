from __future__ import print_function

from datasets import load_dataset
import torch
from jiwer import cer
import librosa
import torchaudio

from dataclasses import dataclass, field
from transformers import HfArgumentParser
import sys
import os
from wenet_knn_ctc_model import KNNSaver_for_ctc, KNNWrapper_for_ctc, KEY_TYPE, DIST
import time

from datetime import datetime

import argparse
import copy
import logging
import os
import sys

import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.paraformer.search.beam_search import build_beam_search
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model

current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")

logger = logging.getLogger(__name__)
logger.setLevel(20)

file_handler = logging.FileHandler("./log/"+current_time+'.log', mode='w')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

@dataclass
class KNNArguments:
    """
    external knn arguments
    """
    knn: bool = field(default=False)
    knn_gpu: bool = field(default=True)
    dstore_size: int = field(default=None, metadata={"help": "The size of the dstore."})
    knn_keytype: KEY_TYPE.from_string = field(default=KEY_TYPE.last_ffn_input)
    save_knnlm_dstore: bool = field(default=False)
    dstore_dir: str = field(default="datastore/aishell_dstore")
    knn_sim_func: DIST.from_string = field(default=DIST.l2)
    lmbda: float = field(default=0.25)
    k: int = field(default=1024)
    knn_temp: float = field(default=1.0)
    # Args for building the faiss index:
    build_index: bool = field(default=False)
    # faiss_index: str = field(default="checkpoints/index")
    ncentroids: int = field(default=4096)
    code_size: int = field(default=64)
    probe: int = field(default=32)
    num_keys_to_add_at_a_time: int = field(default=1000000)
    move_dstore_to_mem: bool = field(default=True)
    no_load_keys: bool = field(default=True)
    recompute_dists: bool = field(default=False)
    thr: float = field(default=0.0)  # pseudo label thr
    decode_skip_blank : bool = field(default = False)
    scale_lmbda : bool = field(default=False)
    scale_lmbda_temp: float=field(default=1.0)
    use_null_mask: bool=field(default=True)
    
@dataclass
class TestArguments:
    """
    normal recogize.py arguments
    """
    config: str = field(default=None, metadata={"help": "config file"})
    test_data: str = field(default=None, metadata={"help": "test data file"})
    data_type: str = field(default="raw", metadata={"help": "train and cv data type"})
    gpu: int = field(default=-1, metadata={"help": "gpu id for this rank, -1 for cpu"})
    checkpoint: str = field(default=None, metadata={"help": "checkpoint model"})
    dict: str = field(default=None, metadata={"help": "dict file"})
    non_lang_syms: str = field(default=None, metadata={"help": "non-linguistic symbol file. One symbol per line."})
    beam_size: int = field(default=10, metadata={"help": "beam size for search"})
    penalty: float = field(default=0.0, metadata={"help": "length penalty"})
    result_file: str = field(default=None, metadata={"help": "asr result file"})
    batch_size: int = field(default=16, metadata={"help": "batch size"})
    mode: str = field(default="knn_ctc", metadata={"help": "decoding mode"})
    search_ctc_weight: float = field(default=1.0, metadata={"help": "ctc weight for nbest generation"})
    search_transducer_weight: float = field(default=0.0, metadata={"help": "transducer weight for nbest generation", })
    ctc_weight: float = field(default=0.0, metadata={"help": "ctc weight for rescoring weight in attention rescoring decode mode"})
    transducer_weight: float = field(default=0.0, metadata={"help": "transducer weight for rescoring weight in transducer attention rescore decode mode"})
    attn_weight: float = field(default=0.0, metadata={"help": "attention weight for rescoring weight in transducer attention rescore decode mode"})
    decoding_chunk_size: int = field(default=-1, metadata={"help": "decoding chunk size"})
    num_decoding_left_chunks: int = field(default=-1, metadata={"help": "number of left chunks for decoding"})
    simulate_streaming: bool = field(default=False, metadata={"help": "simulate streaming inference"})
    reverse_weight: float = field(default=0.0, metadata={"help": "right to left weight for attention rescoring decode mode"})
    bpe_model: str = field(default=None, metadata={"help": "bpe model for english part"})
    connect_symbol: str = field(default='', metadata={"help": "used to connect the output characters"})
    word: str = field(default='', metadata={"help": "word file, only used for hlg decode"})
    hlg: str = field(default='', metadata={"help": "hlg file, only used for hlg decode"})
    lm_scale: float = field(default=0.0, metadata={"help": "lm scale for hlg attention rescore decode"})
    decoder_scale: float = field(default=0.0, metadata={"help": "lm scale for hlg attention rescore decode"})
    r_decoder_scale: float = field(default=0.0, metadata={"help": "lm scale for hlg attention rescore decode"})
    cmvn: str = field(default=None , metadata={"help":"cmvn opt, location of cmvn file."})

def main():
    # step 1, parser
    #------------------------------
    parser = HfArgumentParser((KNNArguments,TestArguments))
    knn_args, args= parser.parse_args_into_dataclasses()
    if knn_args.knn_temp is not None:
        print("knn_temp is :", knn_args.knn_temp)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring',
                     'paraformer_beam_search', ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    #------------------------------
    # step 2, init dataset and model
    # remember to switch dataset
    # when building index , use the training set
    # when test decoing method, use the testing set
    
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_model(configs)
    print(model)
    
    # Load dict
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    
    #------------------------------
    #step 3, insert KNN
    dimension = configs['encoder_conf']['output_size']
    vocab_size = configs['output_dim']
    knn_wrapper = None
    
    if knn_args.build_index:
        knn_wrapper = KNNSaver_for_ctc(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir,
                                   dimension=dimension, knn_keytype=knn_args.knn_keytype, 
                                   use_null_mask=knn_args.use_null_mask)
        knn_wrapper.register(model, knn_args.thr, vocab_size)
    elif knn_args.knn:
        knn_wrapper = KNNWrapper_for_ctc(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir,
                                     dimension=dimension,
                                     knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
                                     no_load_keys=knn_args.no_load_keys,
                                     move_dstore_to_mem=knn_args.move_dstore_to_mem,
                                     knn_gpu=knn_args.knn_gpu,
                                     recompute_dists=knn_args.recompute_dists,
                                     k=knn_args.k, lmbda=knn_args.lmbda, knn_temp=knn_args.knn_temp,
                                     probe=knn_args.probe, decode_skip_blank=knn_args.decode_skip_blank,
                                     scale_lmbda=knn_args.scale_lmbda, scale_lmbda_temp=knn_args.scale_lmbda_temp)
        knn_wrapper.register(model) 
        
    #------------------------------
    # step 4: inference
    start = time.time()
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            if args.mode == 'ctc_greedy_search':
                hyps, _ = model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    )
            elif args.mode =='knn_ctc':
                hyps, _ = model.knn_ctc_greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    knn_args=knn_args,
                    knn_wrapper=knn_wrapper,
                    )

            for i, key in enumerate(keys):
                content = []
                for w in hyps[i]:
                    if w == eos:
                        break
                    content.append(char_dict[w])
                logging.info('{} {}'.format(key, args.connect_symbol
                                            .join(content)))
                fout.write('{} {}\n'.format(key, args.connect_symbol
                                            .join(content)))

    logger.info('eval took {} s'.format(time.time() - start))
    # print('eval took {} s'.format(time.time() - start))
    if knn_args.build_index:
        knn_wrapper.build_index()

    if knn_args.build_index:
        dstore_index = knn_wrapper.show_dstore_index()
        current_path = os.path.dirname(args.result_file)
        with open(os.path.join(current_path, 'index_size'),'w') as f:
            f.write("the total datastore size is :" + str(dstore_index))
            
# --------------------------------------------------------------------------------------- 
if __name__ == "__main__":
    main()
    
    
