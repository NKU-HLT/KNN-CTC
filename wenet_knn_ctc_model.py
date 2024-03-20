import os

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path

import faiss
import faiss.contrib.torch_utils

import math
import random
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(20)

from datetime import datetime
current_time = datetime.now().strftime("%Y_%m_%d")

file_handler = logging.FileHandler("./log/"+current_time+'.log', mode='w')
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# distance metrics
class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()


class KNNWrapper_for_ctc(object):
    def __init__(self, dstore_size, dstore_dir, dimension,
                 knn_sim_func=None, knn_keytype=None,
                 no_load_keys=False, move_dstore_to_mem=False, knn_gpu=True,
                 recompute_dists=False,
                 k=1024, lmbda=0.25, knn_temp=1.0, probe=32, decode_skip_blank=False,
                 scale_lmbda=False, scale_lmbda_temp=1.0):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_input_ids = None
        self.keys = None
        self.values = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.decode_skip_blank= decode_skip_blank
        self.scale_lmbda = scale_lmbda
        self.scale_lmbda_temp =scale_lmbda_temp

        dist_type_to_dist_func = {
            DIST.l2: KNNWrapper_for_ctc.l2,
            DIST.dot: KNNWrapper_for_ctc.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[knn_sim_func]  # l2 or dot product function

    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index_name = get_index_path(self.dstore_dir, self.dstore_size, self.dimension)
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        cpu_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n),
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        cpu_index.make_direct_map()

        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.dstore_size,
                                           self.dimension)
        # read a memory-map to an array stored in a binary file on disk.
        if not self.no_load_keys:
            self.keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_size, self.dimension))
        self.vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                              shape=(self.dstore_size, 1))
        # self.vals = torch.from_numpy(self.vals).to(self.device)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            logger.info('Loading to memory...')
            start = time.time()

            if not self.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(f'{keys_vals_prefix}_keys.npy',
                                                  dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = self.keys_from_memmap[:].astype(np.float16)

            del self.vals
            vals_from_memmap = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                                         shape=(self.dstore_size, 1))
            self.vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            del vals_from_memmap
            logger.info('Loading to memory took {} s'.format(time.time() - start))

        return cpu_index, gpu_index

    def register(self, model):
        self.model = model
        self.reconstruct_index, self.index = self.setup_faiss()

    def process(self, hidden_states, am_logits):
        '''
        :param hidden_states: # (batch, time, dim)
        :param am_logits: # (batch, time, vocab)
        :return: interpolated_scores
        '''

        queries = hidden_states  # (batch, time, dim)

        # vocab should be assigned once
        self.vocab_size = am_logits.shape[-1]
        dists, knns = self.get_knns(queries)  # (batch, time, k)

        if self.recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
            dists = self.dist_func(queries, knns_vecs)

        neg_dists = -dists
        knn_log_probs, vals_at_knns = self.knns_to_log_prob(knns, neg_dists)

        # Time*Vocab -> 1*Time*Vocab
        knn_log_probs = torch.unsqueeze(knn_log_probs, dim=0)
        
        if self.scale_lmbda:
            lmbda = self.dynamic_scale_lmbda(dists)
            interpolated_scores = KNNWrapper_for_ctc.scale_lmbda_interpolate(knn_log_probs, am_logits, 
                                                                lmbda, self.decode_skip_blank)
        else:
            interpolated_scores = KNNWrapper_for_ctc.interpolate(knn_log_probs, am_logits, 
                                                         self.lmbda, self.decode_skip_blank)  # (nonpad, vocab)
        
        return interpolated_scores, dists, vals_at_knns

    
    def dynamic_scale_lmbda(self,dists):
        relu = nn.ReLU()
        min_dists = dists[:,0]
        return relu(1-min_dists/self.scale_lmbda_temp)*self.lmbda
        # return relu(1-min_dists/self.scale_lmbda_temp)
    
    @staticmethod
    def scale_lmbda_interpolate(knn_log_probs, am_log_probs, lmbda, decode_skip_blank):
        # if flag is true , skip the frames which ctc-pseudo label are null
        if decode_skip_blank:
            am_logits = am_log_probs.squeeze()
            predicted_ids = torch.argmax(am_logits, dim=-1)
            mask = predicted_ids!=0
            interpolated = am_log_probs.clone()
            L, dict_size = am_log_probs.size(1), am_log_probs.size(2)
            interpolated[0,mask] = torch.logaddexp(am_log_probs[0,mask] + torch.log(1 - lmbda).unsqueeze(0).unsqueeze(-1).expand(1, L, dict_size)[0,mask], 
                                            knn_log_probs[0,mask] + torch.log(lmbda).unsqueeze(0).unsqueeze(-1).expand(1, L, dict_size )[0,mask])
        else:
            L, dict_size = am_log_probs.size(1), am_log_probs.size(2)
            interpolated = torch.logaddexp(am_log_probs + torch.log(1 - lmbda).unsqueeze(0).unsqueeze(-1).expand(1, L, dict_size), 
                                            knn_log_probs + torch.log(lmbda).unsqueeze(0).unsqueeze(-1).expand(1, L, dict_size ))
        return interpolated
    
    def get_knns(self, queries):
        '''
        queries: 1 * L * k(1024)
        dists: L * k
        knn: L*k
        '''
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(torch.squeeze(queries, 0), self.k)  # queries (batch*times, vocab)
        dists, knns = dists.to(self.device), knns.to(self.device)
        return dists, knns

    def knns_to_log_prob(self, knns, neg_dists):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        # when bz =1 , we use squeeze, but if bz > 1, it will go wrong
        vals_at_knns = self.vals[knns].squeeze(-1)  # (nonpad batch * time, k)

        knn_log_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs).log()  # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs, vals_at_knns

    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys) ** 2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)

    @staticmethod
    def interpolate(knn_log_probs, am_log_probs, lmbda, decode_skip_blank):
        # if flag is true , skip the frames which ctc-pseudo label are null
        if decode_skip_blank :
            am_logits = am_log_probs.squeeze()
            predicted_ids = torch.argmax(am_logits, dim=-1)
            mask = predicted_ids!=0
            interpolated = am_log_probs.clone()
            interpolated[0, mask] = torch.logaddexp(am_log_probs[0, mask] + np.log(1 - lmbda), knn_log_probs[0, mask] + np.log(lmbda))
        else:
            interpolated = torch.logaddexp(am_log_probs + np.log(1 - lmbda), knn_log_probs + np.log(lmbda))
        return interpolated


class KNNSaver_for_ctc(object):
    def __init__(self, dstore_size, dstore_dir, dimension, knn_keytype=None, use_null_mask=True):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = 0
        self.dstore_keys = None
        self.dstore_vals = None
        self.labels = None

        self.thr = 0.0
        self.char_idx = None
        self.char_max_num = None
        self.vocab_size = 0
        self.char_reach_num = None
        self.use_null_mask = use_null_mask

        logger.info(f'keytype being saved: {self.knn_keytype}')
        logger.info('Saving fp16')

    def register(self, model, thr, vocab_size):
        # init parameter
        self.model = model
        
        self.thr = thr
        if thr != 0:
            logger.info(f'confidence thr is {self.thr}')
            
        self.char_idx = [0 for i in range(vocab_size)]
        self.vocab_size = vocab_size

        # init file, this part should be run only once
        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.dstore_size, self.dimension)
        keys_filename = f'{keys_vals_prefix}_keys.npy'
        vals_filename = f'{keys_vals_prefix}_vals.npy'
        if os.path.exists(keys_filename) and os.path.exists(vals_filename):
            mode = 'r'
        else:
            mode = 'w+'
            Path(keys_filename).parent.mkdir(parents=True, exist_ok=True)

        # memmap, 0, size:(self.dstore_size, self.dimension)
        self.dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode=mode,
                                     shape=(self.dstore_size, self.dimension))
        # memmap, 0, size:(self.dstore_size, 1)
        self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode=mode, shape=(self.dstore_size, 1))

    def process(self, captured_keys, captured_values, confidence=None):
        '''
        :param captured_keys: # (batch, time, dim)
        :param captured_values: # (batch, time)
        :param confidence: # (batch, time)
        :return: none
        '''

        # prepare keys and values
        captured_keys = captured_keys.flatten(0, 1)  # (batch * time, dim)
        captured_values = captured_values.flatten(0, 1)  # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        # here we need to fill out pairs with low confidence, using a thr.
        # if use logsoftmax, confidence should be between -inf and 0.
        
        if self.thr != 0 and confidence is not None:
            # bz = 1
            confidence = confidence.flatten(0, 1)
            thr_mask = confidence > self.thr
            keys = keys[thr_mask]
            values = values[thr_mask]
        
        # here we need to fill out label 0 
        if self.use_null_mask:
            keys, values = self.null_mask(keys, values)

        # here, we use reservoir sampling.
        self.reservoir_sampling(keys, values)

    def null_mask(self, keys, values):
        # here we need to fill out label 0
        label_mask = values != 0
        keys = keys[label_mask]
        values = values[label_mask]
        return keys, values

    def reservoir_sampling(self, keys, values):

        batch_time_size = keys.shape[0]
        
        if self.dstore_idx < self.dstore_size:
            if self.dstore_idx + batch_time_size > self.dstore_size:
                batch_time_size = max(self.dstore_size - self.dstore_idx, 0)
                keys = keys[:batch_time_size]
                values = values[:batch_time_size]

            try:
                self.dstore_keys[self.dstore_idx:(batch_time_size + self.dstore_idx)] = keys.cpu().numpy().astype(
                    np.float16)
                self.dstore_vals[self.dstore_idx:(batch_time_size + self.dstore_idx)] = values.unsqueeze(
                    -1).cpu().numpy().astype(np.int32)
            except ValueError as ex:
                logger.error(
                    f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
                logger.error(
                    f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
                raise ex
            self.dstore_idx += batch_time_size
            
        if self.dstore_idx >= self.dstore_size:
            for key, value in zip(keys, values):
                x = random.randint(0, self.dstore_idx)
                if x < self.dstore_size:
                    self.dstore_keys[x] = key.cpu().numpy().astype(np.float16)
                    self.dstore_vals[x] = value.unsqueeze(-1).cpu().numpy().astype(np.int32)
                self.dstore_idx += 1

    def show_dstore_index(self):
        print('dstore_size: '+ str(self.dstore_idx))
        return self.dstore_idx
        
    def show_char_idx_num(self):
        for i in range(self.vocab_size):
            print('char idx:', str(i), ' ', self.char_idx[i])

    def build_index(self, num_keys_to_add_at_a_time=1000000,
                    ncentroids=4096, seed=1, code_size=64, probe=32):
        logger.info('Building index')
        index_name = get_index_path(self.dstore_dir, self.dstore_size, self.dimension)

        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension,
                                 ncentroids, code_size, 8)
        index.nprobe = probe 

        logger.info('Training Index')
        np.random.seed(seed)

        # here we need to sample keys and vals.
        random_sample = np.random.choice(np.arange(self.dstore_vals.shape[0]),
                                         size=[min(1000000, self.dstore_vals.shape[0])], replace=False)
        start = time.time()
        '''
        To speed up the search, it is possible to segment the dataset into pieces. 
        This type of index requires a training stage, that can be performed on any collection of vectors that has the same distribution as the database vectors. 
        In this case we just use the database vectors themselves or subsample of the database vectors.
        '''
        # Faiss does not handle adding keys in fp16 as of writing this.
        index.train(self.dstore_keys[random_sample].astype(np.float32))
        logger.info(f'Training took {time.time() - start} s')

        logger.info('Adding Keys')

        start = 0
        start_time = time.time()
        while start < self.dstore_size:
            end = min(self.dstore_size, start + num_keys_to_add_at_a_time)
            to_add = self.dstore_keys[start:end].copy()
            index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
            start += num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                logger.info(f'Added {start} tokens so far')
                logger.info(f'Writing Index {start}')
                faiss.write_index(index, f'{index_name}')

        logger.info(f'Adding total {start} keys')
        logger.info(f'Adding took {time.time() - start_time} s')
        logger.info(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')


def get_dstore_path(dstore_dir, dstore_size, dimension):
    return f'{dstore_dir}/dstore_{dstore_size}_{dimension}'


def get_index_path(dstore_dir, dstore_size, dimension):
    return f'{dstore_dir}/index_{dstore_size}_{dimension}.indexed'
