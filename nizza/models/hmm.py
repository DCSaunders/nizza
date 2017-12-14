# coding=utf-8
# Copyright 2017 The Nizza Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the neural alignment HMM model. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from nizza.nizza_model import NizzaModel
from nizza.models.model2 import BaseModel2
from nizza.utils import common_utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def register_hparams_sets():
  base = hparam.HParams(
    lex_hidden_units=[512, 512, 512],
    dist_hidden_units=[128],
    inputs_embed_size=512,
    pos_embed_size=128,
    activation_fn=tf.nn.relu,
    max_timescale=250.0,
    logit_fn=tf.sigmoid, # tf.exp, tf.sigmoid
    dropout=None
  )
  all_hparams = {}
  all_hparams["hmm_default"] = base
  return all_hparams


def register_models():
  return {"hmm": HMM}


class HMM(BaseModel2):
  def precompute(self, features, mode, params):
    """We precompute the lexical translation logits for each src token and the
    IxJ table of distortion logits.
    """
    lex_logits = self.compute_lex_logits(features, 'inputs', params)
    I = tf.cast(common_utils.get_sentence_length(features["inputs"]), tf.int32)
    dist_logits = self.compute_distance_scores(I, params)
    return lex_logits, dist_logits


  def compute_distance_scores(self, I, params):
    """Compute the unnormalized DistNet scores.
    Args:
      I: A [batch_size] int32 tensor with source sentence lengths
      params: hyper-parameters for that model
    Returns:
      A [batch_size, max_src_len, max_src_len] float32 tensor where the
      entry at [b, i, i'] stores DistNet(i, i', I[b])
    """
    max_i = tf.reduce_max(I)
    batch_size = tf.shape(I)[0]
    expand_I = common_utils.expand_to_shape(I, ["x", max_i, max_i])
    expand_i = common_utils.expand_to_shape(tf.range(max_i), [batch_size, "x", max_i])
    expand_i_dash = common_utils.expand_to_shape(tf.range(max_i), [batch_size, max_i, "x"])
    int_inputs = tf.stack([expand_I, expand_i, expand_i_dash], axis=-1)
    max_pos = max_i + 1
    pos_embeds = self.compute_positional_embeddings(
        max_pos, params, params.pos_embed_size, params.max_timescale)
    embedded = tf.gather(pos_embeds, int_inputs)
    net = tf.reshape(embedded, [batch_size, max_i, max_i, params.pos_embed_size * 3])
    return self.compute_dist_net(net, params)


  def compute_loss(self, features, mode, params, precomputed):
    '''
    lex_probs_num[b, i, t] stores p(t|e_i) for the example b - shape [batch_size, src_len, trg_vocab]
    dist_logits[b, i, i'] stores DistNet(i, i', I[b])
    '''
    lex_probs_num, dist_logits = precomputed
    inputs = tf.cast(features["inputs"], tf.int32)
    targets = tf.cast(features["targets"], tf.int32)
    inputs_weights = common_utils.weights_nonzero(inputs)   
    targets_weights = common_utils.weights_nonzero(targets) 
    batch_size = tf.shape(inputs)[0]
    max_src_len = tf.shape(inputs)[1]
    max_trg_len = tf.shape(targets)[1]

    dist_logits_zeroed = self.input_weight_mask(features, dist_logits)
    dist_logits_denom = tf.reduce_sum(dist_logits_zeroed, axis=1, keep_dims=True)
    dist_probs = tf.expand_dims(common_utils.safe_div(dist_logits_zeroed, dist_logits_denom), -1)

    targets_repeated = tf.tile(tf.expand_dims(targets, 1), tf.convert_to_tensor([1, max_src_len, 1]))
    # targets_repeated is [batch_size, src_len, trg_len]

    factors = self.get_lex_score_factors(lex_probs_num, inputs)
    lex_probs_flat = tf.reshape(lex_probs_num, [batch_size*max_src_len, params.targets_vocab_size])
    targets_flat = tf.reshape(targets_repeated, [batch_size*max_src_len, max_trg_len])
    lex_scores_flat = common_utils.gather_2d(lex_probs_flat, targets_flat)
    lex_scores = tf.reshape(lex_scores_flat, [batch_size, max_src_len, max_trg_len])
    lex_probs = factors * lex_probs
    last_forward_probs = self.compute_forward_probs(batch_size, max_src_len, max_trg_len,
                                                    lex_probs, dist_probs) 
    # last_forward_probs: [batch_size, src_len, 1]
    return -common_utils.safe_log(tf.reduce_sum(last_forward_probs, 1))

  def get_lex_score_factors(self, lex_probs_num, inputs):
    lex_probs_denom = tf.reduce_sum(lex_probs_num, axis=-1)
    inputs_weights = common_utils.weights_nonzero(inputs)   
    factors = tf.expand_dims(common_utils.safe_div(inputs_weights, lex_probs_denom), -1)
    # factors is [batch_size, src_len, 1]
    return factors
  
  def compute_forward_probs(self, batch_size, max_src_len, max_trg_len, lex_scores, hmm_scores):
    '''
    calculate Baum-Welch forward probabilities a_j(i) =  p(f_j|e_i) sum_i' P(i|i')a_{j-1}(i')

    lex_scores:  [batch_size, src_len, trg_len], contains p(f_j|e_i)
    hmm_scores: [batch_size, src_len, src_len], contains p(i|i', b[I])
    '''
    forward_probs = [tf.gather(lex_scores, [0], axis=2) * tf.gather(hmm_scores, [0], axis=2)]
    for j in range(1, max_trg_len):
      reshaped_forward = tf.reshape(forward_probs[-1], [batch_size, 1, max_src_len]
      lex_prob = tf.gather(lex_scores, [j], axis=2) # shape [batch_size, src_len, 1]
      alpha_j = lex_prob * tf.reduce_sum(reshaped_forward * hmm_scores, -1, keep_dims=True)
      forward_probs.append(alpha_j)
    return forward_probs[-1]

  def input_weight_mask(self, features, to_mask):
    inputs_weights = common_utils.weights_nonzero(features['inputs'])   
    ax1_masked = tf.expand_dims(inputs_weights, -1) * to_mask
    max_src_len = tf.shape(features['inputs'])[1]
    tiled_mask = tf.tile(tf.expand_dims(inputs_weights, 1), [1,max_src_len,1])
    return ax1_masked * tiled_mask

  def predict_next_word(self, features, params, precomputed):
    lex_probs_num, dist_logits = precomputed

    dist_probs_num = self.input_weight_mask(features, dist_logits)
    dist_probs_denom = tf.reduce_sum(dist_probs_num, axis=1, keep_dims=True)
    dist_probs = tf.expand_dims(common_utils.safe_div(dist_probs_num, dist_probs_denom), -1)
    # dist_probs is [batch_size, max_src_len, 1]
    factors = self.get_lex_score_factors(lex_probs_num, features['inputs'])
    lex_probs = factors * lex_probs_num
    last_forward_probs = self.compute_forward_probs(batch_size, max_src_len, max_trg_len,
                                                    lex_probs, dist_probs)                           
    return common_utils.safe_log(tf.reduce_sum(last_forward_probs, axis=1))

    
