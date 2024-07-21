#!/usr/bin/env python3
import time
import queue
import logging
import threading 

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from transformers.generation.streamers import BaseStreamer

from accelerate import init_empty_weights as init_empty_weights_ctx

from nano_llm import NanoLLM, StreamingResponse, KVCache
from nano_llm.utils import convert_tensor, ends_with_token


class HFModel(NanoLLM):
    """
    Huggingface Transformers model
    """
    def __init__(self, model_path, load=True, init_empty_weights=False, **kwargs):
        """
        Load model from path on disk or HuggingFace repo name.
        """
        super(HFModel, self).__init__(model_path, **kwargs)
    
        self.model_path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True).start()  
        
        if not load:
            return

        if init_empty_weights:
            with init_empty_weights_ctx():
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        else:
            if 'gtpq' in self.model_path:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, device=self.device, 
                    torch_dtype=torch.float16, low_cpu_mem_usage=True
                ).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path,
                    torch_dtype=torch.float16, low_cpu_mem_usage=True
                ).to(self.device).eval()
        
        try:
            self.has_embed = (self.model.get_input_embeddings() is not None)
        except Exception as error:
            pass

        if not self.has_embed:
            logging.warning(f"{type(self)} model {self.config.name} did not have text embedding layer (disabling input_embeds)")
                
        self.has_embed = False # TODO monkey-patching for this    
        self.load_config()
 
    def load_config(self):
        """
        @internal get the configuration info from the model
        """
        self.config.type = self.model.config.model_type
        self.config.max_length = self.model.config.max_length
        self.config.vocab_size = self.model.config.vocab_size
 
    def embed_tokens(self, tokens, return_tensors='np', **kwargs):
        """
        Generate embedding from token IDs
        """
        if not self.has_embed:
            raise RuntimeError(f"{self.config.name} does not have embed() in {self.module_path}")
        
        tokens = convert_tensor(tokens, return_tensors='pt', device=self.device)
        
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
            
        return convert_tensor(self.model.get_input_embeddings()(tokens), return_tensors=return_tensors)
         
    def generate(self, inputs, streaming=True, functions=None, **kwargs):
        """
        Generate output from input text, tokens, or an embedding.
        For detailed kwarg descriptions, see `transformers.GenerationConfig <https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig>`_.
        
        Args:
        
          inputs (str|ndarray): Text or embedding inputs to the model/
          
          streaming (bool): If True, an iterator will be returned that returns text chunks.
                            Otherwise, this function will block and return the generated text.
                              
          functions(list[callable]): Dynamic functions or plugins to run inline with token generation 
                                     for things like function calling, guidance, token healing, ect.
                                     These will be passed the text generated by the LLM so far, and any
                                     additional text that these return will be added to the chat.

          max_new_tokens (int): The number of tokens to output in addition to the prompt (default: 128)
          min_new_tokens (int): Force the model to generate a set number of output tokens (default: -1)
          do_sample (bool): If ``True``, temperature/top_p will be used.  Otherwise, greedy search (default: ``False``)
          repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty (default: 1.0)
          temperature (float): Randomness token sampling parameter (default=0.7, only used if ``do_sample=True``)
          top_p (float): If set to float < 1 and ``do_sample=True``, only the smallest set of most probable tokens.
                           with probabilities that add up to top_p or higher are kept for generation (default 0.95)
          stop_tokens (list[int]|list[str]): Stop generation if the bot produces tokens or text from this list (defaults to EOS token ID)
          kv_cache (np.ndarray): Previous kv_cache that the inputs will be appended to.  By default, a blank kv_cache 
                                will be created for each generation (i.e. a new chat).  This generation's kv_cache
                                will be set in the returned :class:`StreamingResponse` iterator after the request is complete.

        Returns:
          An asynchronous :class:`StreamingResponse` iterator (when ``streaming=True``) that outputs one decoded token string at a time.
          Otherwise, this function blocks and a string containing the full reply is returned after it's been completed.
        """
        if functions is None:
            functions = []
        elif not isinstance(functions, list):
            functions = [functions]

        stream = StreamingResponseHF(self, inputs, functions=functions, **kwargs)
        self.queue.put(stream)
        
        if streaming:
            return stream
        else:
            return stream.wait()
     
    def _generate(self, stream):
        """
        Process a generation request in model's inference thread.
        """
        generate_kwargs = {
            'max_new_tokens': stream.kwargs.get('max_new_tokens', 128),
            'do_sample': stream.kwargs.get('do_sample', False),
        }
        
        if generate_kwargs['do_sample']:
            generate_kwargs.update({
                'temperature': stream.kwargs.get('temperature', 0.7),
                'top_p': stream.kwargs.get('top_p', 0.95),
                'repetition_penalty': stream.kwargs.get('repetition_penalty', 1.0),
            })
            
        min_new_tokens = stream.kwargs.get('min_new_tokens', -1)
        
        if min_new_tokens > 0:
            generate_kwargs['min_new_tokens'] = min_new_tokens
        
        # if the stop tokens are strings, tokenize them
        stop_tokens = stream.kwargs.get('stop_tokens')
        
        if stop_tokens is not None:
            if isinstance(stop_tokens, int):
                stop_tokens = [stop_tokens]

            for i, stop in enumerate(stop_tokens):
                if isinstance(stop, str):
                    stop_tokens[i] = self.tokenize(stop).squeeze().tolist()
                
            generate_kwargs['stopping_criteria'] = [StopTokensCriteria(stop_tokens)]
                
        # convert inputs to tokens or embeddings
        if isinstance(stream.input, str):
            if self.has_embed:
                inputs = self.embed_text(stream.input, return_tensors='pt')
            else:
                inputs = self.tokenize(stream.input, return_tensors='pt')
        else:
            inputs = stream.input
            
        inputs = convert_tensor(inputs, return_tensors='pt', device=self.device)
        
        if torch.is_floating_point(inputs):
            total_tokens = inputs.shape[1]
            
            if stream.kv_cache is not None:
                total_tokens += len(stream.kv_cache)
                
            generate_kwargs['input_ids'] = torch.zeros(inputs.shape[0], total_tokens, dtype=torch.int32, device=self.device)
            generate_kwargs['inputs_embeds'] = inputs
            generate_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        else:
            generate_kwargs['input_ids'] = inputs  # TODO pad beginning token ID's with zeros so they are the total length

        self.stats.input_tokens = inputs.shape[1]
        self.stats.output_tokens = 0
        
        # optionally use previous kv cache
        if stream.kv_cache is not None:
            generate_kwargs['past_key_values'] = stream.kv_cache.state
            #self.stats.input_tokens -= len(stream.kv_cache)

        # begin generation
        self.time_begin_prefill = time.perf_counter()
        
        output = self.model.generate(
            streamer=stream,
            return_dict_in_generate=True,
            use_cache=True,
            **generate_kwargs
        )

        self.stats.decode_time = time.perf_counter() - self.time_begin_decode
        self.stats.decode_rate = (self.stats.output_tokens-1) / self.stats.decode_time # subtract one because timing didn't start until after the first token
        
        if stream.kv_cache is None:
            stream.kv_cache = KVCacheHF(self, output.past_key_values)
        else:    
            stream.kv_cache.update(output.past_key_values)

        stream.stopped = True
        stream.event.set()   

    def _run(self):
        """
        Run the generation requests thread.
        """
        while True:
            stream = self.queue.get()
            self._generate(stream)

    def _end_prefill(self):
        """
        Collect profiling info from the end of context prefill and beginning of output generation
        """
        self.time_begin_decode = time.perf_counter()
        self.stats.prefill_time = self.time_begin_decode - self.time_begin_prefill
        self.stats.prefill_rate = self.stats.input_tokens / self.stats.prefill_time
           
            
class StreamingResponseHF(StreamingResponse, BaseStreamer):
    """
    Asynchronous output iterator used by HF Transformer models.
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
    """
    def __init__(self, model, input, **kwargs):
        BaseStreamer.__init__(self)
        StreamingResponse.__init__(self, model, input, **kwargs)
        self.prefill = True

    def put(self, tokens):
        """Function that is called by `.generate()` to push new tokens"""
        if self.prefill:
            self.prefill = False  # HF pushes the prompt through first
            return
        elif self.model.stats.output_tokens == 0:
            self.model._end_prefill()

        self.model.stats.output_tokens += len(tokens)
        self.add_tokens(tokens, event=True)
    
    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        #logging.warning('StreamingResponseHF.end()')
        pass


class KVCacheHF(KVCache):
    """
    Interface for storing & manipulating the chat's KV cache.
    """
    def __init__(self, model, state=None):
        super().__init__()
        self.model = model
        self.update(state)
        
    def update(self, state):
        self.state = state
        
        if self.state is not None:
            # TODO this is one shorter than the input+output token lengths...
            # https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast.past_key_values
            self.num_tokens = self.state[0][0].shape[2] 
            
            
class StopTokensCriteria(StoppingCriteria):
    """
    https://github.com/huggingface/transformers/issues/26959
    """
    def __init__(self, stop_tokens):
        self.stop_tokens = stop_tokens
        self.max_tokens = 1
        
        for stop in stop_tokens:
            if isinstance(stop, list):
                self.max_tokens = max(self.max_tokens, len(stop))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return ends_with_token(input_ids[:,-self.max_tokens:].tolist(), self.stop_tokens)
        
