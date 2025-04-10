#!/usr/bin/env python3
#=========================================================================================================================================
import logging
import threading
import re

from nano_llm import Agent, Pipeline, StopTokens, BotFunctions, bot_function, Plugin, NanoLLM, ChatHistory 
from nano_llm.utils import ArgParser, print_table, KeyboardInterrupt, resample_audio

from nano_llm.plugins import (
    UserPrompt, ChatQuery, PrintStream, 
    AutoASR, AutoTTS, VADFilter, RateLimit,
    ProcessProxy, AudioInputDevice, AudioOutputDevice, AudioRecorder
)

"""
pipeline architecture : 
    AudioInputDevice -> VADFilter ->  ASR -----> BiBotAgent --------------------->  TTS  (external bibot_command)
                                              /            \                    /
                               UserPrompt ---+              +---->  LLM -------+
"""

#=========================================================================================================================================
class BiBotChatLLM(Agent):
    """
    Agent for ASR → LLM → TTS pipeline.
    """
    def __init__(self, language_code : str = 'en-US',  sample_rate_hz : int = 44100,
                 asr : str = 'riva',  audio_input_device :  int = 1,
                 tts : str = 'piper', audio_output_device : int = 1,
                 **kwargs):
        """
        Args:
          asr (NanoLLM.plugins.AutoASR|str): the ASR plugin instance or model name to connect with the LLM.
          tts (NanoLLM.plugins.AutoTTS|str): the TTS plugin instance (or model name)- if None, will be loaded from kwargs.
        """
        kwargs.update(language_code=language_code, 
                      sample_rate_hz=sample_rate_hz, audio_input_device=audio_input_device, audio_output_device=audio_output_device,
                      system_prompt="你是一个中文人工智能助手, 请用中文回答所有问题." if language_code == 'zh-CN' else  "You are a helpful AI assistant.")
        super().__init__(**kwargs)

        logging.debug(f"BXU-DEBUG: audio_input_device={audio_input_device}, audio_output_device={audio_output_device}, asr={asr}\nkwargs='{kwargs}'")
        #-------------------------------------------------------------------------------------
        #: The ASR plugin that listen to the microphone AudioInput
        #-------------------------------------------------------------------------------------
        if not asr or isinstance(asr, str):
            self.asr = AutoASR.from_pretrained(asr=asr, **kwargs) 
        else:
            self.asr = asr
            
        self.vad = VADFilter(**kwargs).add(self.asr) if self.asr else None
        self.audio_input = AudioInputDevice(**kwargs).add(self.vad) if self.vad else None
        
        if self.asr:
            self.asr.add(PrintStream(partial=False, prefix='>> ', color='blue'), AutoASR.OutputFinal)
            self.asr.add(PrintStream(partial=False, prefix='>> ', color='magenta'), AutoASR.OutputPartial)
            
            self.asr.add(self.asr_partial, AutoASR.OutputPartial) # pause output when user is speaking
            self.asr.add(self.asr_final, AutoASR.OutputFinal)     # clear queues on final ASR transcript

            self.asr_history = None  # store the partial ASR transcript

        #-------------------------------------------------------------------------------------
        #: The TTS plugin that speaks the Audio output.
        #-------------------------------------------------------------------------------------
        if not tts or isinstance(tts, str):
            self.tts = AutoTTS.from_pretrained(tts=tts, **kwargs) 
        else:
            self.tts = tts
            
        if self.tts:
            self.tts_output = RateLimit(rate=1.0, chunk=9600) # slow down TTS to realtime and be able to pause it
            self.tts.add(self.tts_output)

            self.audio_output_device = kwargs.get('audio_output_device')
            self.audio_output_file = kwargs.get('audio_output_file')
            
            if self.audio_output_device is not None:
                self.audio_output_device = AudioOutputDevice(**kwargs)
                self.tts.add(self.audio_output_device)    # self.tts_output.add(self.audio_output_device)
            
            if self.audio_output_file is not None:
                self.audio_output_file = AudioRecorder(**kwargs)
                self.tts_output.add(self.audio_output_file)

        #-------------------------------------------------------------------------------------
        # LLM / BiBotAgent /UserPrompt pipeline
        #-------------------------------------------------------------------------------------

        #: The LLM model plugin (like ChatQuery)
        self.llm = ChatQuery(**kwargs) #ProcessProxy('ChatQuery', **kwargs)  

        #: Text prompts input for CLI.
        self.prompt = UserPrompt(interactive=True, **kwargs)

        #: The BiBot Agent plugin (handle ROBOT voice commands)
        self.bibot = BiBotAgent(**kwargs)

        # Text prompts from ASR Audio or UserPrompt CLI.
        if self.asr:
            self.asr.add(self.bibot, AutoASR.OutputFinal)  # runs after asr_final() and any interruptions occur
        self.prompt.add(self.bibot)

        self.bibot.add(self.llm, BiBotAgent.OutputLLM)     # send the LLM query to the LLM

        self.bibot.add(PrintStream(partial=False, prefix='@@ ', color='red'),    BiBotAgent.OutputLLM)
        self.bibot.add(PrintStream(partial=False, prefix='@@ ', color='magenta'), BiBotAgent.OutputTTS)

        if self.tts:
            self.bibot.add(self.tts, BiBotAgent.OutputTTS)  # send the audio output to the TTS
            #self.llm.add(self.tts, ChatQuery.OutputWords)   # send the LLM query to the TTS
            self.llm.add(self.tts, ChatQuery.OutputFinal)   # send the LLM query to the TTS

        self.llm.add(PrintStream(partial=False, prefix='## ', color='green'),   ChatQuery.OutputFinal)
        self.llm.add(PrintStream(partial=False, prefix='## ', color='magenta'), ChatQuery.OutputWords)

        #-------------------------------------------------------------------------------------
        # setup pipeline with two entry nodes
        self.pipeline = [self.prompt]

        if self.audio_input:                                 # if self.vad:
            self.pipeline.append(self.audio_input)           #     self.pipeline.append(self.vad)

    #-------------------------------------------------------------------------------------
    def asr_partial(self, text):
        """
        Callback that occurs when the ASR has a partial transcript (while the user is speaking).
        These partial transcripts get revised mid-stream until the user finishes their phrase.
        This is also used for pausing/interrupting the bot output for when the user starts speaking.
        """
        self.asr_history = text
        if len(text.split(' ')) < 2:
            return
        if self.tts:
            self.tts_output.pause(1.0)

    def asr_final(self, text):
        """
        Callback that occurs when the ASR outputs when there is a pause in the user talking,
        like at the end of a sentence or paragraph.  This will interrupt/cancel any ongoing bot output.
        """
        self.asr_history = None
        self.on_interrupt()
        
    def on_interrupt(self):
        """
        Interrupt/cancel the bot output when the user submits (or speaks) a full query.
        """
        self.llm.interrupt(recursive=False)
        if self.tts:
            self.tts.interrupt(recursive=False)
            self.tts_output.interrupt(block=False, recursive=False) # might be paused/asleep
 

#=========================================================================================================================================
class BiBotAgent(Plugin):
    """
    Inputs:  str -- text of USER input
     
    Outputs:  channel 0 (str) -- the response of recognized ROBOT ACTION text. Can connect to Plugin of TTS or PrintStream
              channel 1 (str) -- the general USER query text for LLM
    """
    OutputTTS = 0
    OutputLLM = 1

    #-------------------------------------------------------------------------------------
    VERBS_LIST   = ["去抓", "去拿", "我想", "我要", "幫我拿", "幫我抓", "帮我拿", "帮我抓"]
    OBJECTS_LIST = ["巧克力口味", "草莓口味", "牛奶口味", "牛奶口味"] 
    COLORS_LIST  = ["紅色",      "粉紅色",  "淺黃色",   "浅黄色"]

    BIBOT_VERBS   = r"(" + "|".join(VERBS_LIST) + r")"
    BIBOT_OBJECTS = r"(" + "|".join(OBJECTS_LIST) + r")"
    BIBOT_COLORS  = r"(" + "|".join(COLORS_LIST) + r")"

    #-------------------------------------------------------------------------------------
    @staticmethod
    def match_bibot_patterns(text):
        """
        Matches preset patterns and returns the matched verb and noun.

        Args:
            text: The input text string.

        Returns:
            A tuple containing:
                - The match result (1: confirmed ROBOT action, 0: likely a ROBOT action, need to double-check with USER,  -1: irrelevant to ROBOT)
                - The matched object (string or None).
        """
        verb_match  = re.search(BiBotAgent.BIBOT_VERBS,   text)
        obj_match   = re.search(BiBotAgent.BIBOT_OBJECTS, text)
        color_match = re.search(BiBotAgent.BIBOT_COLORS,  text)

        matched_verb  = verb_match.group(1)  if verb_match  else None
        matched_obj   = obj_match.group(1)   if obj_match   else None
        matched_color = color_match.group(1) if color_match else None
        index_color   = BiBotAgent.COLORS_LIST.index(matched_color) if matched_color in BiBotAgent.COLORS_LIST else None

        if matched_verb and matched_color:
            return 1, BiBotAgent.OBJECTS_LIST[index_color]
        elif matched_obj:
            return 1, matched_obj
        elif matched_verb:
            return 0, "USER_CONFIRM"
        else:
            return -1, text

    #-------------------------------------------------------------------------------------
    @staticmethod
    def unitest():
        """
        Unit test for the match_bibot_patterns function.
        """
        # Test cases
        test_cases = [
            "去抓蓝色的盒子", "我想要绿色的苹果口味", "我想吃黄色的凤梨口味", "去拿超好吃的口味",
             "巧克力口味", "帮我拿红色口味",   "草莓口味", "去抓粉红色口味",   "牛奶口味","去抓浅黄色口味",
             "一个蓝色的盒子", "我只是想想", "口味如何", "完全不相关的文字",
            "去抓藍色的盒子", "我想要綠色的蘋果口味", "我想吃黃色的鳳梨口味", "去拿超好吃的口味",
            "巧克力口味", "幫我拿紅色口味",   "草莓口味", "去抓粉紅色口味",   "牛奶口味","去抓淺黃色口味",
            "一個藍色的盒子", "我只是想想", "口味如何", "完全不相關的文字"
        ]

        # Run tests
        for text in test_cases:
            result = BiBotAgent.match_bibot_patterns(text)
            print(f"'{text}' - Result: {result}")


    #-------------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        """
        Plugin that feeds incoming text or ChatHistory to
        1) external BizLink Robot Controller for known command-text. Will invoke external shell script with TEXT: /opt/bibot/robot_cmd.sh
        2) LLM for general text queries and get the reply.
        """
        super().__init__(output_channels=2, **kwargs)

    #-------------------------------------------------------------------------------------
    def process(self, input, **kwargs):
        """
        Check the text patterns for ROBOT ACTION commands:
            VERB: 去抓 | 去拿| 我想 | 我要 | 幫我拿 | 幫我抓
            NOUN: 巧克力口味 | 紅色 | 草莓口味 | 粉紅色 | 牛奶口味 | 淺黃色
        """

        if self.interrupted:
            logging.debug(f"BiBotAgent interrupted (input={len(input)})")
            return
        
        result, text = BiBotAgent.match_bibot_patterns(input)
        logging.debug(f"BiBotAgent match ROBOT pattern: (input='{input}', result={result}, text='{text}')")
        logging.debug(f"BXU debug kwargs='{kwargs}' ")

        if result > 0:
            self.output( f"榮幸之至, 且待片刻, 將為汝取: {text}", channel=BiBotAgent.OutputTTS, final=True)
            logging.debug(f"Will invoke external ROBOT: /opt/bibot/robot_cmd.sh '{text}'")
        elif result == 0:
            self.output( "汝欲求何事,  我將為汝效勞?", channel=BiBotAgent.OutputTTS, final=True)
        else:
            self.output( text, channel=BiBotAgent.OutputLLM, final=True)
            #self.output( text, channel=BiBotAgent.OutputLLM, partial=True)


#=========================================================================================================================================
if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['asr', 'tts', 'audio_input', 'audio_output', 'log'])
    args = parser.parse_args()
    
    agent = BiBotChatLLM(**vars(args))
    interrupt = KeyboardInterrupt()
    BiBotAgent.unitest()
    
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop()
    

