#!/usr/bin/env python3
#=========================================================================================================================================
import logging
import threading
import re, termcolor, opencc, time, asyncio, websockets
from nano_llm.test.zh_prompts import zh_prompts_list

from nano_llm import Agent, Pipeline, StopTokens, BotFunctions, bot_function, Plugin, NanoLLM, ChatHistory 
from nano_llm.utils import ArgParser, print_table, KeyboardInterrupt, resample_audio

from nano_llm.plugins import (
    UserPrompt, ChatQuery, PrintStream, 
    AutoASR, AutoTTS, VADFilter, RateLimit,
    ProcessProxy, AudioInputDevice, AudioOutputDevice, AudioRecorder
)

"""
                                                            /-------------------->  external BASH of bibot command
pipeline architecture :               >👻           @🤖    /
    AudioInputDevice -> VADFilter ->  ASR -----> BiBotAgent --------------------->  TTS -->  RateLimit  --> AudioOutputDevice
                                              /            \                    /
                                 >🤡         /              \        #👹       /
                              UserPrompt ---+                +---->  LLM -----+
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
            # Create Plugin objects for ASR and VAD
            #-------------------------------------------------------------
            self.asr = AutoASR.from_pretrained(asr=asr, **kwargs) 
            self.vad = VADFilter(**kwargs)
            self.audio_input = AudioInputDevice(**kwargs)

            # Build up Plugin connections
            #-------------------------------------------------------------
            Pipeline([self.audio_input, self.vad, self.asr])
        
            self.asr.add(PrintStream(partial=False, prefix='👻👻 ', color='blue'),    AutoASR.OutputFinal)
            self.asr.add(PrintStream(partial=False, prefix='👻>> ', color='magenta'), AutoASR.OutputPartial)
            
            self.asr.add(self.asr_partial, AutoASR.OutputPartial)   # pause output when user is speaking
            #self.asr.add(self.asr_final,   AutoASR.OutputFinal)     # clear queues on final ASR transcript

            self.asr_history = None  # store the partial ASR transcript

        #-------------------------------------------------------------------------------------
        #: The TTS plugin that speaks the Audio output.
        #-------------------------------------------------------------------------------------
        if not tts or isinstance(tts, str):
            # Create Plugin objects for TTS and RateLimit
            #-------------------------------------------------------------
            self.tts = AutoTTS.from_pretrained(tts=tts, **kwargs) 
            self.tts_ratelimit = RateLimit(rate=1.0, drop_inputs=True, chunk=int(kwargs.get('sample_rate_hz')/10)) # slow down TTS to realtime and be able to pause it
            self.tts.add(self.tts_ratelimit)

            self.tts.add(self.on_tts_begin)
            self.tts_ratelimit.add(self.on_tts_progress)
            self.TTS_synthesizing = False

            # Create Plugin objects for AudioOutput (device and/or file)
            #-------------------------------------------------------------
            if kwargs.get('audio_output_device') is not None:
                self.audio_output_device = AudioOutputDevice(**kwargs)
                self.tts_ratelimit.add(self.audio_output_device)    # self.tts_ratelimit | self.tts
            
            if kwargs.get('audio_output_file') is not None:
                self.audio_output_file = AudioRecorder(**kwargs)
                self.tts_ratelimit.add(self.audio_output_file)

        #-------------------------------------------------------------------------------------
        # LLM / BiBotAgent /UserPrompt pipeline
        #-------------------------------------------------------------------------------------
        # Create Plugin objects for ASR and VAD
        self.llm = ChatQuery(drop_inputs=True, **kwargs)        #: The LLM model plugin (like ChatQuery)
        self.prompt = UserPrompt(interactive=True, **kwargs)    #: Text prompts input for CLI.
        self.bibot = BiBotAgent(**kwargs)                       #: The BiBot Agent plugin (handle ROBOT voice commands)

        # Build up Plugin connections
        #-----------------------------------------------------------------
        self.bibot.add(PrintStream(partial=False, prefix='🤖🤖 ', color='red'),      BiBotAgent.OutputTTS)
        self.bibot.add(PrintStream(partial=False, prefix='🤖>> ', color='magenta'),  BiBotAgent.OutputLLM)
        self.llm.add(  PrintStream(partial=False, prefix='👹👹 ', color='green'),    ChatQuery.OutputFinal)
        self.llm.add(  PrintStream(partial=False, prefix='👹>> ', color='magenta'),  ChatQuery.OutputWords)

        if self.asr:
            self.asr.add(  self.bibot, AutoASR.OutputFinal)     # Text prompts from ASR Audio.
        self.prompt.add(   self.bibot)                          # Text prompts from UserPrompt CLI.

        self.bibot.add(    self.llm,   BiBotAgent.OutputLLM)    # send the LLM query to the LLM
        if self.tts:
            self.bibot.add(self.tts,   BiBotAgent.OutputTTS)    # send the audio output to the TTS
            self.llm.add(  self.tts,   ChatQuery.OutputFinal)   # send the LLM query to the TTS

        #-------------------------------------------------------------------------------------
        # setup pipeline with two entry nodes
        #-------------------------------------------------------------------------------------
        self.pipeline = [self.prompt, self.audio_input]

        BiBotAgent.unitest()
        self.print_input_prompt()

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
            self.tts_ratelimit.pause(1.0)

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
            self.tts_ratelimit.interrupt(block=False, recursive=False) # might be paused/asleep
 
    def on_tts_begin(self, input):
        self.last_monitor_time = time.perf_counter()
        if not self.TTS_synthesizing:
            logging.info(f"TTS_synthesizing: SPEAKER started")
            self.TTS_synthesizing = True
            threading.Thread(target=self.tts_synth_monitor).start()

    def on_tts_progress(self, input):
        self.audio_output_device.unpause()
        self.last_monitor_time = time.perf_counter()

    def tts_synth_monitor(self):
        #logging.debug(f"TTS_synthesizing:  Monitoring started")
        while self.TTS_synthesizing:
            if (time.perf_counter() - self.last_monitor_time) >= 5:    # Check if 5 seconds have passed in silence
                self.asr.interrupt(recursive=False)                    # clear ASR for captured TTS-feedbacked voice
                self.bibot.interrupt(recursive=False)
                self.bibot.tts_stopped()
                self.audio_output_device.pause()
                self.TTS_synthesizing = False
                self.print_input_prompt()
                logging.info(f"TTS_synthesizing: SPEAKER stopped")
            time.sleep(0.5)          # Avoid busy-waiting; check every 0.5 seconds
        #logging.debug(f"TTS_synthesizing:  Monitoring Stopped")
        
    def print_pipeline_status(self, input):
        dbg_txt =  f"ASR={self.asr.input_queue.qsize()} " + \
                   f"TTS={self.tts.input_queue.qsize()} " + \
                   f"LLM={self.llm.input_queue.qsize()} " + \
                   f"RLMT={self.tts_ratelimit.input_queue.qsize()} " + \
                   f"USER={self.prompt.input_queue.qsize()} " + \
                   f"BIBOT={self.bibot.input_queue.qsize()} "
        dbg_txt += f"  FSM={self.bibot.fsm_state}  TTS-SYN={self.TTS_synthesizing} "
        termcolor.cprint(f"👾👾 Pulgins input_queue size: {dbg_txt}\n", 'red', end='', flush=True)

    def print_input_prompt(self):
        termcolor.cprint('🤡🤡 PROMPT: ', 'blue', end='', flush=True)
        

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
    # converter object for Simplified to Traditional Chinese ('t2s': Traditional to Simplified)
    zh_converter = opencc.OpenCC('s2t')

    #-------------------------------------------------------------------------------------
    # Define all patterns with their variants in a structured dictionary
    PATTERN_LISTS = {
        "VERBS": ["去抓", "去拿", "我想", "我要", "幫我拿", "幫我抓" ],
        "OBJECTS": ["巧克力口味", "草莓口味", "牛奶口味" ],
        "COLORS": ["紅色",       "粉紅色",  "淺黃色" ],
        "STOP": ["不要[說吵]", "停止", "停停停", "停下來" ],
        "EMPTY": ["[。\., ]*" ],
        "SECRETS": ["微妙不可思議", "維妙不可思議", "惟妙不可思議", "慈悲喜捨", "信解受持" ]
    }
    
    # Dynamically generate regex patterns
    BIBOT_PATTERNS = {}
    for pattern_type, pattern_list in PATTERN_LISTS.items():
        BIBOT_PATTERNS[pattern_type] = r"(" + "|".join(pattern_list) + r")"

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
        matches = {}
        text = BiBotAgent.zh_converter.convert(text)
        
        # Process each pattern type
        for pattern_type, pattern in BiBotAgent.BIBOT_PATTERNS.items():
            match = re.search(pattern, text)
            matches[pattern_type] = match.group(1) if match else None
            
        matched_verb = matches["VERBS"]
        matched_obj = matches["OBJECTS"]
        matched_color = matches["COLORS"]
        matched_secret = matches["SECRETS"]
        matched_stop = matches["STOP"]
        matched_empty = matches["EMPTY"]
        
        # Find index of matched color in the color list
        # index_color = None
        # if matched_color:
        #     try:
        #         index_color = BiBotAgent.PATTERN_LISTS["COLORS"].index(matched_color)
        #     except ValueError:
        #         pass
        index_color   = BiBotAgent.PATTERN_LISTS["COLORS"].index(matched_color) if matched_color in BiBotAgent.PATTERN_LISTS["COLORS"] else None
        
        # Determine the result based on the matches
        if matched_verb and matched_color:      return 1, BiBotAgent.PATTERN_LISTS["OBJECTS"][index_color]
        elif matched_obj:                       return 2, matched_obj
        elif matched_verb:                      return 3, "NEED_USER_CONFIRM"
        elif matched_secret:                    return 4, "SECRETS"
        elif matched_stop:                      return 0, "STOP"
        elif matched_empty:                     return -2, ""
        else:                                   return -1, text
        
    #-------------------------------------------------------------------------------------
    @staticmethod
    def unitest():
        """
        Unit test for the match_bibot_patterns function.
        """
        # Test cases
        test_cases = [
            "去抓蓝色的盒子", "我想要绿色的苹果口味", "我想吃黄色的凤梨口味", "去拿超好吃的口味",
             "巧克力口味", "帮我拿红色口味",   "草莓口味", "去抓粉红色口味",   "牛奶口味", "去抓浅黄色口味",
             "一个蓝色的盒子", "我只是想想", "口味如何", "完全不相关的文字",
            "去抓藍色的盒子", "我想要綠色的蘋果口味", "我想吃黃色的鳳梨口味", "去拿超好吃的口味",
            "幫我拿紅色口味",   "去抓粉紅色口味",    "去抓淺黃色口味",
            "一個藍色的盒子", "完全不相關的文字"
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
        self.fsm_state = "FSM_IDLE"  # FSM state

    #-------------------------------------------------------------------------------------
    def stop_tts(self, fsm_state="FSM_TTS_STOPPING"):
        """
        Cancel the TTS synthesizing
        """
        agent.on_interrupt()
        self.fsm_state = fsm_state

    def tts_stopped(self):
        self.fsm_state = "FSM_IDLE"

    #-------------------------------------------------------------------------------------
    def process(self, input, channel=None,  **kwargs):
        """
        Check the text patterns for ROBOT ACTION commands:
            VERB: 去抓 | 去拿| 我想 | 我要 | 幫我拿 | 幫我抓
            NOUN: 巧克力口味 | 紅色 | 草莓口味 | 粉紅色 | 牛奶口味 | 淺黃色
        param: `channel=None` is not only to ignore from input side, but also mainly taking out from `kwargs`
        """

        if self.interrupted:
            logging.debug(f"BiBotAgent interrupted (input={len(input)})")
            return
        
        #---------------------------------------------------------------------------------
        # Console commands: straight to execute without FSM
        if input == "/status":
            agent.print_pipeline_status(input)    # print the status of the pipeline
            return
        elif input == "/cancel" or input == "/stop":
            logging.info(f"BiBotAgent will cancel TTS synthesizing: {input}")
            self.stop_tts()
            return

        #---------------------------------------------------------------------------------
        result, text = BiBotAgent.match_bibot_patterns(input.strip())
        logging.debug(f"BiBotAgent FSM State: {self.fsm_state}, USER INPUT: '{input}', MATCHED result={result}, text='{text}'\n\tkwargs='{kwargs}'")

        #---------------------------------------------------------------------------------
        # FSM_IDLE: the system should have CLEAN state by reset CHAT history
        # FSM_TTS_ROBOT: Action taken for ROBOT control with TTS output. TTS can't be stopped in this state
        # FSM_TTS_SPEAK: TTS ouput text from LLM & etc, can be stopped by new ROBOT Voice command or STOP command
        # FSM_TTS_STOPPING: special state indicating TTS is on stopping
        #---------------------------------------------------------------------------------
        if self.fsm_state == "FSM_IDLE":
            agent.llm.history.reset()         # Reset the chat history

        if result == 0:
            #-----------------------------------------------------------------------------
            # STOP command: to cancel the TTS synthesizing
            #-----------------------------------------------------------------------------
            if self.fsm_state == "FSM_TTS_ROBOT":
                logging.debug(f"BiBotAgent can NOT stop ROBOT action: {self.last_text}")
            else:
                logging.info(f"BiBotAgent will stop TTS synthesizing: {text}")
                self.stop_tts()
                return

        elif result == 1 or result == 2:
            #-----------------------------------------------------------------------------
            # ACTION command: to send to external WS Server
            #-----------------------------------------------------------------------------
            if self.fsm_state == "FSM_TTS_SPEAK":
                agent.on_interrupt()
            
            self.output( f"榮幸之至, 且待片刻, 將為汝取: {text}", channel=BiBotAgent.OutputTTS, final=True)
            asyncio.run(BiBotAgent.send_ws_commands(text))
            logging.info(f"Will invoke external ROBOT to pick object: '{text}'")
            self.last_text = text
            self.fsm_state = "FSM_TTS_ROBOT"

        elif result == 3:
            #-----------------------------------------------------------------------------
            # USER_CONFIRM command: to confirm with the user 
            #-----------------------------------------------------------------------------
            if self.fsm_state != "FSM_TTS_SPEAK":
                self.output( "汝欲求何事,  我將為汝效勞?", channel=BiBotAgent.OutputTTS, final=True)
                self.fsm_state = "FSM_TTS_SPEAK"

        elif result == 4:
            #-----------------------------------------------------------------------------
            # SECRET command: TTS normal speaking
            #-----------------------------------------------------------------------------
            if self.fsm_state != "FSM_TTS_SPEAK":
                for line in zh_prompts_list:
                    self.output( line, channel=BiBotAgent.OutputTTS, partial=True)
                    logging.debug(f"secret workds: '{line}'")
                self.output( "The end", channel=BiBotAgent.OutputTTS, final=True)
                self.fsm_state = "FSM_TTS_SPEAK"

        elif text != "":
            #-----------------------------------------------------------------------------
            # LLM queries: TTS normal speaking
            #-----------------------------------------------------------------------------
            if self.fsm_state != "FSM_TTS_SPEAK":
                self.output( text, channel=BiBotAgent.OutputLLM, final=True, **kwargs)   # partial=True
                self.fsm_state = "FSM_TTS_SPEAK"

    #-------------------------------------------------------------------------------------
    async def send_ws_commands(command:str):
        uri = "ws://localhost:52560"

        async with websockets.connect(uri) as websocket:
            await websocket.send(command)
            logging.debug(f"Sent: '{command}'")


#=========================================================================================================================================
if __name__ == "__main__":
    parser = ArgParser(extras=ArgParser.Defaults+['asr', 'tts', 'audio_input', 'audio_output', 'log'])
    args = parser.parse_args()
    
    agent = BiBotChatLLM(**vars(args))
    interrupt = KeyboardInterrupt()
    
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop()
    

