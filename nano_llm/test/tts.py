#!/usr/bin/env python3
#
# Interactively test streaming TTS models, with support for live output and writing to wav files.
# See the print_help() function below for a description of the commands you can enter this program.
# Here is an example of starting it with XTTS model and sound output:
#
#    python3 -m nano_llm.test.tts --verbose \
#	    --tts xtts \
#	    --voice 'Damien Black' \
#	    --sample-rate-hz 44100 \
#       --audio-output-device 25 \
#	    --audio-output-file /data/audio/tts/test.wav
#
# The sample rate should be set to one that the audio output device supports (like 16000, 44100,
# 48000, ect).  This command will list the connected audio devices available:
#
#    python3 -m nano_llm.test.tts --list-audio-devices
#
# The TTS output is automatically resampled to match the sampling rate of the audio device.
#
import sys
import termcolor

from nano_llm.utils import ArgParser, KeyboardInterrupt
from nano_llm.plugins import AutoTTS, UserPrompt, AudioOutputDevice, AudioRecorder, Callback

args = ArgParser(extras=['tts', 'audio_output', 'prompt', 'log']).parse_args()

def print_prompt():
    termcolor.cprint('\n>> ', 'blue', end='', flush=True)

def print_help():
    print(f"Enter text to synthesize, or one of these commands:\n")
    print(f"  /defaults          Generate a default test sequence")
    print(f"  /voices            List the voice names")
    print(f"  /voice Voice Name  Change the voice (current='{tts.voice}')")
    print(f"  /speakers          List the speaker names")
    print(f"  /speaker Speaker   Change the speaker (current='{tts.speaker}')")
    print(f"  /languages         List the languages")
    print(f"  /language en-US    Set the language code (current='{tts.language}')")
    print(f"  /rate 1.0          Set the speaker rate (current={tts.rate:.2f})")
    print(f"  /buffer none       Disable input buffering (current='{','.join(tts.buffering)}')")
    print(f"  /interrupt or /i   Interrupt/mute the TTS output")
    print(f"  /help or /?        Print the help text")  
    print(f"  /quit or /exit     Exit the program\n")
    print(f"Press Ctrl+C to interrupt output, and Ctrl+C twice to exit.")
    print_prompt()

def commands(text):
    try:
        cmd = text.lower().strip()
        if cmd.startswith('/default'):
            if tts.language.startswith('en'):
                tts("Hello there, how are you today? ")
                tts("The weather is 76 degrees out and sunny. ")
                tts("Your first meeting is in an hour downtown, with normal traffic. ")
                tts("Can I interest you in anything quick for breakfast?")
            if tts.language.startswith('zh'):
                tts("菩薩清涼月. 常遊畢竟空. 眾生心垢淨. 菩提月現前. ")
                tts("菩薩清涼月. 遊於畢竟空. 垂光照三界. 心法無不現. ")
                tts("佛法無人說. 雖慧莫能瞭. 譬如暗中寶. 無燈不可見. ")    # 佛法無人說，雖慧莫能了
                tts("若人欲識佛境界. 當淨其意如虛空. 遠離妄想及諸取. 令心所向皆無礙. ")
                tts("猶如蓮華不著水. 亦如日月不著空. 諸惡趣苦願寂靜.  一切群生令安樂. 於諸群生行利益. 乃至十方諸剎土. ")
                tts("旋風偃嶽而常靜. 江河競注而不流. 野馬漂鼓而不動. 日月歷天而不周. ")
                tts("千丈之堤. 以螻蟻之穴潰.  百尺之室. 以突隙之熛焚. ")
                tts("阿陀那識甚深細. 一切種子如瀑流. 我於凡愚不開演. 恐彼分別執為我. ")
                tts("如海遇風緣. 起種種波浪. 現前作用轉. 無有間斷時.  藏識海亦然. 境等風所擊. 恆起諸識浪. 現前作用轉. ")
                tts("彊觀諸法唯是心相. 虛狀無實. 復當觀此能觀之心. 亦無實念. ")
                tts("心本無生因境有. 前境若無心亦無. ")
                tts("不以囊臭而棄其金. 慢如高山雨水不停. 卑如江海萬川歸集. 我以法故復度敬彼. ")
                tts("狐非獅子類. 燈非日月明. 池無巨海納. 丘無嵩嶽榮.  法雲垂世界. 法雨潤群萌. 顯通稀有事. 處處化群生. ")
                tts("去抓藍色巧克力口味, 白色草莓口味, 綠色牛奶口味. ")
                tts("去拿紅色蘋果口味. ", final=True)
        elif cmd.startswith('/voices'):
            print(tts.voices)
        elif cmd.startswith('/voice'):
            tts.voice = text[6:].strip()
        elif cmd.startswith('/speakers'):
            print(tts.speakers)
        elif cmd.startswith('/speaker'):
            tts.speaker = text[8:].strip()
        elif cmd.startswith('/languages'):
            print(tts.languages)
        elif cmd.startswith('/language'):
            tts.language = text[9:].strip()
        elif cmd.startswith('/rate'):
            tts.rate = float(cmd.split(' ')[1])
        elif cmd.startswith('/buffer'):
            tts.buffering = text[7:].strip()
        elif cmd.startswith('/i'):
            on_interrupt()
        elif cmd.startswith('/quit') or cmd.startswith('/exit'):
            sys.exit(0)
        elif cmd.startswith('/h') or cmd.startswith('/?'):
            print_help()
        elif len(cmd.strip()) == 0:
            pass
        else:
            return tts(f"{text} . " )  # send to TTS
    except Exception as error:
        print(f"\nError: {error}")
    print_prompt()
 
def on_interrupt():
    tts.interrupt()
    print_prompt()

tts = AutoTTS.from_pretrained(**vars(args))

interrupt = KeyboardInterrupt(callback=on_interrupt)

if args.audio_output_device is not None:
    tts.add(AudioOutputDevice(**vars(args)))

if args.audio_output_file is not None:
    tts.add(AudioRecorder(**vars(args)))

prompt = UserPrompt(interactive=True, **vars(args)).add(
    Callback(commands).add(tts)
)

print_help()
prompt.start().join()
