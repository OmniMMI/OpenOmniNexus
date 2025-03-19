# Audio Instruction Synthesis

1. **Audio Prompt Preparation**

   Download the audio from the [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K).

2. **Prepare the TTS Tool**

   Option 1: [ChatTTS](https://github.com/2noise/ChatTTS)

   Option 2: [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (recommend)

3. **Randomly Select the Audio Prompt and Synthesize the Audio Instruction**

   *Note: Before running the script, check the directories in the script.*

   ```bash
   process_cosyvoice.sh
   ```