https://gpushare.com/auth/register
18608916700
Tmp.051220

ssh -p 33819 -L 37201:localhost:37201 root@i-2.gpushare.com
BUxQV5FxQc5CdMuQrZVX6fAt76ynxwqV

wget -c "https://hf-mirror.com/dphn/Dolphin3.0-Llama3.1-8B-GGUF/resolve/main/Dolphin3.0-Llama3.1-8B-Q6_K.gguf?download=true" -O Dolphin3.0-Llama3.1-8B-Q6_K.gguf
wget -c "https://hf-mirror.com/meta-llama/Llama-3.2-1B-Instruct/resolve/main/model.safetensors?download=true" -O model.safetensors
wget -c "https://hf-mirror.com/dphn/Dolphin3.0-Llama3.2-1B/resolve/main/generation_config.json?download=true" -O generation_config.json

~/llama.cpp/build/bin/llama-server -m ~/src/llms/cus2.gguf --port 37201 -c 8192 -ngl 999

wget -c "https://hf-mirror.com/mradermacher/MN-12B-Mag-Mell-R1-Uncensored-Scale1.2-i1-GGUF/resolve/main/MN-12B-Mag-Mell-R1-Uncensored-Scale1.2.i1-Q4_K_M.gguf?download=true" -O cus2.gguf

Heretic-Dolphin3.0-Qwen2.5-1.5B.i1-Q4_K_M.gguf
Dolphin3.0-Llama3.1-1B-abliterated.Q4_K_M.gguf
Mistral-Small-3.1-24B-Instruct-2503-MAX-NEO-D_AU-Q5_K_M-imat.gguf
