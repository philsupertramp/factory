#!/bin/bash
#
#
set -e

(

  cd models/



  echo "Downloading encoder model..."
  mkdir -p image_encoder
  wget -O image_encoder/config.json https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/config.json?download=true
  wget -O image_encoder/model.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true

  echo "Downloading ip-adapter models..."
  mkdir -p ip-adapter
  wget -O ip-adapter/ip-adapter_sd15.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin?download=true
  wget -O ip-adapter/ip-adapter-plus_sd15.bin https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin?download=true
  wget -O ip-adapter/ip-adapter-faceid-portrait_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sd15.bin?download=true
  wget -O ip-adapter/ip-adapter-faceid-plusv2_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin?download=true


  echo "Downloading tts voice samples..."
  mkdir -p tts
  wget -O tts/cmu_us_awb_arctic-wav-arctic_a0002.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_awb_arctic-wav-arctic_a0002.npy?download=true
  wget -O tts/cmu_us_bdl_arctic-wav-arctic_a0009.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_bdl_arctic-wav-arctic_a0009.npy?download=true
  wget -O tts/cmu_us_clb_arctic-wav-arctic_a0144.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_clb_arctic-wav-arctic_a0144.npy?download=true 
  wget -O tts/cmu_us_rms_arctic-wav-arctic_b0353.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_rms_arctic-wav-arctic_b0353.npy?download=true
  wget -O tts/cmu_us_slt_arctic-wav-arctic_a0508.npy https://huggingface.co/spaces/Matthijs/speecht5-tts-demo/resolve/main/spkemb/cmu_us_slt_arctic-wav-arctic_a0508.npy?download=true

  echo "Downloading phi-3 t2t models..."
  mkdir -p phi-3-mini-128k-chat

  file_names=(
    "added_tokens.json"
    "config.json"
    "configuration_phi3.py"
    "genai_config.json"
    "phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"
    "phi3-mini-128k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer.model"
    "tokenizer_config.json"
  )

  for file_name in "${file_names[@]}"
  do
    # check if file exists
    if [ -f "phi-3-mini-128k-chat/$file_name" ]; then
      echo "phi-3-mini-128k-chat/$file_name exists, skipping..."
      continue
    fi
    wget -O phi-3-mini-128k-chat/$file_name https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/$file_name
  done

  echo "Fetching Kokoro files..."

  vendor_file_names=(
    "config.json"
  )

  for file_name in "${vendor_file_names[@]}"
  do
    wget -O ../factory/vendors/kokoro/$file_name https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/$file_name
  done

  file_names=(
    "kokoro-v0_19.pth"
    "voices/af.pt"
    "voices/af_bella.pt"
    "voices/af_nicole.pt"
    "voices/af_sarah.pt"
    "voices/af_sky.pt"
    "voices/am_adam.pt"
    "voices/am_michael.pt"
    "voices/bf_emma.pt"
    "voices/bf_isabella.pt"
    "voices/bm_george.pt"
    "voices/bm_lewis.pt"
    )
  for file_name in "${file_names[@]}"
  do
    wget -O ./tts/kokoro/$file_name https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/$file_name
  done
)
