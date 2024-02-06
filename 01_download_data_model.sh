#!/bin/sh

for data_set in libri_dev libri_test; do   
    dir=data/$data_set
    if [ ! -f $dir/wav.scp ] ; then
        [ -d $dir ] && rm -r $dir
        if [ ! -f $data_set.tar.gz ]; then
            echo "  You will be prompted to enter password for getdata@voiceprivacychallenge.univ-avignon.fr"
            sftp getdata@voiceprivacychallenge.univ-avignon.fr <<EOF
    cd /challengedata/corpora
    get $data_set.tar.gz
    bye
EOF
  fi
  echo "  Unpacking $data_set data set..."
  tar -xf $data_set.tar.gz || exit 1
  [ ! -f $dir/text ] && echo "File $dir/text does not exist" && exit 1
  cut -d' ' -f1 $dir/text > $dir/text1
  cut -d' ' -f2- $dir/text | sed -r 's/,|!|\?|\./ /g' | sed -r 's/ +/ /g' | awk '{print toupper($0)}' > $dir/text2
  paste -d' ' $dir/text1 $dir/text2 > $dir/text
  rm $dir/text1 $dir/text2
fi

done

#Download LibriSpeech-360
check=corpora/LibriSpeech/train-clean-360
if [ ! -d $check ]; then
    echo "Download train-clean-360..."
    mkdir -p corpora
    cd corpora
    if [ ! -f train-clean-360.tar.gz ] ; then
        echo "Download train-clean-360..."
        wget --no-check-certificate https://www.openslr.org/resources/12/train-clean-360.tar.gz
    fi
    echo "Unpacking train-clean-360"
    tar -xvzf train-clean-360.tar.gz
    cd ../
fi

check_data=data/libri_dev_enrolls
check_model=exp/asv_pre_ecapa
#Download kaldi format datadir and SpeechBrain pretrained ASV/ASR models
if [ ! -d $check_data ]; then
    if  [ ! -f data.zip ]; then
        echo "Download VPC kaldi format datadir..."
        wget https://github.com/DigitalPhonetics/VoicePAT/releases/download/v2/data.zip
    fi
    echo "Unpacking data"
    unzip data.zip
fi

if [ ! -d $check_model ]; then
    if [ ! -f pre_model.zip ]; then
        echo "Download pretrained ASV & ASR models trained using original train-clean-360..."
        wget https://github.com/DigitalPhonetics/VoicePAT/releases/download/v2/pre_model.zip
    fi
    echo "Unpacking pretrained evaluation models"
    unzip pre_model.zip
fi

#Download GAN pre-models only if perform GAN anonymization
if [ ! -d models ]; then
    echo "Download pretrained models of GAN-basd speaker anonymization system, only if you use this method to anonymize data.."
    mkdir -p models
    wget -q -O models/anonymization.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/anonymization.zip
    wget -q -O models/asr.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/asr.zip
    wget -q -O models/tts.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/tts.zip
    unzip -oq models/asr.zip -d models
    unzip -oq models/tts.zip -d models
    unzip -oq models/anonymization.zip -d models
    rm models/*.zip
fi




