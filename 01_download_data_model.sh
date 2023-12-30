#!/bin/sh
<<!
for data_set in libri_dev libri_test vctk_dev vctk_test; do   
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
mkdir -p corpora
cd corpora
wget --no-check-certificate https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xvzf train-clean-360.tar.gz
cd ../
!

#Download kaldi format datadir and SpeechBrain pretrained ASV/ASR models
wget https://github.com/DigitalPhonetics/VoicePAT/releases/download/v2/data.zip
unzip data.zip
wget https://github.com/DigitalPhonetics/VoicePAT/releases/download/v2/pre_model.zip
unzip pre_model.zip
