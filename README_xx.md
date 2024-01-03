```
git clone -b vpc  https://github.com/DigitalPhonetics/VoicePAT.git
bash 00_install.sh
bash 01_download_data_model.sh # kaldi format data
bash 02_run.sh  #generate B2 and pre/post evaluation
bash 03_gen_gan.sh #generate GAN anonymized test/dev data 
```
