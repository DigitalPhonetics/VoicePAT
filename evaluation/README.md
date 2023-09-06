# Evaluation
The evaluation scripts are based on external tools (the VoicePrivacy Challenge 2022, SpeechBrain, ESPnet) but have 
been modified to fit into this framework and include the proposed improvements.

## Privacy

### ASV
All scripts regarding the ASV training in [privacy/asv/asv_train](privacy/asv/asv_train), including parts of [privacy/asv/asv.py](privacy/asv/asv.py), are based on the 
[VoxCeleb recipe by SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb) and adapted for LibriSpeech and this 
framework. 

The additional metrics in [privacy/asv/metrics](privacy/asv/metrics) are based on scripts in the [framework for the VoicePrivacy Challenge 2022](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022). They were included from other 
sources, i.e.,
* ZEBRA: [https://gitlab.eurecom.fr/nautsch/zebra](https://gitlab.eurecom.fr/nautsch/zebra)
* Cllr and linkability: [https://gitlab.inria.fr/magnet/anonymization_metrics](https://gitlab.inria.fr/magnet/anonymization_metrics)


## Utility

### ASR
The ASR scripts in [utility/asr](utility/asr) are all based on the [LibriSpeech ASR1 recipe by ESPnet](https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1). Some scripts were copied from that source 
verbatim because they are not part of the ESPnet Python library. 

### Voice Distinctiveness
The scripts for computing Gain of Voice Distinctiveness (GVD) in [utility/voice_distinctiveness](utility/voice_distinctiveness) are based on the [computation of similarity matrices in the VoicePrivacy Challenge 2022](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022/tree/master/baseline/local/similarity_matrices).

*This documentation is still under construction and will be extended soon.*