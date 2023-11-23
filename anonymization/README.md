# Anonymization

The anonymization branch can contain multiple pipelines, modules and models. So far, the only pipeline added is the 
[Speech-to-Text-to-Speech (STTTS) pipeline](https://ieeexplore.ieee.org/document/10096607), based on this code:  
[https://github.com/DigitalPhonetics/speaker-anonymization](https://github.com/DigitalPhonetics/speaker-anonymization).


# Experiment with different speaker embedding mappings

This is now simplified: you can define your anonymizer (a function that yields a speaker embedding when a speaker embedding is supplied) using the `!new` syntax of HyperPyYAML in a config file (e.g., see [ims_gan.yaml](../configs/anon/ims_gan.yaml)). The only requirement is that your anonymizer must implement the `BaseAnonymizer` API (see [base_anon.py](modules/speaker_embeddings/anonymization/base_anon.py)).

*This documentation is still under construction and will be extended soon.*