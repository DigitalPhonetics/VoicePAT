###############################
## CONFIGURATION
###############################
PHONY: install uninstall pretrained_models
.ONESHELL:

PROJECT_NAME = voicepat
ENV_NAME = $(PROJECT_NAME)_env

ifeq (, $(shell mamba --version))
CONDA = conda
else
CONDA = mamba
endif

###############################
##@ INSTALLATION
###############################

install: $(ENV_NAME) ## performs the installation. Currently the only step is to install the conda environment

uninstall:
	@rm -rf $(ENV_NAME)
	@rm -rf models/

pretrained_models: ## downloads the pretrained models from IMS repositories
	@echo Downloading models from IMS repositories
	@rm -rf models
	@mkdir -p models
	@wget -q -O models/anonymization.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/anonymization.zip
	@wget -q -O models/asr.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/asr.zip
	@wget -q -O models/tts.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/tts.zip
	@wget -q -O models/pre_eval_models.zip https://github.com/DigitalPhonetics/VoicePAT/releases/download/v1/pre_eval_models.zip
	@unzip -oq models/asr.zip -d models
	@unzip -oq models/tts.zip -d models
	@unzip -oq models/anonymization.zip -d models
	@mkdir evaluation/utility/asr/exp
	@unzip -oq models/pre_eval_models.zip -d evaluation/utility/asr/exp
	@ln -srf evaluation/utility/asr/exp exp
	@rm models/*.zip


$(ENV_NAME): environment.yaml
	@($(CONDA) env create -f $< -p ./$@ && echo Installation complete, please run `conda develop .` once.) || $(CONDA) env update -f $< -p ./$@
	@conda config --set env_prompt '($$(basename {default_env})) '
	@(cat .gitignore | grep -q $(ENV_NAME)) || echo $(ENV_NAME) >> .gitignore

###############################
##@ SELF-DOCUMENTING COMMAND
###############################

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
