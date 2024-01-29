import numpy as np
import pandas as pd

from .deid_gvd import VoiceDistinctiveness
from utils import find_asv_model_checkpoint


def evaluate_gvd(eval_datasets, eval_data_dir, params, device, anon_data_suffix):

    def get_similarity_matrix(vd_model, out_dir, exp_name, segments_folder):
        if (out_dir / f'similarity_matrix_{exp_name}.npy').exists() and not recompute:
            return np.load(out_dir / f'similarity_matrix_{exp_name}.npy')
        else:
            return vd_model.compute_voice_similarity_matrix(segments_dir_x=segments_folder,
                                                            segments_dir_y=segments_folder, exp_name=exp_name,
                                                            out_dir=out_dir)

    num_utt_per_spk = params['num_utt']
    gvd_tag = f'gvd-{num_utt_per_spk}' if isinstance(num_utt_per_spk, int) else 'gvd'
    save_dir = params['results_dir'] / f'{params["asv_params"]["evaluation"]["distance"]}_out' / gvd_tag
    recompute = params.get('recompute', False)

    vd_settings = {
        'device': device,
        'plda_settings': params['asv_params']['evaluation']['plda'],
        'distance': params['asv_params']['evaluation']['distance'],
        'vec_type': params['asv_params']['vec_type'],
        'num_per_spk': num_utt_per_spk
    }

    if 'model_dir' in params['asv_params']:  # use the same ASV model for original and anon speaker space
        spk_ext_model_dir = find_asv_model_checkpoint(params['asv_params']['model_dir'])
        vd = VoiceDistinctiveness(spk_ext_model_dir=spk_ext_model_dir, score_save_dir=save_dir,
                                  **vd_settings)
        vd_orig, vd_anon = None, None
        save_dir_orig, save_dir_anon = None, None
        print(f'Use ASV model {spk_ext_model_dir} for computing voice similarities of original and anonymized speakers')
    elif 'orig_model_dir' in params['asv_params'] and 'anon_model_dir' in params['asv_params']:
        # use different ASV models for original and anon speaker spaces
        spk_ext_model_dir_orig = find_asv_model_checkpoint(params['asv_params']['orig_model_dir'])
        save_dir_orig = params['orig_results_dir'] / f'{params["asv_params"]["evaluation"]["distance"]}_out' / gvd_tag
        vd_orig = VoiceDistinctiveness(spk_ext_model_dir=spk_ext_model_dir_orig, score_save_dir=save_dir_orig,
                                       **vd_settings)
        spk_ext_model_dir_anon = find_asv_model_checkpoint(params['asv_params']['anon_model_dir'])
        save_dir_anon = params['anon_results_dir'] / f'{params["asv_params"]["evaluation"]["distance"]}_out' / gvd_tag
        vd_anon = VoiceDistinctiveness(spk_ext_model_dir=spk_ext_model_dir_anon,  score_save_dir=save_dir_anon,
                                       **vd_settings)
        vd = None
        print(f'Use ASV model {spk_ext_model_dir_orig} for computing voice similarities of original speakers and ASV '
              f'model {spk_ext_model_dir_anon} for voice similarities of anonymized speakers')
    else:
        raise ValueError('GVD: You either need to specify one "model_dir" for both original and anonymized data or '
                         'one "orig_model_dir" and one "anon_model_dir"!')

    results = []
    for _, trial in eval_datasets:
        osp_set_folder = eval_data_dir / trial
        psp_set_folder = eval_data_dir / f'{trial}_{anon_data_suffix}'
        trial_out_dir = save_dir / trial
        trial_out_dir.mkdir(exist_ok=True, parents=True)

        if vd_orig:
            orig_trial_out_dir = save_dir_orig / trial
            orig_trial_out_dir.mkdir(exist_ok=True, parents=True)
            oo_sim = get_similarity_matrix(vd_model=vd_orig, out_dir=orig_trial_out_dir, exp_name='osp_osp',
                                           segments_folder=osp_set_folder)
        else:
            oo_sim = get_similarity_matrix(vd_model=vd, out_dir=trial_out_dir, exp_name='osp_osp',
                                           segments_folder=osp_set_folder)
        if vd_anon:
            anon_trial_out_dir = save_dir_anon / trial
            anon_trial_out_dir.mkdir(exist_ok=True, parents=True)
            pp_sim = get_similarity_matrix(vd_model=vd_anon, out_dir=anon_trial_out_dir, exp_name='psp_psp',
                                           segments_folder=psp_set_folder)
        else:
            pp_sim = get_similarity_matrix(vd_model=vd, out_dir=trial_out_dir, exp_name='psp_psp',
                                           segments_folder=psp_set_folder)

        gvd_value = vd.gvd(oo_sim, pp_sim) if vd else vd_orig.gvd(oo_sim, pp_sim)
        with open(trial_out_dir / 'gain_of_voice_distinctiveness', 'w') as f:
            f.write(str(gvd_value))
        print(f'{trial} gvd={gvd_value}')
        trial_info = trial.split('_')
        gender = trial_info[3]
        if 'common' in trial:
            gender += '_common'
        results.append({'dataset': trial_info[0], 'split': trial_info[1], 'gender': gender,
                         'GVD': round(gvd_value, 3)})
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(save_dir / 'results.csv')
    return results_df

