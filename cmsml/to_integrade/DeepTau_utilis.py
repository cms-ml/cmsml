#import uproot
#import pandas
import numpy as np
import tensorflow as tf

truth_branches = [ 'gen_e', 'gen_mu', 'gen_tau', 'gen_jet' ]
weight_branches = [ 'trainingWeight' ]
navigation_branches = [ 'innerCells_begin', 'innerCells_end', 'outerCells_begin', 'outerCells_end' ]
tau_id_branches = [ 'againstElectronMVA6', 'againstElectronMVA6raw', 'againstElectronMVA62018',
                    'againstElectronMVA62018raw', 'againstMuon3', 'againstMuon3raw',
                    'byCombinedIsolationDeltaBetaCorr3Hits', 'byCombinedIsolationDeltaBetaCorr3Hitsraw',
                    'byIsolationMVArun2v1DBoldDMwLT2016', 'byIsolationMVArun2v1DBoldDMwLT2016raw',
                    'byIsolationMVArun2v1DBnewDMwLT2016', 'byIsolationMVArun2v1DBnewDMwLT2016raw',
                    'byIsolationMVArun2017v2DBoldDMwLT2017', 'byIsolationMVArun2017v2DBoldDMwLT2017raw',
                    'byIsolationMVArun2017v2DBoldDMdR0p3wLT2017', 'byIsolationMVArun2017v2DBoldDMdR0p3wLT2017raw',
                    'byIsolationMVArun2017v2DBnewDMwLT2017', 'byIsolationMVArun2017v2DBnewDMwLT2017raw',
                    'byDeepTau2017v1VSe', 'byDeepTau2017v1VSeraw', 'byDeepTau2017v1VSmu', 'byDeepTau2017v1VSmuraw',
                    'byDeepTau2017v1VSjet', 'byDeepTau2017v1VSjetraw', 'byDpfTau2016v0VSall', 'byDpfTau2016v0VSallraw' ]
input_event_branches = [ 'rho' ]
input_tau_branches = [ 'tau_pt', 'tau_eta', 'tau_phi', 'tau_mass', 'tau_E_over_pt', 'tau_charge',
                       'tau_n_charged_prongs', 'tau_n_neutral_prongs', 'chargedIsoPtSum',
                       'chargedIsoPtSumdR03_over_dR05', 'footprintCorrection', 'neutralIsoPtSum',
                       'neutralIsoPtSumWeight_over_neutralIsoPtSum', 'neutralIsoPtSumWeightdR03_over_neutralIsoPtSum',
                       'neutralIsoPtSumdR03_over_dR05', 'photonPtSumOutsideSignalCone', 'puCorrPtSum',
                       'tau_dxy_pca_x', 'tau_dxy_pca_y', 'tau_dxy_pca_z', 'tau_dxy_valid', 'tau_dxy', 'tau_dxy_sig',
                       'tau_ip3d_valid', 'tau_ip3d', 'tau_ip3d_sig', 'tau_dz', 'tau_dz_sig_valid', 'tau_dz_sig',
                       'tau_flightLength_x', 'tau_flightLength_y', 'tau_flightLength_z', 'tau_flightLength_sig',
                       'tau_pt_weighted_deta_strip', 'tau_pt_weighted_dphi_strip', 'tau_pt_weighted_dr_signal',
                       'tau_pt_weighted_dr_iso', 'tau_leadingTrackNormChi2', 'tau_e_ratio_valid', 'tau_e_ratio',
                       'tau_gj_angle_diff_valid', 'tau_gj_angle_diff', 'tau_n_photons', 'tau_emFraction',
                       'tau_inside_ecal_crack', 'leadChargedCand_etaAtEcalEntrance_minus_tau_eta' ]

input_cell_external_branches = [ 'rho', 'tau_pt', 'tau_eta', 'tau_inside_ecal_crack' ]

cell_index_branches = [ 'eta_index', 'phi_index' ]
input_cell_pfCand_ele_branches = [ 'pfCand_ele_valid', 'pfCand_ele_rel_pt', 'pfCand_ele_deta', 'pfCand_ele_dphi',
                                   'pfCand_ele_pvAssociationQuality', 'pfCand_ele_puppiWeight', 'pfCand_ele_charge',
                                   'pfCand_ele_lostInnerHits', 'pfCand_ele_numberOfPixelHits', 'pfCand_ele_vertex_dx',
                                   'pfCand_ele_vertex_dy', 'pfCand_ele_vertex_dz', 'pfCand_ele_vertex_dx_tauFL',
                                   'pfCand_ele_vertex_dy_tauFL', 'pfCand_ele_vertex_dz_tauFL',
                                   'pfCand_ele_hasTrackDetails', 'pfCand_ele_dxy', 'pfCand_ele_dxy_sig',
                                   'pfCand_ele_dz', 'pfCand_ele_dz_sig', 'pfCand_ele_track_chi2_ndof',
                                   'pfCand_ele_track_ndof' ]
input_cell_pfCand_muon_branches = [ 'pfCand_muon_valid', 'pfCand_muon_rel_pt', 'pfCand_muon_deta', 'pfCand_muon_dphi',
                                    'pfCand_muon_pvAssociationQuality', 'pfCand_muon_fromPV',
                                    'pfCand_muon_puppiWeight', 'pfCand_muon_charge', 'pfCand_muon_lostInnerHits',
                                    'pfCand_muon_numberOfPixelHits', 'pfCand_muon_vertex_dx', 'pfCand_muon_vertex_dy',
                                    'pfCand_muon_vertex_dz', 'pfCand_muon_vertex_dx_tauFL',
                                    'pfCand_muon_vertex_dy_tauFL', 'pfCand_muon_vertex_dz_tauFL',
                                    'pfCand_muon_hasTrackDetails', 'pfCand_muon_dxy', 'pfCand_muon_dxy_sig',
                                    'pfCand_muon_dz', 'pfCand_muon_dz_sig', 'pfCand_muon_track_chi2_ndof',
                                    'pfCand_muon_track_ndof' ]
input_cell_pfCand_chHad_branches = [ 'pfCand_chHad_valid', 'pfCand_chHad_rel_pt', 'pfCand_chHad_deta',
                                     'pfCand_chHad_dphi', 'pfCand_chHad_leadChargedHadrCand',
                                     'pfCand_chHad_pvAssociationQuality', 'pfCand_chHad_fromPV',
                                     'pfCand_chHad_puppiWeight', 'pfCand_chHad_puppiWeightNoLep',
                                     'pfCand_chHad_charge', 'pfCand_chHad_lostInnerHits',
                                     'pfCand_chHad_numberOfPixelHits', 'pfCand_chHad_vertex_dx',
                                     'pfCand_chHad_vertex_dy', 'pfCand_chHad_vertex_dz',
                                     'pfCand_chHad_vertex_dx_tauFL', 'pfCand_chHad_vertex_dy_tauFL',
                                     'pfCand_chHad_vertex_dz_tauFL', 'pfCand_chHad_hasTrackDetails',
                                     'pfCand_chHad_dxy', 'pfCand_chHad_dxy_sig', 'pfCand_chHad_dz',
                                     'pfCand_chHad_dz_sig', 'pfCand_chHad_track_chi2_ndof', 'pfCand_chHad_track_ndof',
                                     'pfCand_chHad_hcalFraction', 'pfCand_chHad_rawCaloFraction' ]
input_cell_pfCand_nHad_branches = [ 'pfCand_nHad_valid', 'pfCand_nHad_rel_pt', 'pfCand_nHad_deta', 'pfCand_nHad_dphi',
                                    'pfCand_nHad_puppiWeight', 'pfCand_nHad_puppiWeightNoLep',
                                    'pfCand_nHad_hcalFraction' ]
input_cell_pfCand_gamma_branches = [ 'pfCand_gamma_valid', 'pfCand_gamma_rel_pt', 'pfCand_gamma_deta',
                                     'pfCand_gamma_dphi', 'pfCand_gamma_pvAssociationQuality', 'pfCand_gamma_fromPV',
                                     'pfCand_gamma_puppiWeight', 'pfCand_gamma_puppiWeightNoLep',
                                     'pfCand_gamma_lostInnerHits', 'pfCand_gamma_numberOfPixelHits',
                                     'pfCand_gamma_vertex_dx', 'pfCand_gamma_vertex_dy', 'pfCand_gamma_vertex_dz',
                                     'pfCand_gamma_vertex_dx_tauFL', 'pfCand_gamma_vertex_dy_tauFL',
                                     'pfCand_gamma_vertex_dz_tauFL', 'pfCand_gamma_hasTrackDetails',
                                     'pfCand_gamma_dxy', 'pfCand_gamma_dxy_sig', 'pfCand_gamma_dz',
                                     'pfCand_gamma_dz_sig', 'pfCand_gamma_track_chi2_ndof', 'pfCand_gamma_track_ndof' ]
input_cell_ele_branches = [ 'ele_valid', 'ele_rel_pt', 'ele_deta', 'ele_dphi', 'ele_cc_valid', 'ele_cc_ele_rel_energy',
                            'ele_cc_gamma_rel_energy', 'ele_cc_n_gamma', 'ele_rel_trackMomentumAtVtx',
                            'ele_rel_trackMomentumAtCalo', 'ele_rel_trackMomentumOut',
                            'ele_rel_trackMomentumAtEleClus', 'ele_rel_trackMomentumAtVtxWithConstraint',
                            'ele_rel_ecalEnergy', 'ele_ecalEnergy_sig', 'ele_eSuperClusterOverP',
                            'ele_eSeedClusterOverP', 'ele_eSeedClusterOverPout', 'ele_eEleClusterOverPout',
                            'ele_deltaEtaSuperClusterTrackAtVtx', 'ele_deltaEtaSeedClusterTrackAtCalo',
                            'ele_deltaEtaEleClusterTrackAtCalo', 'ele_deltaPhiEleClusterTrackAtCalo',
                            'ele_deltaPhiSuperClusterTrackAtVtx', 'ele_deltaPhiSeedClusterTrackAtCalo',
                            'ele_mvaInput_earlyBrem', 'ele_mvaInput_lateBrem', 'ele_mvaInput_sigmaEtaEta',
                            'ele_mvaInput_hadEnergy', 'ele_mvaInput_deltaEta', 'ele_gsfTrack_normalizedChi2',
                            'ele_gsfTrack_numberOfValidHits', 'ele_rel_gsfTrack_pt', 'ele_gsfTrack_pt_sig',
                            'ele_has_closestCtfTrack', 'ele_closestCtfTrack_normalizedChi2',
                            'ele_closestCtfTrack_numberOfValidHits' ]
input_cell_muon_branches = [ 'muon_valid', 'muon_rel_pt', 'muon_deta', 'muon_dphi', 'muon_dxy', 'muon_dxy_sig',
                             'muon_normalizedChi2_valid', 'muon_normalizedChi2', 'muon_numberOfValidHits',
                             'muon_segmentCompatibility', 'muon_caloCompatibility', 'muon_pfEcalEnergy_valid',
                             'muon_rel_pfEcalEnergy', 'muon_n_matches_DT_1', 'muon_n_matches_DT_2',
                             'muon_n_matches_DT_3', 'muon_n_matches_DT_4', 'muon_n_matches_CSC_1',
                             'muon_n_matches_CSC_2', 'muon_n_matches_CSC_3', 'muon_n_matches_CSC_4',
                             'muon_n_matches_RPC_1', 'muon_n_matches_RPC_2', 'muon_n_matches_RPC_3',
                             'muon_n_matches_RPC_4', 'muon_n_hits_DT_1', 'muon_n_hits_DT_2', 'muon_n_hits_DT_3',
                             'muon_n_hits_DT_4', 'muon_n_hits_CSC_1', 'muon_n_hits_CSC_2', 'muon_n_hits_CSC_3',
                             'muon_n_hits_CSC_4', 'muon_n_hits_RPC_1', 'muon_n_hits_RPC_2', 'muon_n_hits_RPC_3',
                             'muon_n_hits_RPC_4' ]

df_tau_branches = truth_branches + weight_branches + navigation_branches + input_event_branches + input_tau_branches
df_cell_branches = cell_index_branches + input_cell_pfCand_ele_branches + input_cell_pfCand_muon_branches + \
                   input_cell_pfCand_chHad_branches + input_cell_pfCand_nHad_branches + \
                   input_cell_pfCand_gamma_branches + input_cell_ele_branches + input_cell_muon_branches

match_suffixes = [ 'e', 'mu', 'tau', 'jet' ]
e, mu, tau, jet = 0, 1, 2, 3
#cell_locations = ['inner', 'outer']
n_cells_eta = { 'inner': 11, 'outer': 21 }
n_cells_phi = { 'inner': 11, 'outer': 21 }
n_cells = { 'inner': n_cells_eta['inner'] * n_cells_phi['inner'], 'outer': n_cells_eta['outer'] * n_cells_phi['outer'] }

class NetConf:
    def __init__(self, name, final, tau_branches, cell_locations, component_names, component_branches):
        self.name = name
        self.final = final
        self.tau_branches = tau_branches
        self.cell_locations = cell_locations
        self.comp_names = component_names
        self.comp_branches = component_branches

netConf_preTau = NetConf("preTau", False, input_event_branches + input_tau_branches, [], [], [])
netConf_preInner = NetConf("preInner", False, [], ['inner'], ['egamma', 'muon', 'hadrons'], [
    input_cell_pfCand_ele_branches + input_cell_ele_branches + input_cell_pfCand_gamma_branches,
    input_cell_pfCand_muon_branches + input_cell_muon_branches,
    input_cell_pfCand_chHad_branches + input_cell_pfCand_nHad_branches
])
netConf_preTauInter = NetConf("preTauInter", False, netConf_preTau.tau_branches, netConf_preInner.cell_locations,
                              netConf_preInner.comp_names, netConf_preInner.comp_branches)
netConf_preOuter = NetConf("preOuter", False, [], ['outer'], netConf_preInner.comp_names,
                           netConf_preInner.comp_branches)
netConf_full = NetConf("full", True, netConf_preTau.tau_branches,
                       netConf_preInner.cell_locations + netConf_preOuter.cell_locations,
                       netConf_preInner.comp_names, netConf_preInner.comp_branches)

netConf_full_cmb = NetConf("full_cmb", True, input_tau_branches,
                           netConf_preInner.cell_locations + netConf_preOuter.cell_locations, ['cmb'],
                           [ input_cell_pfCand_ele_branches + input_cell_pfCand_muon_branches + \
                             input_cell_pfCand_chHad_branches + input_cell_pfCand_nHad_branches + \
                             input_cell_pfCand_gamma_branches + input_cell_ele_branches + input_cell_muon_branches ])

# component_names = [ 'cmb' ]
# component_branches = [
#     input_cell_pfCand_ele_branches + input_cell_pfCand_muon_branches + input_cell_pfCand_chHad_branches + \
#     input_cell_pfCand_nHad_branches + input_cell_pfCand_gamma_branches + input_cell_ele_branches + \
#     input_cell_muon_branches,
# ]

n_outputs = len(truth_branches)

def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="deepTau")
    return graph

class TauLosses:
    Le_sf = 1
    Lmu_sf = 1
    Ltau_sf = 1
    Ljet_sf = 1
    epsilon = 1e-7
    merge_thr = 0.1

    @staticmethod
    def SetSFs(sf_e, sf_mu, sf_tau, sf_jet):
        sf_corr = 4. / (sf_e + sf_mu + sf_tau + sf_jet)
        TauLosses.Le_sf = sf_e * sf_corr
        TauLosses.Lmu_sf = sf_mu * sf_corr
        TauLosses.Ltau_sf = sf_tau * sf_corr
        TauLosses.Ljet_sf = sf_jet * sf_corr

    @staticmethod
    def Lbase(target, output, genuine_index, fake_index):
        epsilon = tf.constant(TauLosses.epsilon, output.dtype.base_dtype)
        genuine_vs_fake = output[:, genuine_index] / (output[:, genuine_index] + output[:, fake_index] + epsilon)
        genuine_vs_fake = tf.clip_by_value(genuine_vs_fake, epsilon, 1 - epsilon)
        loss = -target[:, genuine_index] * tf.log(genuine_vs_fake) - target[:, fake_index] * tf.log(1 - genuine_vs_fake)
        return loss

    @staticmethod
    def Hbase(target, output, index, inverse):
        epsilon = tf.constant(TauLosses.epsilon, output.dtype.base_dtype)
        x = tf.clip_by_value(output[:, index], epsilon, 1 - epsilon)
        if inverse:
            return - (1 - target[:, index]) * tf.log(1 - x)
        return - target[:, index] * tf.log(x)

    @staticmethod
    def Fbase(target, output, index, gamma, apply_decay, inverse):
        decay_factor = (tf.math.tanh(70 * (output[:, tau] - 0.1)) + 1) / 2 if apply_decay else 1
        if gamma <= 0:
            raise RuntimeError("Focal Loss requires gamma > 0.")
        epsilon = tf.constant(TauLosses.epsilon, output.dtype.base_dtype)
        gamma_t = tf.constant(gamma, output.dtype.base_dtype)
        x = tf.clip_by_value(output[:, index], epsilon, 1 - epsilon)
        if inverse:
            return - decay_factor * (1 - target[:, index]) * tf.pow(x, gamma_t) * tf.log(1 - x)
        return - decay_factor * target[:, index] * tf.pow(1-x, gamma_t) * tf.log(x)

    @staticmethod
    def Le(target, output):
        return TauLosses.Lbase(target, output, tau, e)

    @staticmethod
    def Lmu(target, output):
        return TauLosses.Lbase(target, output, tau, mu)

    @staticmethod
    def Ljet(target, output):
        return TauLosses.Lbase(target, output, tau, jet)

    @staticmethod
    def sLe(target, output):
        sf = tf.constant(TauLosses.Le_sf, output.dtype.base_dtype)
        return sf * TauLosses.Le(target, output)

    @staticmethod
    def sLmu(target, output):
        sf = tf.constant(TauLosses.Lmu_sf, output.dtype.base_dtype)
        return sf * TauLosses.Lmu(target, output)

    @staticmethod
    def sLjet(target, output):
        sf = tf.constant(TauLosses.Ljet_sf, output.dtype.base_dtype)
        return sf * TauLosses.Ljet(target, output)

    @staticmethod
    def He(target, output):
        return TauLosses.Hbase(target, output, e, False)

    @staticmethod
    def Hmu(target, output):
        return TauLosses.Hbase(target, output, mu, False)

    @staticmethod
    def Htau(target, output):
        return TauLosses.Hbase(target, output, tau, False)

    @staticmethod
    def Hjet(target, output):
        return TauLosses.Hbase(target, output, jet, False)

    @staticmethod
    def Hcat_base(target, output, index, inverse):
        decay_factor = (tf.math.tanh(70 * (output[:, tau] - 0.1)) + 1) / 2
        if inverse:
            decay_factor = 1 - decay_factor
        return decay_factor * TauLosses.Hbase(target, output, index, False)

    @staticmethod
    def Hcat_e(target, output):
        return TauLosses.Hcat_base(target, output, e, False)

    @staticmethod
    def Hcat_mu(target, output):
        return TauLosses.Hcat_base(target, output, mu, False)

    @staticmethod
    def Hcat_jet(target, output):
        return TauLosses.Hcat_base(target, output, jet, False)

    @staticmethod
    def Hcat_eInv(target, output):
        return TauLosses.Hcat_base(target, output, e, True)

    @staticmethod
    def Hcat_muInv(target, output):
        return TauLosses.Hcat_base(target, output, mu, True)

    @staticmethod
    def Hcat_jetInv(target, output):
        return TauLosses.Hcat_base(target, output, jet, True)

    @staticmethod
    def Hbin(target, output):
        return TauLosses.Hbase(target, output, tau, True)

    @staticmethod
    def Fe(target, output):
        F_factor = tf.constant(1.63636, dtype=output.dtype.base_dtype)
        return F_factor * TauLosses.Fbase(target, output, e, 2, True, False)

    @staticmethod
    def Fmu(target, output):
        F_factor = tf.constant(1.63636, dtype=output.dtype.base_dtype)
        return F_factor * TauLosses.Fbase(target, output, mu, 2, True, False)

    @staticmethod
    def Fjet(target, output):
        F_factor = tf.constant(1.63636, dtype=output.dtype.base_dtype)
        return F_factor * TauLosses.Fbase(target, output, jet, 2, True, False)

    @staticmethod
    def Fcmb(target, output):
        F_factor = tf.constant(1.17153, dtype=output.dtype.base_dtype)
        return F_factor * TauLosses.Fbase(target, output, tau, 0.5, False, True)

    @staticmethod
    def tau_crossentropy(target, output):
        return TauLosses.sLe(target, output) + TauLosses.sLmu(target, output) + TauLosses.sLjet(target, output)

    @staticmethod
    def tau_crossentropy_v2(target, output):
        F_factor = tf.constant(5, dtype=output.dtype.base_dtype)
        sf = tf.constant([TauLosses.Le_sf, TauLosses.Lmu_sf, TauLosses.Ltau_sf, TauLosses.Ljet_sf],
                         dtype=output.dtype.base_dtype)
        return sf[tau] * TauLosses.Htau(target, output) + (sf[e] + sf[mu] + sf[jet]) * TauLosses.Fcmb(target, output) \
               + F_factor * (sf[e] * TauLosses.Fe(target, output) + sf[mu] * TauLosses.Fmu(target, output) \
                             + sf[jet] * TauLosses.Fjet(target, output))

    @staticmethod
    def tau_vs_other(prob_tau, prob_other):
        #return np.where(prob_tau > TauLosses.merge_thr, prob_tau / (prob_tau + prob_other), prob_tau)
        #return np.where(prob_tau > TauLosses.merge_thr, prob_tau / np.exp(prob_other), prob_tau)
        return prob_tau / (prob_tau + prob_other + TauLosses.epsilon)


def LoadModel(model_file, compile=True):
    from keras.models import load_model
    if compile:
        return load_model(model_file, custom_objects = {
            'tau_crossentropy': TauLosses.tau_crossentropy, 'tau_crossentropy_v2': TauLosses.tau_crossentropy_v2,
            'Le': TauLosses.Le, 'Lmu': TauLosses.Lmu, 'Ljet': TauLosses.Ljet,
            'sLe': TauLosses.sLe, 'sLmu': TauLosses.sLmu, 'sLjet': TauLosses.sLjet,
            'He': TauLosses.He, 'Hmu': TauLosses.Hmu, 'Htau': TauLosses.Htau, 'Hjet': TauLosses.Hjet,
            'Hcat_e': TauLosses.Hcat_e, 'Hcat_mu': TauLosses.Hcat_mu, 'Hcat_jet': TauLosses.Hcat_jet,
            'Hcat_eInv': TauLosses.Hcat_eInv, 'Hcat_muInv': TauLosses.Hcat_muInv, 'Hcat_jetInv': TauLosses.Hcat_jetInv,
            'Hbin': TauLosses.Hbin, 'HbinInv': TauLosses.Hbin,
            'Fe': TauLosses.Fe, 'Fmu': TauLosses.Fmu, 'Fjet': TauLosses.Fjet, 'Fcmb': TauLosses.Fcmb
        })
    else:
        return load_model(model_file, compile = False)


def quantile_ex(data, quantiles, weights):
    quantiles = np.array(quantiles)
    indices = np.argsort(data)
    data_sorted = data[indices]
    weights_sorted = weights[indices]
    prob = np.cumsum(weights_sorted) - weights_sorted / 2
    prob = (prob[:] - prob[0]) / (prob[-1] - prob[0])
    return np.interp(quantiles, prob, data_sorted)