# ChanEst-Dataset-Generation
A Reproducible and Reconfigurable Dataset Generation Framework for Deep Learning–Based 6G Channel Estimation.

ChanEst is a reproducible and reconfigurable dataset for deep learning–based channel estimation generated using standard-compliant 3GPP procedures in MATLAB (6G Exploration Library). Each sample is created by inserting PDSCH DM-RS pilots into an OFDM resource grid, transmitting through a randomized 3GPP TDL channel realization, adding AWGN at a randomized SNR, and applying a receiver-aligned pipeline that performs pilot extraction, LS estimation on DM-RS, and 2-D time–frequency interpolation to form a dense-grid learning input. The corresponding supervision label is the perfect OFDM-grid channel response obtained from the channel model for the same realization. The dataset is provided as real-valued tensors with stacked real/imaginary parts and includes per-sample metadata logs (SNR, delay spread, Doppler, speed, TDL profile, and (for MIMO) correlation settings) and configuration objects to support stratified evaluation and reproducible benchmarking.
A block diagram representing the ChanEst dataset generation model is given in Figure 1.

<img width="739" height="412" alt="image" src="https://github.com/user-attachments/assets/fe84c6be-11e7-4f97-87f8-97e64a4e4397" />

**Fig 1:** System model block diagram of the ChanEst dataset generation framework.

Table 1 summarizes the default configuration used in this release and highlights parameters that are directly reconfigurable in the generator. When configuration settings change (e.g., antenna order or grid size), derived quantities such as packed channels C=2N_Tx N_Rx and tensor dimensions update automatically.

Table 1. ChanEst dataset configuration and reconfigurable parameters
| Block     | Parameter                                  | Default (this dataset)                                   |
|-----------|---------------------------------------------|-----------------------------------------------------------|
| Dataset   | Number of samples (N)                       | 10,000                                                    |
| Dataset   | Sampling                                    | Stratified                                                |
| Antenna   | N_Tx × N_Rx                                 | 1×1 (2×2, 4×4)                                            |
| Antenna   | Packed channels (C = 2·N_Tx·N_Rx)           | 2                                                         |
| NR Grid   | Subcarriers (K)                             | 612                                                       |
| NR Grid   | OFDM symbols (L)                            | 14                                                        |
| NR Grid   | Subcarrier spacing (Δf)                     | 60 kHz                                                    |
| DM-RS     | DM-RS type                                  | Type 2                                                    |
| DM-RS     | Layers (N_layers)                           | N_Tx                                                      |
| DM-RS     | CDM lengths [l_fᵐ, l_t]                     | [2, 1]                                                    |
| DM-RS     | Pilot density                                | 4.76% (408 / 8568 REs per slot, per layer)               |
| Channel   | Model                                       | 3GPP TDL Channel                                          |
| Channel   | Delay profiles                              | TDL-A…TDL-E                                               |
| Channel   | Delay spread range                          | 10–2000 ns                                                |
| Channel   | Doppler range                               | 5–5000 Hz                                                 |
| Channel   | Carrier frequency f_c                       | 7 GHz                                                     |
| Channel   | MIMO correlation                            | Low / Medium / High                                       |
| Noise     | SNR range                                   | −10 to 30 dB                                              |
| Input/Output | Input (Ĥ_in)                             | DM-RS LS/CDM (if MIMO) + 2D interpolation                |
| Input/Output | Label (H_gt)                             | Perfect OFDM-response                                     |
| Input/Output | Tensor size                              | 612 × 14 × C × N                                          |
