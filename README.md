# ChanEst-Dataset-Generation
A Reproducible and Reconfigurable Dataset Generation Framework for Deep Learning–Based 6G Channel Estimation.

ChanEst is a reproducible and reconfigurable dataset for deep learning–based channel estimation generated using standard-compliant 3GPP procedures in MATLAB (6G Exploration Library). Each sample is created by inserting PDSCH DM-RS pilots into an OFDM resource grid, transmitting through a randomized 3GPP TDL channel realization, adding AWGN at a randomized SNR, and applying a receiver-aligned pipeline that performs pilot extraction, LS estimation on DM-RS, and 2-D time–frequency interpolation to form a dense-grid learning input. The corresponding supervision label is the perfect OFDM-grid channel response obtained from the channel model for the same realization. The dataset is provided as real-valued tensors with stacked real/imaginary parts and includes per-sample metadata logs (SNR, delay spread, Doppler, speed, TDL profile, and (for MIMO) correlation settings) and configuration objects to support stratified evaluation and reproducible benchmarking.
A block diagram representing the ChanEst dataset generation model is given in Figure 1.

<img width="739" height="412" alt="image" src="https://github.com/user-attachments/assets/fe84c6be-11e7-4f97-87f8-97e64a4e4397" />

**Fig 1:** System model block diagram of the ChanEst dataset generation framework.
