# References

논문 PDF는 저장소에 포함하지 않고 DOI 링크와 프로젝트에서 참고한 부분만 정리합니다.

## SNN decoder baseline

L. Martis, G. Leone, L. Raffo, and P. Meloni, “Low-Power FPGA-Based Spiking Neural Networks for Real-Time Decoding of Intracortical Neural Activity,” *IEEE Sensors Journal*, vol. 24, no. 24, 2024.

- DOI: https://doi.org/10.1109/JSEN.2024.3487021
- 참고 범위: SNN topology, target preprocessing, split, training protocol, evaluation metrics, FPGA 수치 정밀도
- fixed-point 재현 범위: 공개 checkpoint에 observer와 모든 fixed-point 소수 비트 정보가 없어 명시적인 scale 가정을 사용하며, 논문 하드웨어를 bit-exact하게 재현하지 않음

## Channel selection

G. Leone, L. Martis, L. Raffo, and P. Meloni, “Enabling SNN-Based Near-MEA Neural Decoding with Channel Selection: An Open-HW Approach,” *DATE 2025*.

- DOI: https://doi.org/10.23919/DATE64628.2025.10993220
- 참고 범위: firing-rate/behavior Pearson correlation, training 앞 절반을 이용한 calibration, 64-channel operating point
- 재현 가정: 논문에 여러 behavioral axis의 correlation을 하나의 순위로 합치는 방법이 명시되지 않아 RMS magnitude를 사용

## Dataset

J. E. O’Doherty et al., “Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology,” Zenodo dataset, 2020.

- DOI: https://doi.org/10.5281/zenodo.3854034
- 사용 범위: Indy session recording과 electrode label

공개 전에는 논문 원문과 공식 서지정보를 다시 대조하고, 저장소의 실제 구현이 각 참고문헌에서 직접 가져온 코드인지 아이디어만 참고한 것인지 구분해 표시해야 합니다.
