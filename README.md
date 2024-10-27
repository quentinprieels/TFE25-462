# TFE25-462

## Abstract

This repository contains all files related to my Master Thesis at the Université Catholique de Louvain (UCLouvain) in the field of Electrical Engineering and Computer Science. The thesis is entitled "Hardware FPGA-Acceleration of an Integrated Sensing and Communication OFDM Chain for a Wi-Fi 6 USRP-based Experimental Setup".

## Description

RADAR and communication systems both exploit the frequency spectrum for specific purposes. On the one hand, communication systems are used to transmit information between a transmitter and a receiver that are generally not co-located. On the other hand, RADAR systems are deployed to detect targets and estimate parameters such as distance, speed and angle of arrival. In recent years, more and more attention has been paid to solutions that exploit both systems jointly. These solutions are called ISAC (integrated sensing and communications). In particular, *Wi-Fi Sensing has made its appearance in the literature and in discussions to define the next 802.11bf standard*. OFDM-modulated communication signals can be used for radar localization functions.

In this context, *an experimental setup was developed at UCLouvain* during the previous academic year as part of a first dissertation on the subject. This setup is already operational and can be used to perform basic communication functions (bit transmission and detection) and radar functions (target range and speed estimation) using signals defined by the Wi-Fi 6 standard. The current setup is based on a software implementation using SDR (software-defined radio), where the data needs to be stored and post processed. This creates a big limitation however and makes real-time processing impossible.

The *goal of the thesis* is to *implement on FPGA* several blocks of the OFDM communication chain and of the radar chain to speed up the processing and target real-time data processing with the experimental setup. The different blocks that are considered are the FFT, several synchronization operations (frame synchronization, symbol timing offset, carrier frequency offset estimation), and/or the building of the RDM (Range/Doppler map). The implementation will of course be constrained by the available FPGA characteristics on the SDR setup, as well as the expected transmission and processing speed.

## Project structure

The project is structured as follows:

- `docs/`: contains some documentation related to the project and helpful guides.
- `experiments/`: contains the source code of the experiments conducted during the project and the results obtained.
- `RADCOM_library/`: contains the source code of the RADCOM library developed by UCLouvain and used in the project.
- `src/`
  - `baseline/`: contains the source code of the baseline implementation of the OFDM chain.
- `uhd`: contains the source code of the UHD library used to program the USRP devices. It is a submodule pointing to the [UHD repository](https://github.com/EttusResearch/uhd).

## Student, promotors and teaching assistants

- **Student**: [Quentin Prieels](mailto:quentin.prieels@student.uclouvain.be)
- **Promotor**: [Prof. Jérôme Louveaux](mailto:jerome.louveaux@uclouvain.be)
- **Promotor**: [Prof. Martin Andraud](mailto:martin.andraud@uclouvain.be)
- **Teaching assistant**: [Martin Willame](mailto:martin.willame@uclouvain.be)
- **Teaching assistant**: [Pol Maistriaux](mailto:pol.maistriaux@uclouvain.be)
