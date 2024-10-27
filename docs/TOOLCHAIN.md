# Tool chain used in this TFE

The tool chain used in this TFE is inspired by the one [proposed by Ettus](https://files.ettus.com/manual/md_usrp3_build_instructions.html) to program their USRP devices. It is part of the [USRP Hardware Driver and USRP Manual (UHD)](https://files.ettus.com/manual/index.html). See the [UHD repository](#uhd) section of this document for more information about the installation of the UHD library.

In this TFE, the USRP device used is the [USRP-2944R](https://www.ni.com/fr-be/shop/model/usrp-2944.html) in the NI nominal. It is called the [X310 + UBX (x2)](https://www.ettus.com/all-products/usrp-x310/) in the Ettus nominal. Ettus presents it as a high-performance, scalable software defined radio (SDR) platform for designing and deploying next generation wireless communications systems.

:information_source: **About NI and Ettus**: Ettus Research, a subsidiary of National Instruments (NI), specializes in software-defined radios (SDRs) and developed the USRP product line, widely used in academic and industry research. While Ettus focuses on creating flexible SDR hardware for research and development, NI incorporates these technologies into broader test and measurement solutions, enabling more comprehensive applications across industries such as telecommunications, defense, and aerospace. This connection allows for advanced use cases in both open-source SDR environments and with NI’s software platforms, like LabVIEW.

## Prerequisites

1. Download the required packages: `python3`, `bash`, `make` and `doxygen`. You can install them using the following command, depending on your Linux distribution:

    **Note**: The following commands are for Ubuntu, Fedora and Arch Linux. If you are using another distribution, please refer to the documentation of your distribution to install the required packages.

    For **Ubuntu**:

    ```bash
    sudo apt-get install python3 bash make doxygen
    ```

    For **Fedora**:

    ```bash
    sudo dnf install python3 bash make doxygen
    ```

    For **Arch Linux**:

    ```bash
    sudo pacman -S python3 bash make doxygen
    ```

2. Before continuing the installation of the tool chain, you are advised to complete the [RADCOM library installation](RADCOM.md) to setup the python environment that will be used by the tool chain.

## Vivado

Vivado is the software used to program the FPGA located on the USRP devices. It is a proprietary software developed by Xilinx. Before installing it, make sure to meet the requirements for the installation:

- **Disk Space**: 100 GB.
- **RAM**: 32 GB.

Then, follow the steps below to install Vivado:

1. Download Vivado from [Xilinx website](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools.html)

2. Install Vivado by following the instructions provided by Xilinx.

    1. **Tips** : To save disk space and installation time, you can install oµnly the support for the FPGAs you intend to use. (i.e. Kintex-7 to work with the USRP used in this TFE)

    2. The recommended installation directory is `/opt/Xilinx/`. Change it when you are asked for the installation directory.

        **Note**: if the warning *Cannot write to /opt/Xilinx/ Check the read/write permissions* appears, create the directory `/opt/Xilinx/` and give the write permissions to the user. To do so, run the following commands:

        ```bash
        sudo mkdir /opt/Xilinx/
        sudo chown $USER:$USER /opt/Xilinx/
        ```

        and re-select the directory `/opt/Xilinx/` in the installation process.

## UHD

The UHD library (USRP Hardware Driver) is used to communicate with the USRP devices. To install UHD, follow the steps below:

If you cloned the current repository, then you have normally already installed the UHD library, because it is a submodule of the repository. If you do not find this submodule, try to fetch it by running the following command:

```bash
git submodule update --init --recursive
```

If you do not want to use the submodule, you can install the UHD library by cloning it from the official repository, available on GitHub [here](https://github.com/EttusResearch/uhd):

```bash
git clone git@github.com:EttusResearch/uhd.git
```
