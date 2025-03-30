import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Radcomlib imports
from radcomlib.comm_toolbox import symbol_mapping


class Frame:
    """
    This class represents an OFDM frame. 
    
    Frame structure:
    
    An OFDM frame is composed of a preamble and a payload.
    The preamble is composed of two symbols: a timing synchronization symbol used to estimate the
    STO and a equalization symbol used to estimate the channel. The payload is composed of N OFDM
    symbols. The frame is generated with random bits.
    
    |<-------------------------------------- OFDM frame ------------------------------------->|
    |<------------ Preamble ----------->|<--------------------- Payload --------------------->|
    +-----------------+-----------------+-----------------+-------...-------+-----------------+
    | Timing preamble | Equalization pr |  OFDM symbol 1  |       ...       |  OFDM symbol N  |
    +-----------------+-----------------+-----------------+-------...-------+-----------------+
    
    Payload structure:
    
    The payload contains data and pilots symbols. Pilots symbols are used to estimate the channel
    to perform radar detection. Nt and Nf parameters define the pilot spacing in time and frequency
    domain. In the example below, Nt=1 and Nf=3. The first and the last subcarrier is always included.
    
    |<------------- Subarriers ------------>|
    +---+---+---+---+---+---+---+---+---+---+
    | P | D | D | P | D | D | P | D | D | P |  |
    +---+---+---+---+---+---+---+---+---+---+  |
    | P | D | D | P | D | D | P | D | D | P | time
    +---+---+---+---+---+---+---+---+---+---+  |
    | P | D | D | P | D | D | P | D | D | P |  |
    +---+---+---+---+---+---+---+---+---+---+  v
    """
    
    _bits_per_fsymbol = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "16PSK": 4}
    
    def __init__(self, K: int = 1024, CP: int = 128, CP_preamble: int = 128, 
                 M: int = 5, N: int = 10, 
                 preamble_mod: str = "QPSK", payload_mod: str = "QPSK",
                 Nt: int = 1, Nf: int = 1,
                 random_seed: int = None, verbose: bool = False
        ) -> None:       
        """
        Initialize a SchmidlAndCoxFrame.
        
        Parameters:
        - K: Number of subcarriers                                          [# of samples] >= 1
        - CP: Cyclic prefix length for payload symbols                      [# of samples] >= 0
        - CP_preamble: Cyclic prefix length for preamble symbol             [# of samples] >= 0
        - M: Oversampling factor (entire frame)                             [# of samples] >= 1
        - N: Number OFDM symbols in the payload                             [# of samples] >= 1
        - preamble_mod: Modulation scheme for the preamble                  [BPSK, QPSK, 16QAM, 16PSK]
        - payload_mod: Modulation scheme for the payload                    [BPSK, QPSK, 16QAM, 16PSK]
        - Nt: Pilot spacing in time domain (applies only on payload)        [# of symbols] >= 1
        - Nf: Pilot spacing in frequency domain  (applies only on payload)  [# of subcarriers] >= 1
        - random_seed: Random seed for the generator                        [int]
        - verbose: Print information                                        [bool]
        """   
        # Arguments validation check
        if K < 1 or CP < 0 or CP_preamble < 0 or M < 1 or N < 1 or Nt < 1 or Nf < 1:
            raise ValueError("Invalid frame parameters")
        if preamble_mod not in self._bits_per_fsymbol or payload_mod not in self._bits_per_fsymbol:
            raise ValueError("Invalid modulation scheme")
        if Nt > N or Nf > K:
            raise ValueError("Invalid pilot spacing")
            
        # Randomness control
        if random_seed is None:
            random_seed = np.random.default_rng().integers(0, 2**32)
            if verbose:
                print(f"Frame random seed: {random_seed}")
        self.generator = np.random.default_rng(random_seed)
        
        # Parameters
        self.K = K
        self.CP = CP
        self.CP_preamble = CP_preamble
        self.M = M
        self.N = N
        self.preamble_mod = preamble_mod
        self.payload_mod = payload_mod
        self.Nt = Nt
        self.Nf = Nf
        self.len = (2 * (self.CP_preamble + self.K) * self.M) + (self.N * (self.CP + self.K) * self.M)
        
        # Frame data
        self.pilots_idx_t_mesh, self.pilots_idx_f_mesh = self.generate_pilots_grid()
        self.fpreamble = self.generate_preamble()
        self.fpayload, self.bits = self.generate_payload()
        
        # Actual frame
        self.frame = None # To be used by the OFDM system (via setter and getter)
        
        self.verbose = verbose
        
    def set_frame(self, frame: np.ndarray) -> None:
        """
        Set the frame data.
        """
        self.frame = frame


    def get_frame(self) -> np.ndarray:
        """
        Get the frame data.
        """
        if self.frame is None:
            raise ValueError("Frame data is not set")
        return self.frame
        

    def generate_pilots_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a pilot grid: a matrix to represent where pilots symbols are located.
        Note: the first and the last subcarrier is always included in the pilots, same
        for first and last OFDM symbol. This ensure that we not perform any extrapolation.
        """
        pilots_idx_f = np.concatenate((np.arange(0, self.K - 1, self.Nf), [self.K - 1]))
        pilots_idx_t = np.concatenate((np.arange(0, self.N - 1, self.Nt), [self.N - 1]))
        pilots_idx_t_mesh, pilots_idx_f_mesh = np.meshgrid(pilots_idx_t, pilots_idx_f)
        return pilots_idx_t_mesh, pilots_idx_f_mesh
    
    def generate_symbol(self, mod: str, bits: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate an OFDM symbol based on the modulation scheme.
        If bits are not provided, generate random bits.
        This function generates a column vector of the (time x subcarriers) matrix.
        
        Returns:
        - fsymbol: The generated OFDM frequency domain symbol
        - bits: The bits used to generate the symbol
        """
        if mod not in self._bits_per_fsymbol:
            raise ValueError(f"Invalid modulation scheme: {mod}")
        
        if bits is None:
            n_bits = self._bits_per_fsymbol[mod] * self.K
            bits = self.generator.integers(0, 2, n_bits)
        
        elif len(bits) != self._bits_per_fsymbol[mod] * self.K:
            raise ValueError(f"Invalid number of bits for {mod} modulation")
            
        fsymbol = symbol_mapping(bits, mod)
        return fsymbol, bits # Shape: (K,), (K * bits_per_fsymbol,)


    def generate_preamble(self) -> tuple[np.ndarray]:
        """
        Generate the preamble symbol.
        Odd subcarriers are set to 0, even contains random bits.
        
        Returns:
        - fsymbol: The generated OFDM frequency domain preamble symbol
        """
        # Timing synchronization preamble
        fsymbol_t, _ = self.generate_symbol(self.preamble_mod)
        fsymbol_t[1::2] = 0 # Set odd subcarriers to 0
        fsymbol_t = np.array([fsymbol_t])
        
        # Equalization preamble
        fsymbol_e, _ = self.generate_symbol(self.payload_mod) # Must estimate channel for payload
        fsymbol_e = np.array([fsymbol_e])
        return np.concatenate([fsymbol_t, fsymbol_e]) # Shape: (2, K)


    def generate_payload(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        Generate the payload symbols.
        
        Returns:
        - fsymbols: Matrix (N x K) containing the generated OFDM frequency domain symbols
        - bits: Matrix (N x K) containing the bits used to generate the symbols
        """
        fsymbols = []
        bits = []
        for _ in range(self.N):
            fsymbol, bits_ = self.generate_symbol(self.payload_mod)
            fsymbols.append(fsymbol)
            bits.append(bits_)
        return np.array(fsymbols), np.array(bits) # Shape: (N, K), (N, K * bits_per_fsymbol)

    
    def get_pilots(self) -> np.ndarray:
        """
        Return the pilots symbols.
        Pilots symbols only exist in the payload.
        """
        return self.fpayload[self.pilots_idx_t_mesh.T, self.pilots_idx_f_mesh.T]
    
    
    def save(self, filename: str) -> None:
        """
        Save the frame to a .txt file in a I/Q format, each line contains a sample.
        The file is saved in the following format:
        I0
        Q0
        I1
        Q1
        ...
        """
        signal_len = self.len
        
        # Create I/Q signal
        split_signal = np.zeros((signal_len * 2))
        split_signal[0::2] = np.real(self.frame)
        split_signal[1::2] = np.imag(self.frame)
        
        # Normalize the signal
        # Ensure norm < 1
        split_signal = split_signal / np.max(np.abs(split_signal)) * 0.7      
        np.savetxt(filename, split_signal)
    
    
    def load(self, filename: str) -> None:
        """
        Load a file containing a I/Q signal. The file has the same format as
        the save function.        
        """
        with open(filename, "rb") as file:
            data = np.fromfile(file, dtype=np.float32)
            data = data.astype(np.complex64)
            
            rx_sig = data[0::2] + 1j * data[1::2]
            rx_sig.reshape(-1, 1)
            rx_sig = np.squeeze(rx_sig)
        
        # Check the signal length
        if len(rx_sig) != self.len:
            print(f"CAUTION: Invalid signal length: expected {self.len}, got {len(rx_sig)}\nOverwriting the length...")
            self.len = len(rx_sig)
        
        # Create the complex signal
        self.frame = rx_sig
        
    
    def plot(self, bits: bool = False) -> None:
        """
        Plot the pilot matrix: the signal in a (time x subcarrier) matrix.
        - bits: If True, fill the matrix with the bits used to generate the symbols.
        """
        # Create the matrix of symbol types
        matrix = np.zeros((2 + self.N, self.K))
        matrix[0, :] = 3  # Timing synchronization preamble
        matrix[1, :] = 2  # Equalization preamble
        matrix[self.pilots_idx_t_mesh.T + 2, self.pilots_idx_f_mesh.T] = 1  # Pilots
        
        # Create the matrix of bits sent
        bits_per_symbol = self._bits_per_fsymbol[self.payload_mod]
        bits_matrix = np.zeros((2 + self.N, self.K * bits_per_symbol), dtype=int)
        bits_matrix[2:, :] = self.bits
        
        # Reshape and convert to binary format
        bits_matrix = bits_matrix.reshape((2 + self.N, self.K, bits_per_symbol))
        binary_matrix = np.array([
            ["".join(map(str, bits_matrix[i, j])) for j in range(self.K)]
            for i in range(2 + self.N)
        ])
        
        binary_matrix[0, :] = ""  # Preamble has no bits
        binary_matrix[1, :] = ""  # Preamble has no bits
        
        # Plot
        plt.figure("Pilots matrix", figsize=(10, 8))
        plt.title("Pilots matrix", fontsize=14, fontweight="bold")
        sns.heatmap(matrix, 
                    cmap=sns.color_palette(["tab:blue", "tab:red", "tab:gray", "black"]), 
                    cbar_kws={"ticks": [0, 1, 2, 3], "format": "%d"}, 
                    linewidths=0.5, 
                    linecolor="black", 
                    alpha=0.5, 
                    annot=binary_matrix if bits else None, 
                    fmt="")
        colorbar = plt.gca().collections[0].colorbar
        colorbar.set_ticks([0.375, 1.125, 1.875, 2.625])
        colorbar.set_ticklabels(["data", "pilots", "Equalization preamble", "Timing sync. preamble"])
        plt.ylabel("OFDM symbols (time)")
        plt.xlabel("Subcarriers (frequency)")
        plt.tight_layout()
