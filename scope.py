import pyvisa
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, List
from pathlib import Path
from datetime import datetime

VALID_CHANNELS = {
    "CH1", "CH2", "CH3", "CH4",
    "MATH",
    "REF1", "REF2", "REF3", "REF4",
}

# ============================================================
# Helper functions
# ============================================================

def get_scope_id_ethernet(ip: str = "10.0.0.10"):
    return f"TCPIP0::{ip}::inst0::INSTR"

def parse_channels(channels: Union[List[str], str]) -> List[str]:
    # Normalize to list
    if isinstance(channels, str):
        channels = [channels]
    elif not isinstance(channels, list):
        raise TypeError("channels must be a str or list[str]")

    #TODO: check that len is >=1

    # Validate elements
    invalid = [ch for ch in channels if ch not in VALID_CHANNELS]
    if invalid:
        raise ValueError(f"Invalid channel(s): {invalid}. Must be one of {sorted(VALID_CHANNELS)}")

    return channels

# ============================================================
# Acquisition
# ============================================================

def acquire_signal(
        scope_id,
        channels: list[str] | str,
    ) -> Dict[str, np.ndarray]:

    """
    Adquiere datos de forma de onda del osciloscopio (Tektronix MSO 2024B).
    
    Parámetros:
    inst_ID (str): Dirección VISA del instrumento.
    channels (list[str] | str): canal/es de donde adquirir. Posibles valores:
        CH1, CH2, CH3, CH4, MATH, REF1, REF2, REF3, and REF4
    points (int): Número de puntos a adquirir. Valor máximo 1M. Por defecto es 100k. 
    
    Returns
    -------
    dict with keys:
        t
        CH1 (optional)
        CH2 (optional)
        CH3 (optional)
        CH4 (optional)
        MATH (optional)
        REF1 (optional)
        REF2 (optional)
        REF3 (optional)
        REF4 (optional)
    """
    try:
        print(scope_id)
        channels = parse_channels(channels)
        n_points = 10000
        # Inicializar el administrador de recursos
        rm = pyvisa.ResourceManager()
        scope = rm.open_resource(scope_id)#, read_termination='\n', write_termination='\n')
        # scope.timeout = 4000
        
        prev_acq_mode = scope.query('ACQuire:MODe?')
        prev_acq_state = scope.query('ACQuire:STATE?')
        prev_acq_stopafter = scope.query('ACQuire:STOPAfter?')
        prev_trigger_mode = scope.query('TRIGger:A:MODe?')

        scope.write('ACQuire:MODe SAM; STATE 1; STOPAfter SEQ;:TRIGger:A:MODe NORMal')
        scope.write(f'DATA:ENC RPB; WIDTH 2; STARt 1; STOP {n_points}')
        scope.write(f'HORizontal:RECOrdlength {n_points}')

        # t = np.arange()

        result = {}

        # TODO: wait for triggering
        for channel in channels:
            scope.write(f'DATA:SOURCE {channel}')
            ymult = float(scope.query('WFMPRE:YMULT?'))
            yzero = float(scope.query('WFMPRE:YZERO?'))
            yoff = float(scope.query('WFMPRE:YOFF?'))
            xincr = float(scope.query('WFMPRE:XINCR?'))
            scope.write('CURVE?')
            measurement = scope.read_raw()
            measurement = np.frombuffer(measurement, dtype="<u2").astype(np.float32)
            measurement = (measurement - yoff) * ymult + yzero
            result[channel] = measurement[4:]   # TODO: fix limits

        n_points_read = len(result[channels[0]])

        result['t'] = np.arange(n_points_read) * xincr

        # input("press enter to restore") # TODO: borrar
        # TODO: NO SE HACE RESTORE SI ESTABA EN MODO RUN
        scope.write(f'ACQuire:MODe {prev_acq_mode}; ACQuire:STATE {prev_acq_state}; ACQuire:STOPAfter {prev_acq_stopafter}; TRIGger:A:MODe {prev_trigger_mode}')

        # Configurar el osciloscopio
        # scope.write('*CLS')  # Clear status
        # scope.write('DATA:START 1')

        # ACQuire:STOPAfter SEQuence
        # TRIGger:A:MODe NORMal
        # ACQuire:STATE RUN

        # points += 48 
        # if points < 1e5:
        #     scope.write('HORIZONTAL:RECORDLENGTH 100000')
        #     scope.write(f'DATA:STOP {points}')
        # elif points > 1e5 and points <= 1e6:
        #     scope.write('HORIZONTAL:RECORDLENGTH 1000000')
        #     scope.write(f'DATA:STOP {points}')
        # else:
        #     print(f"Advertencia: Se solicitaron más puntos que los disponibles. Se limitará a 1M de puntos.")
        #     scope.write('HORIZONTAL:RECORDLENGTH 1000000')
        #     scope.write(f'DATA:STOP 1000000')
        
        # # Obtener parámetros de la forma de onda
        # ymult = float(scope.query('WFMPRE:YMULT?'))
        # yzero = float(scope.query('WFMPRE:YZERO?'))
        # yoff = float(scope.query('WFMPRE:YOFF?'))
        # xincr = float(scope.query('WFMPRE:XINCR?'))
        
        # # Adquirir datos
        # scope.write('CURVE?')
        # datos = scope.read_raw()
        
        # # Procesar header y convertir datos a array
        # header_len = 2 + int(datos[1])
        # adc_wave = np.frombuffer(datos[header_len:-1], dtype=np.uint8)
        
        # # Convertir a valores reales
        # voltaje = (adc_wave - yoff) * ymult + yzero
        # tiempo = np.linspace(0, len(voltaje) * xincr, len(voltaje))
        
        # Cerrar conexión
        scope.close()
        rm.close()
        
        return result
        
    except Exception as e:
        print(f"Error en la adquisición: {str(e)}")
        return None, None
    

def load_signal(filepath: str | Path) -> dict[str, np.ndarray]:
    """
    Load signal data from a .csv or .npz file.

    Expected array (for .npz) / column (for .csv) names:
        t
        CH1 (optional)
        CH2 (optional)
        CH3 (optional)
        CH4 (optional)
        MATH (optional)
        REF1 (optional)
        REF2 (optional)
        REF3 (optional)
        REF4 (optional)

    Parameters
    ----------
    filepath : str | Path
        Path to .csv or .npz file.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
            {
                "t": np.ndarray
                "CH1": np.ndarray (optional)
                "CH2": np.ndarray (optional)
                "CH3": np.ndarray (optional)
                "CH4": np.ndarray (optional)
                "MATH": np.ndarray (optional)
                "REF1": np.ndarray (optional)
                "REF2": np.ndarray (optional)
                "REF3": np.ndarray (optional)
                "REF4": np.ndarray (optional)
            }

    Raises
    ------
    ValueError
        If file extension is unsupported or required columns are missing.
    """

    path = Path(filepath).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    data = {}

    if path.suffix == ".npz":
        with np.load(path) as data:
            if "t" not in data:
                raise ValueError(f"Missing column 't' in .csv ({path})")
            _ = parse_channels([k for k in data if k != "t"]) # throws exception if wrong column is present
            return {k: data[k] for k in data}

    elif path.suffix == ".csv":
        data = np.genfromtxt(path, delimiter=",", names=True)
        if "t" not in data.dtype.names:
            raise ValueError(f"Missing array 't' in .npz ({path})")
        _ = parse_channels([k for k in data.dtype.names if k != "t"]) # throws exception if wrong column is present
        return {name: data[name] for name in data.dtype.names}

    else:
        raise ValueError(f"Unsupported file type when loading spectrum: {path.suffix}")



# ============================================================
# Saving
# ============================================================

def save_signal(
    data: Dict[str, np.ndarray],
    name: str = "scope",
    timestamp: str | None = None,
    outdir: Path | str = Path("out"),
    save_csv: bool = True,
    save_npz: bool = True,
):
    """
    Save acquisition as BOTH NPZ and CSV using a shared timestamp.

    Returns
    -------
    dict with filenames of files saved. If no file was saved, returns empty dictionary
        possible keys: ["npz", "csv"].
    """

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d%Hh%Mm%Ss")

    base = f"{name}_{timestamp}"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    filenames = {}

    # --- NPZ ---
    if save_npz:
        npz_filename = outdir / f"{base}.npz"
        np.savez(npz_filename, **data)
        filenames['npz'] = npz_filename

    # --- CSV ---
    if save_csv:
        csv_filename = outdir / f"{base}.csv"
        stacked = np.column_stack(list(data.values()))
        header = ",".join(data)
        np.savetxt(csv_filename, stacked, delimiter=",", header=header, comments="")
        filenames['csv'] = csv_filename

    return filenames

# ============================================================
# Plotting
# ============================================================

def plot_signal(
    signal: Dict[str,np.ndarray],
    y_axis: List[str] | None = None,
    x_axis: str = 't',
    show: bool = True,
    save_path: str | None = None,
    labels: Dict[str, np.ndarray] = {},
    xmin: float = 0, 
    xmax: float = np.inf,
    xlabel: str = 't[s]',
    ylabel: str = 'V[V]',
):
    """
    Plot one or more channels from a signal dictionary.

    Parameters
    ----------
    signal : Dict[str, np.ndarray]
        Dictionary containing the signal data. Must include a field
        corresponding to `x_axis` (default: 't') and one or more fields for the
        y-axis channels.
    y_axis : List[str] | None, optional
        List of field names to plot on the y-axis. If None, all fields except
        't' are plotted.
    x_axis : str, optional
        Field name to use as the x-axis. Default is 't'.
    show : bool, optional
        Whether to display the plot using `plt.show()`. Default is True.
    save_path : str | None, optional
        If provided, the figure is saved to this path.
    labels : Dict[str], optional
        Mapping from channel names to display labels. Missing entries default
        to the channel name.
    xmin : float, optional
        Minimum x value for filtering the data. Default is 0.
    xmax : float, optional
        Maximum x value for filtering the data. Default is np.inf.
    xlabel : str, optional
        Label for the x-axis. Default is 't[s]'.
    ylabel : str, optional
        Label for the y-axis. Default is 'V[V]'.

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib figure.

    Notes
    -----
    - The function filters the signal data to the range [xmin, xmax].
    """
    X = signal[x_axis]

    mask = (X >= xmin) & (X <= xmax)
    
    X = X[mask]
    signal_masked = {k: signal[k][mask] for k in signal}

    if not y_axis:
        y_axis = [k for k in signal_masked if k != x_axis]

    Y = [signal_masked[ch_name] for ch_name in y_axis]

    labels_ = labels.copy()

    for ch_name in y_axis:
        labels_.setdefault(ch_name, ch_name)

    fig = plt.figure()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style='plain')

    for y,ch_name in zip(Y, y_axis):
        ax.plot(X, y, linewidth=0.8, label=labels_[ch_name])

    ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    
    return fig


if __name__ == '__main__':
    result = acquire_signal(get_scope_id_ethernet("10.0.0.10"), ['CH1', 'CH2'])
    save_signal(result, name="soa_I_60mA")
    plot_signal(result)
    # for path in ('out/scope_2026040618h36m31s.csv', 'out/scope_2026040618h36m31s.npz'):
    #     result = load_signal(path)
    #     plot_signal(result, labels={'CH1':'channel 1'}, xmax=0.0005)



