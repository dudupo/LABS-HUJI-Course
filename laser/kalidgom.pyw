import configparser
import logging
import logging.handlers
from math import nan
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import pandas as pd
import pyvisa
import sys
import time
import tkinter as tk
from tkinter import filedialog, font, ttk

__version__ = 1.1


# CONFIG_PATH = '.kalidgom.cfg'

TIME_HEADER = 'Time (s)'
CURRENT_HEADER = 'Current (A)'

DEFAULT_FILENAME = 'measurements.xlsx'
import platform
LOG_FILENAME = 'kalidgom.log'
if platform.system() == 'Windows':
    LOG_FILENAME = os.path.join(os.environ.get('LOCALAPPDATA'), LOG_FILENAME)
SLEEP_TIME = 0.05 # Half of minimal interval time

class App(object):
    def __init__(self, root, logger):
        super(App, self).__init__()
        self._root = root
        self._logger = logger
        self._is_measuring = False # Flag set by the measuring loop to prevent 
        self._stop_flag = False # Flag set by the stop button to notify the measuring loop to exit.

        # TODO: Use configuration file?
        # self._load_configuration()
        self._window_height = 700
        self._window_width = 800
        self._root.title('Kalidgom')

        # Configure default fonts
        default_font = font.nametofont("TkDefaultFont")
        default_font.config(size=16)
        text_font = font.nametofont("TkTextFont")
        text_font.config(size=16)

        self._rm = pyvisa.ResourceManager()         # PROD
        self._addresses = self._rm.list_resources() # PROD
        # self._addresses = ['aaa', 'bbb']          # DEBUG
        self._logger.info('Found addresses: {0}'.format(self._addresses))
        self._data = None

        self._outfile_string = tk.StringVar(value=DEFAULT_FILENAME)
        self._measurement_interval_double = tk.DoubleVar()
        self._total_runtime_int = tk.IntVar()
        self._delay_int = tk.IntVar()
        self._device_address_string = tk.StringVar(value=self._addresses[-1])

        self._top_frame = tk.Frame(self._root)
        self._top_frame.grid(row=0, column=0)
        
        self._config_frame = tk.Frame(self._top_frame)
        self._config_frame.grid(row=0, column=0)
    
        # Outfile
        self._outfile_label = tk.Label(self._config_frame, text='Save File to:')
        self._outfile_label.grid(row=1, column=0, sticky='W', padx=5, pady=2)

        self._outfile_text = tk.Entry(self._config_frame, width=32, textvariable=self._outfile_string)
        self._outfile_text.grid(row=1, column=1, sticky='W', columnspan=4, pady=2)

        self._outfile_button = tk.Button(self._config_frame, text='Browse ...', command=self._outfile_btn_command)
        self._outfile_button.grid(row=1, column=8, sticky='W', padx=5, pady=2)

        # Interval between measurements
        self._measurement_interval_label = tk.Label(self._config_frame, text='Interval between measurements (s):')
        self._measurement_interval_label.grid(row=2, column=0, sticky='W', padx=5)
        self._measurement_interval_spinbox = tk.Spinbox(self._config_frame, from_=0.1, to=100, increment=0.1, textvariable=self._measurement_interval_double)
        self._measurement_interval_spinbox.grid(row=2, column=1)

        # Run time
        self._total_runtime_label = tk.Label(self._config_frame, text='Run Time (s):')
        self._total_runtime_label.grid(row=3, column=0, sticky='W', padx=5)
        self._total_runtime_spinbox = tk.Spinbox(self._config_frame, from_=1, to=100, increment=1, textvariable=self._total_runtime_int)
        self._total_runtime_spinbox.grid(row=3, column=1)

        # Delay
        self._delay_label = tk.Label(self._config_frame, text='Delay before start (s):')
        self._delay_label.grid(row=4, column=0, sticky='W', padx=5)
        self._delay_scale = tk.Scale(self._config_frame, from_=0, to=10, orient='horizontal', variable=self._delay_int)
        self._delay_scale.grid(row=4, column=1)

        # Start Frame
        self._start_frame = tk.Frame(self._top_frame)
        self._start_frame.grid(row=0, column=1)
    
        # Device Menu
        self._device_combobox = ttk.Combobox(self._start_frame, textvariable=self._device_address_string, values=self._addresses)
        self._device_combobox.grid(row=0, column=0, columnspan=2, sticky='W', padx=5, pady=2)
        self._help_button = tk.Button(self._start_frame, text='?', command=self._device_help_btn_command)
        self._help_button.grid(row=0, column=2, sticky='W', padx=5, pady=2)
        
        # Start button
        self._start_button = tk.Button(self._start_frame, text='Start', command=self._start_btn_command)
        self._start_button.grid(row=1, column=0, sticky='W', padx=5, pady=2)
        
        # Stop button
        self._stop_button = tk.Button(self._start_frame, text='Stop', command=self._stop_btn_command)
        self._stop_button.grid(row=1, column=1, sticky='W', padx=5, pady=2)

        # Save button
        self._start_button = tk.Button(self._start_frame, text='Save', command=self._save_btn_command)
        self._start_button.grid(row=1, column=2, sticky='W', padx=5, pady=2)

        # Graph
        self._graph_frame = tk.Frame(self._root)
        self._graph_frame.grid(row=1, column=0)
        self._figure = Figure(figsize=(7, 5), dpi=100)
        self._plot_ax = self._figure.add_subplot(111)
        self._plot_ax.grid()
        self._plot_ax.set_xlabel('t (s)')
        self._plot_ax.set_ylabel('I (A)')
        self._line, = self._plot_ax.plot([], [])
        self._fig_canvas = FigureCanvasTkAgg(self._figure, master=self._graph_frame)
        self._fig_canvas.draw()
        self._fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._graph_toolbar = NavigationToolbar2Tk(self._fig_canvas, self._graph_frame)
        self._graph_toolbar.update()
        self._fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def __del__(self):
        self._rm.close()

    def _device_help_btn_command(self):
        tk.messagebox.showinfo('', 'Choose device. Should start with GPIB.')

    def _load_configuration(self):
        parser = configparser.ConfigParser()
        with open(CONFIG_PATH) as f:
            parser.read_file(f)
        self._font = font.Font()
        self._window_width = int(parser.get('GUI', 'window_width', fallback=800))
        self._window_height = int(parser.get('GUI', 'window_width', fallback=700))
 
    def _outfile_btn_command(self):
        outfile = filedialog.asksaveasfilename(initialfile=DEFAULT_FILENAME, filetypes=(('Open XML Spreadsheet', '*.xlsx'), ('Comma Delimited', '*.csv')), defaultextension='.xlsx')
        if outfile:
            self._outfile_string.set(outfile)
        return outfile

    def setup_keithley_device(self):
        # Set up sensor communication
        keith = self._rm.open_resource(self._device_address_string.get()) # connect to keithley
        self._logger.info('Device identity: {0}'.format(keith.query('*IDN?')))
        self._logger.info('Reset output: {0}'.format(keith.write('*RST')))
        return keith

    def _start_btn_command(self):
        if not self.validate_params():
            return
        self._logger.info('Starting measurement with params: device = {device}, delay = {delay}, interval = {interval}, length = {length}, output_file = "{output_file}"'.format(device=self._device_address_string.get(), delay=self._delay_int.get(), interval=self._measurement_interval_double.get(), length=self._total_runtime_int.get(), output_file=self._outfile_string.get()))
        keithley_device = self.setup_keithley_device() # PROD
        # keithley_device = None                       # DEBUG
        self.run_measurements(keithley_device)

    def run_measurements(self, keithley_device):
        self._stop_flag = False

        if self._is_measuring:
            tk.messagebox.showinfo('', 'Measurement is running, please wait or stop it.')
            return
        self._is_measuring = True

        # Delay
        time.sleep(self._delay_int.get())

        # Initialise lists into which data will be written
        self._data = {TIME_HEADER: [], CURRENT_HEADER: []}

        # Sample data
        self._measure(time.time(), True, keithley_device)

    def _measure(self, start_time, is_first, keithley_device):
        if is_first or time.time() - self._data[TIME_HEADER][-1] >= self._measurement_interval_double.get():
            measurement_time = time.time()
            # from numpy import random                             # DEBUG
            # measurement_data = str(random.logistic()) + ','      # DEBUG
            measurement_data = keithley_device.query('MEAS:CURR?') # PROD
            ind = measurement_data.find(',') # last index for current reading
            current_measurement = float(measurement_data[1:ind-1])
            self._data['Time (s)'].append(measurement_time - start_time) # Save measurement time
            self._data['Current (A)'].append(current_measurement) # Save measurement value
            self.update_graph()
        if (time.time() - start_time) < self._total_runtime_int.get() and not self._stop_flag:
            # Schedule next run
            self._root.after(int(SLEEP_TIME * 1000), self._measure, start_time, False, keithley_device)
        else:
            # Done measuring
            keithley_device.close() # PROD
            self.save_data()
            self._is_measuring = False

    def _stop_btn_command(self):
        self._logger.info('Setting stop flag.')
        self._stop_flag = True

    def get_run_settings_df(self):
        return pd.DataFrame({
            'Setting': ['Delay (s)', 'Run Time (s)', 'Measurement Interval (s)'],
            'Values': [self._delay_int.get(), self._total_runtime_int.get(), self._measurement_interval_double.get()]
            })
        
    def validate_params(self):
        if not self._is_out_file_valid():
            # Error in out file, propagate.
            return False
        # More checks?
        return True

    def _is_out_file_valid(self):
        # Check if path has been set and if not notify user and fail.
        if self._outfile_string.get() == '':
            tk.messagebox.showerror('Error', 'Please choose where to save data')
            return False
        file_extension = os.path.splitext(self._outfile_string.get())[1]
        if file_extension not in ('.csv', '.xlsx'):
            # Set default file type to Open XML Spreadsheet
            self._outfile_string.set(self._outfile_string.get() + '.xlsx')
        return True

    def update_graph(self):
        self._line.set_data(self._data[TIME_HEADER], self._data[CURRENT_HEADER])
        self._plot_ax.set_xbound(min(self._data[TIME_HEADER]), max(self._data[TIME_HEADER]))
        self._plot_ax.set_ybound(min(self._data[CURRENT_HEADER]), max(self._data[CURRENT_HEADER]))
        self._fig_canvas.draw()

    def _save_btn_command(self):
        if not self._is_out_file_valid():
            return
        if self._data is None:
            tk.messagebox.showerror('Error', 'Nothing to save yet, please run experiment before saving.')
            return
        self.save_data()
        
    def save_data(self):
        # Save Data back to the csv/xlsx file
        self._logger.debug('Trying to save data to "{path}".'.format(path=self._outfile_string.get()))
        file_extension = os.path.splitext(self._outfile_string.get())[1]
        self.get_run_settings_df().to_csv(header=False)
        if file_extension == '.csv':
            self.save_csv()
        else:
            self.save_excel()
        self._logger.debug('Data saved successfully.')

    def save_csv(self):
        if os.path.exists(self._outfile_string.get()):
            answer = tk.messagebox.askokcancel('Replace file', 'File exists, are you sure you want to overwrite it?')
            if not answer:
                self._logger.debug('Data not saved. Rename file and press the "Save" button to save data.')
                tk.messagebox.showinfo('', 'Data not saved. Rename file and press the "Save" button to save data.')
                return
            
        with open(self._outfile_string.get(), 'w') as f:
            self.get_run_settings_df().to_csv(f, header=False, index=False)
            f.write('\n\n')
            pd.DataFrame(self._data).to_csv(f, index=False)

    def save_excel(self):
        # Get session sheet name
        if not os.path.exists(self._outfile_string.get()):
            sheet_name = 'Sheet1'
            mode = 'w'
        else:
            session_dfs = pd.read_excel(self._outfile_string.get(), sheet_name=None)
            sheet_name = 'Sheet{0}'.format(len(session_dfs) + 1)
            mode = 'a'

        retry = True
        while retry:
            try:
                run_settings_df = self.get_run_settings_df()
                with pd.ExcelWriter(self._outfile_string.get(), engine='openpyxl', mode=mode) as excel_writer:
                    run_settings_df.to_excel(excel_writer, sheet_name=sheet_name, header=False, index=False)
                    pd.DataFrame(self._data).to_excel(excel_writer, sheet_name=sheet_name, index=False, startrow=len(run_settings_df) + 2)
                return # Save successful
            except PermissionError as error:
                self._logger.exception('Failed to save because file is open.')
                retry = tk.messagebox.askretrycancel('Error', 'Data cannot be saved while Excel file is opened! Please close excel and try again!')


if __name__ == '__main__':
    # Set up a specific logger with our desired output level
    logger = logging.getLogger('Kalidgom')
    logger.setLevel(logging.DEBUG)

    # Add the log message handler to the logger
    handler = logging.handlers.RotatingFileHandler(
                  LOG_FILENAME, maxBytes=1024 * 1024, backupCount=1)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logger.info('Starting Kalidgom, version = {version} (python version {major}.{minor})'.format(version=__version__, major=sys.version_info.major, minor=sys.version_info.minor))

    root = tk.Tk()
    App(root, logger)
    root.mainloop()

    logger.debug('Exiting Kalidgom')
